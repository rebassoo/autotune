"""
End-to-end pipeline:
  preprocessing → k-fold surrogate evaluation → surrogate training → optimization.

All preprocessing is done in memory; nothing is written between stages.
Results (optimization CSV) are written to cfg.paths.output_dir.

Usage:
    python scripts/run_end_to_end.py --config configs/scream_autocal.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import load_config
from autotune_gp.backend import get_backend
from autotune_gp.transforms import fit_transform_X, fit_transform_Y, transform_obs
from autotune_gp.io import split_zrg_obs
from autotune_gp.gp import GPWrapper
from autotune_gp.cost import zrg_cost_function_mae_weighted
from autotune_gp.optimize import optimize_parallel

from autotune_gp.evaluate import run_kfold_evaluation

from preprocessing.pipeline import (
    build_run_list,
    load_and_mask,
    compute_zrg,
    stack_all_data,
    make_folds,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg.preprocess is None:
        raise ValueError("Config must include a [preprocess] section for this script.")

    pp = cfg.preprocess
    var_names = list(pp.variables.keys())

    # ------------------------------------------------------------------ #
    # Stage 1: Preprocessing                                               #
    # ------------------------------------------------------------------ #
    print("=== Stage 1: Build run list ===")
    run_list = build_run_list(
        params_json=pp.params_json,
        DY1_sim_dir=pp.DY1_sim_dir,
        DY1_nc_suffix=pp.DY1_nc_suffix,
        DY2_sim_dir=pp.DY2_sim_dir,
        DY2_nc_suffix=pp.DY2_nc_suffix,
    )

    print("=== Stage 1: Load simulations and observations ===")
    sim_data = load_and_mask(
        run_list=run_list,
        DY1_obs_dir=pp.DY1_obs_dir,
        DY2_obs_dir=pp.DY2_obs_dir,
        variables=pp.variables,
    )

    print("=== Stage 1: Compute ZRG averages ===")
    zrg_result = compute_zrg(
        sim_names=run_list["sim_names"],
        ppe_dataset_small=sim_data["ppe_dataset_small"],
        obs_dict=sim_data,
        control_file=pp.control_file,
        regions_file=pp.regions_file,
        variables=pp.variables,
    )

    print("=== Stage 1: Stack training arrays ===")
    X_train, Y_train_ZRG = stack_all_data(zrg_result, run_list["ppe_params"],
                                          var_names=var_names)

    # Build obs ZRG parts (one DataFrame per variable, columns as strings)
    zrg_obs   = zrg_result["zrg_obs"]
    obs_parts = split_zrg_obs(zrg_obs, n_vars=len(var_names))

    # ------------------------------------------------------------------ #
    # Stage 2: K-fold surrogate evaluation                                 #
    # ------------------------------------------------------------------ #
    print("=== Stage 2: K-fold surrogate evaluation ===")
    folds = make_folds(zrg_result, run_list["ppe_params"], var_names=var_names)
    run_kfold_evaluation(folds, train_gp=cfg.runtime.train_gp)

    # ------------------------------------------------------------------ #
    # Stage 3: Normalise (on full dataset for final surrogate)             #
    # ------------------------------------------------------------------ #
    print("=== Stage 3: Normalise (full dataset) ===")
    param_bounds = np.array([cfg.optimize.bounds["low"], cfg.optimize.bounds["high"]])
    X_sc,      X_train_norm   = fit_transform_X(X_train, param_bounds=param_bounds)
    Y_scalers, Y_train_norm   = fit_transform_Y(Y_train_ZRG)
    obs_norm = transform_obs(obs_parts, Y_scalers)

    # ------------------------------------------------------------------ #
    # Stage 4: Train surrogate on full dataset                             #
    # ------------------------------------------------------------------ #
    print("=== Stage 4: Train GP surrogate (full dataset) ===")
    gp = GPWrapper(X_train_norm, Y_train_norm)
    if cfg.runtime.train_gp:
        gp.train()

    # ------------------------------------------------------------------ #
    # Stage 5: Optimize                                                    #
    # ------------------------------------------------------------------ #
    print("=== Stage 5: Optimize ===")
    backend          = get_backend(cfg.runtime.backend, cfg.runtime.device)
    n_regions        = len(cfg.data.regions_list)
    n_zonal          = int(cfg.data.n_zonal)
    var_w            = cfg.weights.variables
    zrg_w            = cfg.weights.zrg
    dy_w             = cfg.weights.dy
    zonal_weights    = cfg.weights.zonal_weights
    regional_weights = cfg.weights.regional_weights

    # Build parameter ordering constraint index pairs from column names
    param_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    constraint_pairs = []
    for pair in (cfg.optimize.param_ordering_constraints or []):
        low_name, high_name = pair
        if param_names and low_name in param_names and high_name in param_names:
            constraint_pairs.append((param_names.index(low_name), param_names.index(high_name)))

    def cost_fn(x):
        x = np.asarray(x, dtype=float)
        for low_idx, high_idx in constraint_pairs:
            violation = x[low_idx] - x[high_idx]
            if violation > 0:
                return 1e2 + 1e2 * violation
        m, _ = gp.predict(x)
        return zrg_cost_function_mae_weighted(
            m, obs_norm, var_w, zrg_w, dy_w,
            n_zonal=n_zonal, n_regions=n_regions, backend=backend,
            zonal_weights=zonal_weights, regional_weights=regional_weights,
        )

    _, _, csv_path = optimize_parallel(
        cost_fn=cost_fn,
        n_params=cfg.optimize.n_params,
        bounds_low=cfg.optimize.bounds["low"],
        bounds_high=cfg.optimize.bounds["high"],
        seed=cfg.optimize.seed,
        n_xstarts=cfg.optimize.n_xstarts,
        niter=cfg.optimize.niter,
        method=cfg.optimize.method,
        out_dir=cfg.paths.output_dir,
        max_workers=cfg.optimize.max_workers,
    )

    print(f"Done. Results: {csv_path}")


if __name__ == "__main__":
    main()
