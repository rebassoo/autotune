"""
Two-stage pipeline: preprocessing → save to disk → evaluation → surrogate → optimization.

Stage 1 writes pickles to cfg.paths.preprocess_dir:
    obs.pkl        {'zrg_obs': pd.DataFrame}         (readable by load_obs)
    gp_proj.pkl    {'X_train': DataFrame,             (readable by load_gp_proj)
                    'Y_train': ndarray}
    scalers.pkl    {'X_pipeline': MinMaxScaler,       fitted scalers (physical → ML units)
                    'Y_pipeline_<VAR>': StandardScaler, ...}
    kfold_data.pkl {'folds': list}                   (k-fold splits for evaluation)
    run_list.pkl   (intermediate)
    zrg_data.pkl   (intermediate; per-variable ZRG DataFrames)

Stage 2 loads those pickles and runs:
    k-fold surrogate evaluation (R² / RMSE per variable)
    → train final surrogate on full dataset
    → optimize

You can re-run stage 2 alone (skipping slow preprocessing) with --stage 2.

Usage:
    python scripts/run_two_stage.py --config configs/scream_autocal.yaml
    python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 1
    python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 2
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import load_config
from autotune_gp.backend import get_backend
from autotune_gp.io import load_obs, load_gp_proj, split_zrg_obs
from autotune_gp.transforms import fit_transform_X, fit_transform_Y, transform_obs
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


def run_stage1(cfg):
    """Preprocessing: build ZRG arrays and save to preprocess_dir."""
    pp = cfg.preprocess
    if pp is None:
        raise ValueError("Config must include a [preprocess] section for stage 1.")

    var_names = list(pp.variables.keys())
    out_dir = Path(cfg.paths.preprocess_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: Build run list ===")
    run_list = build_run_list(
        params_json=pp.params_json,
        DY1_sim_dir=pp.DY1_sim_dir,
        DY1_nc_suffix=pp.DY1_nc_suffix,
        DY2_sim_dir=pp.DY2_sim_dir,
        DY2_nc_suffix=pp.DY2_nc_suffix,
    )
    with open(out_dir / "run_list.pkl", "wb") as f:
        pickle.dump(run_list, f)
    print(f"  Saved run_list.pkl  ({len(run_list['sim_names'])} runs)")

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
    with open(out_dir / "zrg_data.pkl", "wb") as f:
        pickle.dump(zrg_result, f)
    print("  Saved zrg_data.pkl")

    print("=== Stage 1: Stack training arrays ===")
    X_train, Y_train_ZRG = stack_all_data(zrg_result, run_list["ppe_params"],
                                          var_names=var_names)

    print("=== Stage 1: Build k-fold splits ===")
    folds = make_folds(zrg_result, run_list["ppe_params"], var_names=var_names)
    with open(out_dir / "kfold_data.pkl", "wb") as f:
        pickle.dump({"folds": folds}, f)
    print(f"  Saved kfold_data.pkl  ({len(folds)} folds)")

    # Save in the format expected by load_obs / load_gp_proj
    with open(out_dir / "obs.pkl", "wb") as f:
        pickle.dump({"zrg_obs": zrg_result["zrg_obs"]}, f)
    with open(out_dir / "gp_proj.pkl", "wb") as f:
        pickle.dump({"X_train": X_train, "Y_train": Y_train_ZRG}, f)

    print(f"  Saved obs.pkl  (zrg_obs shape: {zrg_result['zrg_obs'].shape})")
    print(f"  Saved gp_proj.pkl  (X: {X_train.shape}, Y: {Y_train_ZRG.shape})")

    print("=== Stage 1: Fit and save scalers ===")
    phys = cfg.optimize.param_physical_bounds
    if phys and hasattr(X_train, "columns"):
        param_bounds_arr = np.array([phys[col] for col in X_train.columns])
        X_sc, _ = fit_transform_X(X_train, param_bounds=param_bounds_arr.T)
    else:
        X_sc, _ = fit_transform_X(X_train)
    Y_scalers, _ = fit_transform_Y(Y_train_ZRG)
    scalers_dict = {
        "X_pipeline": X_sc,
        **{f"Y_pipeline_{v}": Y_scalers[i] for i, v in enumerate(var_names)},
    }
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers_dict, f)
    print(f"  Saved scalers.pkl  (X: MinMaxScaler on bounds, Y: StandardScaler x{len(var_names)})")
    print(f"Stage 1 complete. All outputs in: {out_dir}")


def run_stage2(cfg):
    """K-fold evaluation + surrogate training + optimization.

    Loads from cfg.paths.preprocess_dir if stage 1 has been run there;
    otherwise falls back to cfg.paths.obs_pkl / cfg.paths.gp_proj_pkl.
    K-fold evaluation is skipped if kfold_data.pkl is not found.
    """
    out_dir = Path(cfg.paths.preprocess_dir)
    kfold_pkl = out_dir / "kfold_data.pkl"

    # Prefer preprocess_dir outputs; fall back to cfg.paths (reference pkls)
    obs_pkl_candidate     = out_dir / "obs.pkl"
    gp_proj_pkl_candidate = out_dir / "gp_proj.pkl"
    obs_pkl     = str(obs_pkl_candidate)     if obs_pkl_candidate.exists()     else cfg.paths.obs_pkl
    gp_proj_pkl = str(gp_proj_pkl_candidate) if gp_proj_pkl_candidate.exists() else cfg.paths.gp_proj_pkl

    print(f"=== Stage 2: Load data ===")
    print(f"  obs:     {obs_pkl}")
    print(f"  gp_proj: {gp_proj_pkl}")

    obs_loaded  = load_obs(obs_pkl)
    zrg_obs     = obs_loaded["zrg_obs"]
    obs_parts   = split_zrg_obs(zrg_obs, n_vars=len(cfg.data.variables))

    gp_loaded   = load_gp_proj(gp_proj_pkl)
    X_train     = gp_loaded["X_train"]
    Y_train_ZRG = gp_loaded["Y_train"]

    print("=== Stage 2: K-fold surrogate evaluation ===")
    if not kfold_pkl.exists():
        print(f"  kfold_data.pkl not found in {out_dir} — skipping k-fold evaluation.")
        print("  (Run stage 1 first to generate k-fold splits.)")
    else:
        with open(kfold_pkl, "rb") as f:
            kfold_data = pickle.load(f)
        run_kfold_evaluation(kfold_data["folds"], train_gp=cfg.runtime.train_gp)

    print("=== Stage 2: Normalise (full dataset) ===")
    var_names = cfg.data.variables
    # Try to load saved scalers: first from gp_proj_pkl, then from scalers.pkl
    X_sc   = gp_loaded.get("X_pipeline")
    Y_scalers = [gp_loaded.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is None or not all(s is not None for s in Y_scalers):
        scalers_pkl_path = out_dir / "scalers.pkl"
        if scalers_pkl_path.exists():
            with open(scalers_pkl_path, "rb") as f:
                saved = pickle.load(f)
            X_sc      = saved.get("X_pipeline")
            Y_scalers = [saved.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is not None and all(s is not None for s in Y_scalers):
        X_train_norm  = X_sc.transform(X_train)
        Y_train_norm  = np.stack(
            [Y_scalers[j].transform(Y_train_ZRG[:, :, j]) for j in range(len(var_names))],
            axis=0).transpose(1, 2, 0)
        obs_norm = transform_obs(obs_parts, Y_scalers)
        print("  Using saved scalers (exact match with training)")
    else:
        print("  Warning: no saved scalers found — refitting from scratch")
        phys = cfg.optimize.param_physical_bounds
        if phys and hasattr(X_train, "columns"):
            param_bounds = np.array([phys[col] for col in X_train.columns])
            X_sc, X_train_norm = fit_transform_X(X_train, param_bounds=param_bounds.T)
        else:
            X_sc, X_train_norm = fit_transform_X(X_train)
        Y_scalers, Y_train_norm = fit_transform_Y(Y_train_ZRG)
        obs_norm = transform_obs(obs_parts, Y_scalers)

    print("=== Stage 2: Train GP surrogate (full dataset) ===")
    gp = GPWrapper(X_train_norm, Y_train_norm)
    if cfg.runtime.train_gp:
        gp.train(tf_determinism=cfg.runtime.tf_determinism)

    print("=== Stage 2: Optimize ===")
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

    results, top_rows, csv_path = optimize_parallel(
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

    if cfg.diagnostics.enabled:
        from autotune_gp.diagnostics import run_diagnostics
        diag_dir = Path(cfg.diagnostics.output_dir) if cfg.diagnostics.output_dir \
                   else Path(cfg.paths.output_dir) / "diagnostics"
        print(f"=== Stage 2: Diagnostics ===")
        run_diagnostics(
            results=results,
            top_rows=top_rows,
            gp=gp,
            Y_train_ZRG=Y_train_ZRG,
            Y_scalers=Y_scalers,
            obs_parts=obs_parts,
            param_names=param_names,
            var_names=var_names,
            n_zonal=n_zonal,
            n_regions=n_regions,
            regions_list=cfg.data.regions_list,
            out_dir=diag_dir,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only stage 1 (preprocess) or stage 2 (surrogate+optimize). "
             "Default: run both.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if not cfg.paths.preprocess_dir:
        raise ValueError("cfg.paths.preprocess_dir must be set in the config.")

    if args.stage == 1:
        run_stage1(cfg)
    elif args.stage == 2:
        run_stage2(cfg)
    else:
        run_stage1(cfg)
        run_stage2(cfg)


if __name__ == "__main__":
    main()
