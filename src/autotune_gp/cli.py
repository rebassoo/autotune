from __future__ import annotations
import argparse
import sys
import numpy as np

from .config import load_config
from .backend import get_backend
from .io import load_obs, load_gp_proj, split_zrg_obs
from .transforms import fit_transform_X, fit_transform_Y, transform_obs
from .gp import GPWrapper
from .cost import zrg_cost_function_mae_weighted
from .optimize import optimize_parallel

def _require_py3():
    if sys.version_info[0] < 3:
        raise RuntimeError("This package requires Python 3. You are running: %s" % sys.version)

def main():
    _require_py3()

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    opt = sub.add_parser("optimize")
    opt.add_argument("--config", required=True)

    args = p.parse_args()
    cfg = load_config(args.config)

    backend = get_backend(cfg.runtime.backend, cfg.runtime.device)

    obs_loaded = load_obs(cfg.paths.obs_pkl)
    zrg_obs = obs_loaded["zrg_obs"]
    obs_parts = split_zrg_obs(zrg_obs, n_vars=len(cfg.data.variables))

    gp_loaded = load_gp_proj(cfg.paths.gp_proj_pkl)
    X_train = gp_loaded["X_train"]
    Y_train_ZRG = gp_loaded["Y_train"]

    var_names = cfg.data.variables
    # Try to load saved scalers from gp_proj_pkl, then from preprocess_dir/scalers.pkl
    X_sc      = gp_loaded.get("X_pipeline")
    Y_scalers = [gp_loaded.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is None or not all(s is not None for s in Y_scalers):
        from pathlib import Path as _Path
        import pickle as _pickle
        scalers_pkl_path = _Path(cfg.paths.preprocess_dir) / "scalers.pkl"
        if scalers_pkl_path.exists():
            with open(scalers_pkl_path, "rb") as f:
                saved = _pickle.load(f)
            X_sc      = saved.get("X_pipeline")
            Y_scalers = [saved.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is not None and all(s is not None for s in Y_scalers):
        X_train_norm = X_sc.transform(X_train)
        Y_train_norm = np.stack(
            [Y_scalers[j].transform(Y_train_ZRG[:, :, j]) for j in range(len(var_names))],
            axis=0).transpose(1, 2, 0)
        obs_norm = transform_obs(obs_parts, Y_scalers)
        print("Using saved scalers (exact match with training)")
    else:
        print("Warning: no saved scalers found — refitting from scratch")
        phys = cfg.optimize.param_physical_bounds
        if phys and hasattr(X_train, "columns"):
            param_bounds = np.array([phys[col] for col in X_train.columns])
            X_sc, X_train_norm = fit_transform_X(X_train, param_bounds=param_bounds.T)
        else:
            X_sc, X_train_norm = fit_transform_X(X_train)
        Y_scalers, Y_train_norm = fit_transform_Y(Y_train_ZRG)
        obs_norm = transform_obs(obs_parts, Y_scalers)

    gp = GPWrapper(X_train_norm, Y_train_norm)
    if cfg.runtime.train_gp:
        gp.train(tf_determinism=cfg.runtime.tf_determinism)

    n_regions        = len(cfg.data.regions_list)
    n_zonal          = int(cfg.data.n_zonal)
    var_w            = cfg.weights.variables
    zrg_w            = cfg.weights.zrg
    dy_w             = cfg.weights.dy
    zonal_weights    = cfg.weights.zonal_weights
    regional_weights = cfg.weights.regional_weights

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
        return zrg_cost_function_mae_weighted(m, obs_norm, var_w, zrg_w, dy_w,
                                              n_zonal=n_zonal, n_regions=n_regions, backend=backend,
                                              zonal_weights=zonal_weights, regional_weights=regional_weights)

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

    print("Wrote: %s" % csv_path)

if __name__ == "__main__":
    main()
