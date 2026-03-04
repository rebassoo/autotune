from __future__ import annotations
import argparse
import sys
import numpy as np

from .config import load_config
from .backend import get_backend
from .io import load_obs, load_gp_proj, split_zrg_obs
from .transforms import fit_transform_X, fit_transform_Y, transform_obs
from .gp import GPWrapper
from .cost import zrg_cost_function_rmse_like_reference
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

    _, X_train_norm = fit_transform_X(X_train)
    Y_scalers, Y_train_norm = fit_transform_Y(Y_train_ZRG)
    obs_norm = transform_obs(obs_parts, Y_scalers)

    gp = GPWrapper(X_train_norm, Y_train_norm)
    if cfg.runtime.train_gp:
        gp.train()

    n_regions = len(cfg.data.regions_list)
    n_zonal = int(cfg.data.n_zonal)

    var_w = cfg.weights.variables
    zrg_w = cfg.weights.zrg
    dy_w  = cfg.weights.dy

    def cost_fn(x):
        x = np.asarray(x, dtype=float)
        m, v = gp.predict(x)  # mean, var
        return zrg_cost_function_rmse_like_reference(m, obs_norm, var_w, zrg_w, dy_w,
                                                     n_zonal=n_zonal, n_regions=n_regions, backend=backend)

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
