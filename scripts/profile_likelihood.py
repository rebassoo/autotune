"""
Profile-likelihood 68% intervals for the MF Laplace run's best-fit parameters.

Unlike the Laplace/Hessian approximation (autotune_gp.uncertainty), this does
not assume a local quadratic cost surface, and it works even for parameters
pinned at a bound (where the Hessian approach returns NaN by design).

For each parameter, walk outward from x_best in both directions; at each
trial value, RE-OPTIMIZE the other 18 parameters (warm-started from the
previous step) rather than holding them fixed. Wilks' theorem: the point
where that re-optimized cost first rises by delta=0.5 above cost_best is the
68% (1 sigma) profile-likelihood bound in that direction. If the cost never
reaches the threshold before hitting the parameter's own search-box edge,
the interval is open in that direction (reported as such, not as a false
finite bound).

Parallelized across the 19*2=38 (parameter, direction) tasks via a forked
process pool, following the same _WORKER copy-on-write handoff
autotune_gp.optimize.py uses for this same (1.9GB) GP object.

Usage:
    python scripts/profile_likelihood.py \
        --run-dir /path/to/results_..._laplace \
        --gp /path/to/results_..._laplace/mf_gp_trained.pkl \
        --preprocess-dir /path/to/preprocess_output_ne128_prod \
        --out /path/to/results_..._laplace/diagnostics/profile_likelihood.pkl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_WORKER: dict = {}


def _reoptimize(fixed_idx, fixed_val, x0_other, other_idx, other_bounds):
    cost_fn = _WORKER["cost_fn"]
    n = _WORKER["n_params"]

    def reduced_cost(x_other):
        full = np.empty(n)
        full[fixed_idx] = fixed_val
        full[other_idx] = x_other
        return cost_fn(full)

    res = minimize(reduced_cost, x0_other, method="L-BFGS-B", bounds=other_bounds)
    return res.x, float(res.fun)


def _profile_task(args):
    pi, direction = args
    x_best = _WORKER["x_best"]
    bounds_low = _WORKER["bounds_low"]
    bounds_high = _WORKER["bounds_high"]
    cost_best = _WORKER["cost_best"]
    delta = _WORKER["delta"]
    n = _WORKER["n_params"]

    target = cost_best + delta
    edge = bounds_high[pi] if direction > 0 else bounds_low[pi]
    other_idx = [i for i in range(n) if i != pi]
    other_bounds = [(bounds_low[i], bounds_high[i]) for i in other_idx]

    val = x_best[pi]
    x_other = x_best[other_idx].copy()
    box_width = bounds_high[pi] - bounds_low[pi]
    step = 0.03 * box_width * direction

    n_reopt = 0
    prev_val, prev_other = val, x_other
    val_next, cost_next, reached_edge = None, None, False
    for _ in range(25):
        val_next = val + step
        reached_edge = (direction > 0 and val_next >= edge) or (direction < 0 and val_next <= edge)
        if reached_edge:
            val_next = edge
        x_other_next, cost_next = _reoptimize(pi, val_next, prev_other, other_idx, other_bounds)
        n_reopt += 1
        if cost_next >= target or reached_edge:
            break
        prev_val, prev_other = val_next, x_other_next
        val = val_next
        step *= 1.4

    if reached_edge and cost_next < target:
        return dict(param_idx=pi, direction=direction, bound_val=edge,
                    hit_edge=True, n_reopt=n_reopt, edge_cost=cost_next)

    # bisection refine between (prev_val, cost<target) and (val_next, cost>=target)
    lo, hi = prev_val, val_next
    lo_other = prev_other
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        x_mid_other, cost_mid = _reoptimize(pi, mid, lo_other, other_idx, other_bounds)
        n_reopt += 1
        if cost_mid < target:
            lo, lo_other = mid, x_mid_other
        else:
            hi = mid
    return dict(param_idx=pi, direction=direction, bound_val=0.5 * (lo + hi),
                hit_edge=False, n_reopt=n_reopt, edge_cost=None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gp", required=True)
    p.add_argument("--preprocess-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-workers", type=int, default=32)
    p.add_argument("--delta", type=float, default=0.5,
                    help="Wilks' 68%% threshold for a 1-parameter profile: Delta(-logL)=0.5.")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from autotune_gp.backend import get_backend
    from autotune_gp.io import load_obs, split_zrg_obs
    from autotune_gp.transforms import transform_obs
    from autotune_gp.cost import zrg_cost_function_laplace

    var_names = ["PCP", "OSR", "OLR", "TLWP"]
    var_w = {"PCP": 0.3, "OSR": 0.1, "OLR": 0.3, "TLWP": 0.3}
    zrg_w = {"zonal": 1 / 3, "regional": 1 / 3, "global": 1 / 3}
    dy_w = {"ANN": 1.0}
    n_regions = 6
    backend = get_backend("numpy", "cpu")

    prep = Path(args.preprocess_dir)
    with open(prep / "scalers.pkl", "rb") as f:
        saved = pickle.load(f)
    Y_scalers = [saved[f"Y_pipeline_{v}"] for v in var_names]
    obs_loaded = load_obs(str(prep / "obs.pkl"))
    obs_parts = split_zrg_obs(obs_loaded["zrg_obs"], n_vars=len(var_names))
    obs_norm = transform_obs(obs_parts, Y_scalers)
    n_zonal = obs_norm.shape[1] - n_regions - 1

    print("Loading MF GP checkpoint (can be slow) ...", flush=True)
    t0 = time.time()
    with open(args.gp, "rb") as f:
        gp = pickle.load(f)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    def cost_fn(x):
        x = np.asarray(x, dtype=float)
        m, v = gp.predict(x)
        return zrg_cost_function_laplace(
            m, v, obs_norm, var_w, zrg_w, dy_w,
            n_zonal=n_zonal, n_regions=n_regions, backend=backend,
            var_names=var_names, var_floor=1e-6, variance_penalty_kappa=0.0,
        )

    run_dir = Path(args.run_dir)
    with open(run_dir / "diagnostics" / "parameter_uncertainty_seed50.pkl", "rb") as f:
        unc = pickle.load(f)
    param_names = unc["param_names"]
    x_best = unc["x"]
    n_params = len(x_best)

    with open(run_dir / "optimize_checkpoints" / "manifest.pkl", "rb") as f:
        man = pickle.load(f)
    bounds_low = np.asarray(man["bounds_low"], dtype=float)
    bounds_high = np.asarray(man["bounds_high"], dtype=float)

    cost_best = cost_fn(x_best)
    print(f"cost at x_best = {cost_best:.6f}", flush=True)

    _WORKER.clear()
    _WORKER.update(cost_fn=cost_fn, x_best=x_best, bounds_low=bounds_low,
                    bounds_high=bounds_high, cost_best=cost_best, delta=args.delta,
                    n_params=n_params)

    tasks = [(pi, d) for pi in range(n_params) for d in (+1, -1)]
    print(f"Running {len(tasks)} profile-likelihood searches "
          f"({n_params} params x 2 directions) across {args.max_workers} workers ...",
          flush=True)

    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.max_workers,
                              mp_context=mp.get_context("fork")) as ex:
        for out in ex.map(_profile_task, tasks):
            pi, d = out["param_idx"], out["direction"]
            results.setdefault(pi, {})[d] = out
            tag = "EDGE" if out["hit_edge"] else "cross"
            print(f"  [{param_names[pi]:<38s} dir={'+' if d>0 else '-'}] "
                  f"bound={out['bound_val']:.4f} ({tag}, {out['n_reopt']} reopts) "
                  f"[{time.time()-t0:.0f}s elapsed]", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(param_names=param_names, x_best=x_best, cost_best=cost_best,
                    bounds_low=bounds_low, bounds_high=bounds_high,
                    delta=args.delta, results=results)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved {out_path}")

    print(f"\n{'Parameter':<38s} {'value':>8s} {'lo68':>8s} {'hi68':>8s} "
          f"{'lo_edge':>8s} {'hi_edge':>8s}")
    for pi, name in enumerate(param_names):
        lo = results[pi][-1]
        hi = results[pi][+1]
        print(f"{name:<38s} {x_best[pi]:8.4f} {lo['bound_val']:8.4f} {hi['bound_val']:8.4f} "
              f"{'yes' if lo['hit_edge'] else 'no':>8s} {'yes' if hi['hit_edge'] else 'no':>8s}")


if __name__ == "__main__":
    main()
