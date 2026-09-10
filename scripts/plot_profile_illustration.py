"""
Illustrative 2-D plot of what profile likelihood is actually doing, for one
correlated pair of MF Laplace parameters (default: length_fac vs coeff_kh,
the strongest-correlated free pair per the Hessian's correlation matrix).

Shows, on the same axes:
  - a cost contour (other 17 params frozen at x_best) -- the raw GP-based
    2-D slice through the cost surface;
  - the *true* profile-likelihood ridge: sweep param_x outward in both
    directions from x_best, re-optimizing all 18 other parameters (not just
    param_y) at each step, warm-started from the previous step -- exactly the
    same walk autotune_gp's profile_likelihood.py does for the full 1-D
    bound, but here every step's param_y value is also recorded so the path
    can be drawn;
  - a naive reference line at param_y = param_y_best (what a frozen-slice /
    Hessian-only view implicitly assumes nuisance parameters do);
  - two 1-sigma ellipses computed for free from the already-saved Hessian and
    inverse-Hessian: a narrow one from the 2x2 Hessian sub-block (naive,
    other-17-frozen curvature) and a wide one from the corresponding 2x2
    block of the full inverse Hessian (correctly marginalized over all 17
    other correlated directions).

Usage:
    python scripts/plot_profile_illustration.py \
        --run-dir /path/to/results_..._laplace \
        --gp /path/to/results_..._laplace/mf_gp_trained.pkl \
        --preprocess-dir /path/to/preprocess_output_ne128_prod \
        --out /path/to/results_..._laplace/diagnostics/profile_illustration.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def _reoptimize(cost_fn, n, fixed_idx, fixed_val, x0_other, other_idx, other_bounds):
    def reduced_cost(x_other):
        full = np.empty(n)
        full[fixed_idx] = fixed_val
        full[other_idx] = x_other
        return cost_fn(full)

    res = minimize(reduced_cost, x0_other, method="L-BFGS-B", bounds=other_bounds)
    return res.x, float(res.fun)


def _walk_profile(cost_fn, n, pi, x_best, edge_lo, edge_hi):
    """Walk pi outward from x_best in both directions to edge_lo/edge_hi,
    re-optimizing the other n-1 params at each step. Returns arrays of
    (val, x_other, cost) visited, sorted low-to-high, including x_best."""
    other_idx = [k for k in range(n) if k != pi]
    other_bounds = None  # filled by caller via closure (see main())

    def one_direction(direction, edge):
        vals, others, costs = [], [], []
        val = x_best[pi]
        x_other = x_best[other_idx].copy()
        box_width = abs(edge - x_best[pi])
        step = 0.06 * box_width * direction if box_width > 0 else 0.0
        for _ in range(30):
            if step == 0:
                break
            val_next = val + step
            reached = (direction > 0 and val_next >= edge) or (direction < 0 and val_next <= edge)
            if reached:
                val_next = edge
            x_other_next, cost_next = _reoptimize(
                cost_fn, n, pi, val_next, x_other, other_idx, other_bounds
            )
            vals.append(val_next); others.append(x_other_next); costs.append(cost_next)
            val, x_other = val_next, x_other_next
            if reached:
                break
            step *= 1.25
        return vals, others, costs

    return one_direction, other_idx


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gp", required=True)
    p.add_argument("--preprocess-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--param-x", default="length_fac")
    p.add_argument("--param-y", default="coeff_kh")
    p.add_argument("--grid-n", type=int, default=41)
    p.add_argument("--range-sigma", type=float, default=20.0,
                    help="Half-width of the plotted range, in units of that "
                         "parameter's Hessian stderr, before clipping to the box.")
    p.add_argument("--safety-frac", type=float, default=0.6,
                    help="Never plot closer than this fraction of the way to the box edge.")
    args = p.parse_args()

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

    print("Loading MF GP checkpoint ...", flush=True)
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
    x_best = np.asarray(unc["x"], dtype=float)
    stderr = np.asarray(unc["stderr"], dtype=float)
    H = unc["hessian"]
    cov = unc["cov"]
    n = len(param_names)

    with open(run_dir / "optimize_checkpoints" / "manifest.pkl", "rb") as f:
        man = pickle.load(f)
    bounds_low = np.asarray(man["bounds_low"], dtype=float)
    bounds_high = np.asarray(man["bounds_high"], dtype=float)

    i = param_names.index(args.param_x)
    j = param_names.index(args.param_y)

    cost_best = cost_fn(x_best)
    print(f"cost at x_best = {cost_best:.6f}", flush=True)

    def axis_range(k):
        lo_room = args.safety_frac * (x_best[k] - bounds_low[k])
        hi_room = args.safety_frac * (bounds_high[k] - x_best[k])
        half_lo = min(args.range_sigma * stderr[k], lo_room)
        half_hi = min(args.range_sigma * stderr[k], hi_room)
        return x_best[k] - half_lo, x_best[k] + half_hi

    xi_lo, xi_hi = axis_range(i)
    yj_lo, yj_hi = axis_range(j)
    print(f"{args.param_x} range: [{xi_lo:.5f}, {xi_hi:.5f}]  (x_best={x_best[i]:.5f})")
    print(f"{args.param_y} range: [{yj_lo:.5f}, {yj_hi:.5f}]  (x_best={x_best[j]:.5f})")

    # ------------------------------------------------------------------
    # Base layer: cost contour with the other 17 params frozen at x_best
    # ------------------------------------------------------------------
    print("Computing base contour grid ...", flush=True)
    xs = np.linspace(xi_lo, xi_hi, args.grid_n)
    ys = np.linspace(yj_lo, yj_hi, args.grid_n)
    Z = np.empty((args.grid_n, args.grid_n))
    t0 = time.time()
    for a, xv in enumerate(xs):
        for b, yv in enumerate(ys):
            xfull = x_best.copy()
            xfull[i] = xv
            xfull[j] = yv
            Z[b, a] = cost_fn(xfull)
    print(f"  grid done in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------
    # True profile-likelihood ridge: sweep param_x, re-optimize all 18
    # other params (including param_y) at each step, warm-started.
    # ------------------------------------------------------------------
    print("Walking the true profile ridge (re-optimizing all 18 other params) ...", flush=True)
    other_idx = [k for k in range(n) if k != i]
    other_bounds = [(bounds_low[k], bounds_high[k]) for k in other_idx]

    def walk(direction, edge):
        vals, y_of_other, costs = [x_best[i]], [x_best[j]], [cost_best]
        val = x_best[i]
        x_other = x_best[other_idx].copy()
        box_room = abs(edge - x_best[i])
        step = 0.06 * box_room * direction if box_room > 0 else 0.0
        for _ in range(30):
            if step == 0:
                break
            val_next = val + step
            reached = (direction > 0 and val_next >= edge) or (direction < 0 and val_next <= edge)
            if reached:
                val_next = edge
            x_other_next, cost_next = _reoptimize(
                cost_fn, n, i, val_next, x_other, other_idx, other_bounds
            )
            vals.append(val_next)
            y_of_other.append(x_other_next[other_idx.index(j)])
            costs.append(cost_next)
            val, x_other = val_next, x_other_next
            if reached:
                break
            step *= 1.25
        return vals, y_of_other, costs

    t0 = time.time()
    v_hi, y_hi, c_hi = walk(+1, xi_hi)
    v_lo, y_lo, c_lo = walk(-1, xi_lo)
    print(f"  profile walk done in {time.time()-t0:.0f}s "
          f"({len(v_lo)+len(v_hi)-1} re-optimizations)", flush=True)

    profile_x = np.array(list(reversed(v_lo)) + v_hi[1:])
    profile_y = np.array(list(reversed(y_lo)) + y_hi[1:])
    profile_c = np.array(list(reversed(c_lo)) + c_hi[1:])

    # ------------------------------------------------------------------
    # Two 1-sigma ellipses from already-saved Hessian / inverse-Hessian
    # ------------------------------------------------------------------
    def ellipse_from_cov(cov2, center, n_pts=200):
        eigval, eigvec = np.linalg.eigh(cov2)
        eigval = np.clip(eigval, 0, None)
        theta = np.linspace(0, 2 * np.pi, n_pts)
        circle = np.stack([np.cos(theta), np.sin(theta)])
        pts = eigvec @ (np.sqrt(eigval)[:, None] * circle)
        return center[0] + pts[0], center[1] + pts[1]

    H_pair = H[np.ix_([i, j], [i, j])]
    cov_naive = np.linalg.inv(H_pair)
    cov_marg = cov[np.ix_([i, j], [i, j])]
    center = (x_best[i], x_best[j])
    ex_n, ey_n = ellipse_from_cov(cov_naive, center)
    ex_m, ey_m = ellipse_from_cov(cov_marg, center)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cf = ax.contourf(xs, ys, Z, levels=30, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("cost (Laplace NLL, other 17 params frozen at x_best)")

    ax.axhline(x_best[j], color="white", ls="--", lw=1.3,
               label=f"naive: {args.param_y} frozen at best")
    ax.plot(profile_x, profile_y, color="red", lw=2.2, marker="o", ms=3.5,
            label="true profile ridge (all 18 others re-optimized)")
    ax.plot(ex_n, ey_n, color="cyan", lw=1.6, ls="-",
            label="1σ ellipse, naive (2x2 Hessian block, others frozen)")
    ax.plot(ex_m, ey_m, color="orange", lw=1.8, ls="-",
            label="1σ ellipse, marginal (2x2 of full inverse Hessian)")
    ax.plot(x_best[i], x_best[j], marker="*", ms=16, color="white",
            mec="black", mew=0.8, ls="none", label="x_best", zorder=5)

    ax.set_xlim(xi_lo, xi_hi)
    ax.set_ylim(yj_lo, yj_hi)
    ax.set_xlabel(f"{args.param_x} (optimizer-normalized units)")
    ax.set_ylabel(f"{args.param_y} (optimizer-normalized units)")
    ax.set_title("What profile likelihood actually does:\n"
                 "true ridge (re-optimized) vs. naive frozen slice vs. Hessian ellipses")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")

    payload = dict(param_x=args.param_x, param_y=args.param_y, i=i, j=j,
                   x_best=x_best, cost_best=cost_best,
                   xs=xs, ys=ys, Z=Z,
                   profile_x=profile_x, profile_y=profile_y, profile_c=profile_c,
                   cov_naive=cov_naive, cov_marg=cov_marg)
    with open(out_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {out_path.with_suffix('.pkl')}")


if __name__ == "__main__":
    main()
