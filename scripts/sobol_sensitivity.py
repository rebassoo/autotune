"""
Global (Sobol) sensitivity of the surrogate's outputs to the tuning parameters:
which parameters could be fixed without losing anything the outputs respond to.

The trained multi-fidelity surrogate stands in for SCREAM. Parameters are
sampled (Saltelli scheme, SALib) uniformly over the optimizer's search box --
the range the PPE sampled, in the normalized (log-scaled where configured)
space, which is the design's own distribution. Parameters given with --fix are
held at a value instead (by default lambda_low and lambda_high, which the
design ties together with lambda_low <= lambda_high, so they are not
independent inputs; they are held at the best fit given by --theta-from).

For every output feature (zonal bands, regions, global; per variable) it
computes the first-order (S1: that parameter alone) and total-order (ST:
including all its interactions) indices, with bootstrap confidence intervals.
Per variable they are summarized as a variance-weighted mean over features,
so features the parameters barely move count little. It also reports ST of
each variable's misfit to the observations (mean squared normalized residual),
i.e. which parameters matter for the fit rather than for the outputs alone.

A parameter with total-order ST below ~0.01-0.05 for every variable and every
misfit is a candidate to fix; confirm by re-optimizing without it.

Uses the surrogate's predictive mean only. Needs a compute node (the 3-level
surrogate does not fit in login-node memory).

    python scripts/sobol_sensitivity.py \\
        --gp-pkl /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl \\
        --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \\
        --theta-from /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_studentt_dof4 \\
        --n-base 2048 --out /pscratch/sd/r/rebassoo/autotune/sobol_n85
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from autotune_gp.config import load_config                        # noqa: E402
from autotune_gp.io import load_obs, split_zrg_obs                 # noqa: E402
from autotune_gp.transforms import transform_obs                   # noqa: E402


def _ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _best_theta(results_dir):
    rows = list(csv.DictReader(open(sorted(glob.glob(f"{results_dir}/results_*.csv"))[-1])))
    return np.asarray(json.loads(min(rows, key=lambda r: float(r["cost"]))["params"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp-pkl", required=True)
    ap.add_argument("--config", required=True,
                    help="Optimization config: parameter names, variables, obs and scalers.")
    ap.add_argument("--fix", nargs="*", default=["lambda_low", "lambda_high"],
                    help="Parameters held fixed (default: lambda_low lambda_high).")
    ap.add_argument("--theta-from", required=True,
                    help="Results dir whose best point gives the fixed parameters' values.")
    ap.add_argument("--n-base", type=int, default=2048,
                    help="Saltelli base sample size (power of 2); points = n_base*(d+2).")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output directory.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    names = list(cfg.optimize.param_physical_bounds)
    var_names = list(cfg.data.variables)
    with open(args.gp_pkl, "rb") as f:
        gp = pickle.load(f)
    _ts(f"loaded surrogate {args.gp_pkl}")

    # search box = the sampled range over all levels (optimize.bounds_from_data)
    X_all = np.vstack([np.asarray(x, float) for x in gp.X_levels])
    lo, hi = np.clip(X_all.min(0), 0, 1), np.clip(X_all.max(0), 0, 1)
    theta = _best_theta(args.theta_from)
    fixed = {n: float(theta[names.index(n)]) for n in args.fix}
    free = [i for i, n in enumerate(names) if n not in fixed]
    _ts(f"fixed at the best fit: {fixed}; {len(free)} free parameters")

    problem = {"num_vars": len(free), "names": [names[i] for i in free],
               "bounds": [[lo[i], hi[i]] for i in free]}
    S = sobol_sample.sample(problem, args.n_base, calc_second_order=False, seed=args.seed)
    X = np.tile(theta, (len(S), 1))
    X[:, free] = S
    _ts(f"{len(X)} Saltelli points")

    preds = []
    for k in range(0, len(X), args.chunk):
        m, _ = gp.predict_batch(X[k:k + args.chunk])
        preds.append(np.asarray(m))
        _ts(f"  predicted {min(k + args.chunk, len(X))}/{len(X)}")
    M = np.concatenate(preds)                                     # (n, n_feat, n_vars)
    np.save(out / "predictions.npy", M)

    pre = Path(cfg.paths.preprocess_dir)
    sc_dir = Path(cfg.multi_fidelity.scalers_dir) if cfg.multi_fidelity.scalers_dir else pre
    with open(sc_dir / "scalers.pkl", "rb") as f:
        sc = pickle.load(f)
    obs = transform_obs(split_zrg_obs(load_obs(str(pre / "obs.pkl"))["zrg_obs"],
                                      n_vars=len(var_names)),
                        [sc[f"Y_pipeline_{v}"] for v in var_names])[0]   # (n_feat, n_vars)

    n_feat = M.shape[1]
    ST = np.zeros((n_feat, len(var_names), len(free)))
    S1 = np.zeros_like(ST)
    ST_conf = np.zeros_like(ST)
    for j in range(len(var_names)):
        for f in range(n_feat):
            r = sobol_analyze.analyze(problem, M[:, f, j], calc_second_order=False,
                                      seed=args.seed)
            ST[f, j], S1[f, j], ST_conf[f, j] = r["ST"], r["S1"], r["ST_conf"]
    out_var = M.var(axis=0)                                       # (n_feat, n_vars)
    w = out_var / out_var.sum(axis=0, keepdims=True)
    ST_var = np.einsum("fj,fjp->jp", w, ST)                       # (n_vars, n_free)
    S1_var = np.einsum("fj,fjp->jp", w, S1)

    misfit = ((M - obs[None]) ** 2).mean(axis=1)                  # (n, n_vars)
    ST_mis = np.zeros((len(var_names), len(free)))
    for j in range(len(var_names)):
        ST_mis[j] = sobol_analyze.analyze(problem, misfit[:, j], calc_second_order=False,
                                          seed=args.seed)["ST"]

    pnames = problem["names"]
    with open(out / "sobol.pkl", "wb") as fh:
        pickle.dump(dict(params=pnames, variables=var_names, fixed=fixed, n_base=args.n_base,
                         ST=ST, S1=S1, ST_conf=ST_conf, ST_var=ST_var, S1_var=S1_var,
                         ST_misfit=ST_mis, box=(lo, hi)), fh)
    with open(out / "sobol_summary.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["parameter"] + [f"ST_{v}" for v in var_names]
                    + [f"ST_misfit_{v}" for v in var_names] + ["max"])
        for p, n in enumerate(pnames):
            row = list(ST_var[:, p]) + list(ST_mis[:, p])
            wr.writerow([n] + [f"{x:.4f}" for x in row] + [f"{max(row):.4f}"])

    order = np.argsort(-np.maximum(ST_var.max(0), ST_mis.max(0)))
    print(f"\nTotal-order Sobol indices (variance-weighted over features), n_base={args.n_base}")
    print(f"  {'parameter':38s}" + "".join(f"{v:>8s}" for v in var_names)
          + "".join(f"{'mis_' + v:>10s}" for v in var_names))
    for p in order:
        print(f"  {pnames[p]:38s}" + "".join(f"{x:8.3f}" for x in ST_var[:, p])
              + "".join(f"{x:10.3f}" for x in ST_mis[:, p]))
    print(f"  max bootstrap 95% half-width on a feature's ST: {ST_conf.max():.3f}")

    # heatmap: parameters (sorted) x [variables | misfits]
    H = np.hstack([ST_var.T, ST_mis.T])[order]
    fig, ax = plt.subplots(figsize=(1.0 * H.shape[1] + 4.5, 0.36 * len(pnames) + 1.8))
    im = ax.imshow(H, cmap="Blues", vmin=0, vmax=max(0.3, float(H.max())), aspect="auto")
    for (a, b), x in np.ndenumerate(H):
        ax.text(b, a, f"{x:.2f}", ha="center", va="center", fontsize=7,
                color="white" if x > 0.6 * max(0.3, H.max()) else "#1a1a1a")
    ax.set_yticks(range(len(pnames)))
    ax.set_yticklabels([pnames[p] for p in order], fontsize=8)
    ax.set_xticks(range(H.shape[1]))
    ax.set_xticklabels(var_names + [f"misfit {v}" for v in var_names], rotation=45,
                       ha="right", fontsize=8)
    ax.axvline(len(var_names) - 0.5, color="#1a1a1a", lw=1)
    fig.colorbar(im, ax=ax, fraction=0.04, label="total-order Sobol index ST")
    ax.set_title(f"Total-order Sobol indices (fixed: {', '.join(fixed)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sobol_total_order.png", dpi=150)
    _ts(f"saved {out}/sobol.pkl, sobol_summary.csv, sobol_total_order.png")


if __name__ == "__main__":
    main()
