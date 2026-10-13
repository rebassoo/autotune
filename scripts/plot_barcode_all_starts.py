"""
Barcode of optimized parameter values for EVERY multi-start, not just the top K.

The diagnostics written during a run plot only the rows kept in results_*.csv
(top 10 by cost). optimize_parallel checkpoints every start to
<output_dir>/optimize_checkpoints/start_XXXX.pkl as [19 params, cost], so the
full set is already on disk -- this reads those.

Seeing all of them matters: it shows how many starts reached the optimum versus
how many stalled elsewhere, and how much the parameter values disagree among
starts that reached the same cost (the flat-cost-surface question).

Usage:
    python scripts/plot_barcode_all_starts.py \
        --results-dir /path/to/results_mf3_ne256_studentt_dof4 \
        --config configs/perlmutter_mf3_ne256_studentt_dof4_annual.yaml \
        --out docs/barcode_all_starts_mf3_ne256.png
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

SURFACE = "#fcfcfb"
TEXT2 = "#52514e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--config", default=None,
                    help="Config to read parameter names from (optional).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    files = sorted(glob.glob(str(rd / "optimize_checkpoints" / "start_*.pkl")))
    if not files:
        raise SystemExit(f"No start_*.pkl checkpoints in {rd / 'optimize_checkpoints'}")
    A = np.array([pickle.load(open(f, "rb")) for f in files], dtype=float)
    params, cost = A[:, :-1], A[:, -1]
    n, n_params = params.shape

    names = None
    if args.config:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from autotune_gp.config import load_config
        pb = load_config(args.config).optimize.param_physical_bounds
        if pb and len(pb) == n_params:
            names = list(pb)
    if names is None:
        unc = sorted(glob.glob(str(rd / "diagnostics" / "parameter_uncertainty*.pkl")))
        if unc:
            names = list(pickle.load(open(unc[-1], "rb"))["param_names"])
    if names is None:
        names = [f"p{i}" for i in range(n_params)]

    order = np.argsort(cost)                 # best first
    params, cost = params[order], cost[order]

    near = int((cost <= cost.min() * 1.01).sum()) if cost.min() > 0 else 0
    spread = params[cost <= cost.min() * 1.01].ptp(axis=0).max() if near > 1 else 0.0

    fig, ax = plt.subplots(figsize=(max(14, n_params * 0.62), 0.22 * n + 4.4),
                           facecolor=SURFACE)
    im1 = ax.imshow(np.clip(params, 0, 1), aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    im2 = ax.imshow(np.flipud(cost[:, None]), aspect="auto", cmap="Greens",
                    vmin=cost.min(), vmax=cost.max(),
                    extent=[n_params - 0.5, n_params + 0.5, -0.5, n - 0.5])

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f"{i+1}" for i in range(n)], fontsize=7)
    ax.set_xticks(list(range(n_params)) + [n_params])
    ax.set_xticklabels(names + ["Cost"], rotation=90)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Start, ranked by cost (1 = best)")
    sub = (f"cost: best {cost.min():.6f} · median {np.median(cost):.6f} · "
           f"worst {cost.max():.6f}   |   {near}/{n} starts within 1% of the best"
           + (f"; among those the same parameter differs by at most {spread:.2f} "
              "in normalised units" if near > 1 else ""))
    ax.set_title((args.title or
                  f"All {n} optimizer starts — optimized parameter values, ranked by cost")
                 + "\n" + sub, pad=14, fontsize=12)
    ax.set_xticks(np.arange(-0.5, n_params + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylim(n - 0.5, -0.5)

    fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.065,
                 label="Normalized parameter value")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.025, pad=0.01, label="Cost")
    # Keep absolute cost values on the ticks: with many starts at nearly the
    # same cost the default formatter factors out an offset and the labels then
    # look like costs but are not.
    cb2.formatter = ScalarFormatter(useOffset=False)
    cb2.formatter.set_scientific(False)
    cb2.update_ticks()

    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {args.out}  ({n} starts)")


if __name__ == "__main__":
    main()
