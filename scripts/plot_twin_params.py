"""
Twin experiments: the true parameters of the target member next to the optimizer's
best fit (and, with --second-best, the second-best optimum), one panel per twin, for all parameters.

x is the normalized parameter (0-1 over the declared range, log space where
configured); the grey band is the range the PPE sampled (the optimizer's box).
Parameters are ordered by their largest total-order Sobol index of the outputs
(not the misfits) over the variables (sobol_summary.csv).
Optima come from all checkpointed starts, grouped by cost (score_twin.clusters).
Reads results only (no GP).

    python scripts/plot_twin_params.py --out docs/twin_params_n85.png
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_twin import clusters   # noqa: E402

S = Path("/pscratch/sd/r/rebassoo/autotune")
TWINS = [("m060_3level", "m060, 3-level (ne256 member held out)"),
         ("m060_2level", "m060, 2-level (ne128 + ne32 only)"),
         ("m202_3level", "m202, 3-level (ne256 member held out)"),
         ("m202_2level", "m202, 2-level (ne128 + ne32 only)")]
INK, BLUE, ORANGE, BAND = "#1a1a1a", "#2a6fdb", "#e8833a", "#e6e8eb"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sobol", default=str(S / "sobol_n85" / "sobol_summary.csv"))
    ap.add_argument("--second-best", action="store_true", help="Also draw the second-best optimum.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L = lambda p: pickle.load(open(p, "rb"))

    # largest output (not misfit) total-order index over the variables
    imp = {r["parameter"]: max(float(v) for k, v in r.items() if k.startswith("ST_") and "misfit" not in k)
           for r in csv.DictReader(open(args.sobol))}
    fig, axes = plt.subplots(1, 4, figsize=(17, 8.2), sharey=True)
    order = None
    for a, (tag, title) in zip(axes, TWINS):
        tw, rd = S / "twin_preprocess" / tag, S / f"results_twin_{tag}"
        truth = L(tw / "truth.pkl")
        names = list(truth["params"])
        xt = L(tw / "scalers.pkl")["X_pipeline"].transform(truth["x_phys"][None, :])[0]
        man = L(rd / "optimize_checkpoints" / "manifest.pkl")
        lo, hi = np.asarray(man["bounds_low"]), np.asarray(man["bounds_high"])
        rows = [np.asarray(L(f), float) for f in sorted((rd / "optimize_checkpoints").glob("start_*.pkl"))]
        groups = clusters(rows)
        if order is None:
            order = sorted(range(len(names)), key=lambda j: imp.get(names[j], 0.0)
                           if names[j] not in ("lambda_low", "lambda_high") else -1)
        y = np.arange(len(order))
        a.barh(y, (hi - lo)[order], left=lo[order], height=0.6, color=BAND, lw=0, zorder=0,
               label="range sampled by the PPE")
        best, nb = groups[0][0], len(groups[0])
        a.hlines(y, np.minimum(xt, best[:-1])[order], np.maximum(xt, best[:-1])[order],
                 color=BLUE, lw=1.2, alpha=0.5, zorder=1)
        if args.second_best and len(groups) > 1:
            g2 = groups[1][0]
            a.plot(g2[:-1][order], y, ls="", marker="o", ms=6.5, mfc="none", mec=ORANGE, mew=1.5,
                   zorder=2, label="second-best optimum")
        a.plot(best[:-1][order], y, ls="", marker="o", ms=7, color=BLUE, zorder=3, label="best fit")
        a.plot(xt[order], y, ls="", marker="D", ms=6.5, color=INK, zorder=4, label="true value")
        e = np.abs(best[:-1] - xt)
        sec = (f"\nsecond-best: cost {groups[1][0][-1]:.3f} ({len(groups[1])} start{'s' if len(groups[1]) > 1 else ''})"
               if args.second_best and len(groups) > 1 else "")
        a.set_title(f"{title}\nbest cost {best[-1]:.3f} ({nb}/40 starts), "
                    f"RMS error {np.sqrt(np.mean(e ** 2)):.3f}{sec}", loc="left", fontsize=10)
        a.set_xlim(-0.03, 1.03)
        a.set_xlabel("normalized value (0–1 over declared range)")
        a.grid(axis="x", color="#eeeeee", lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(order)))
    sob = lambda n: f"{imp[n]:.2f}" if n in imp and n not in ("lambda_low", "lambda_high") else "fixed"
    axes[0].set_yticklabels([f"{names[j].replace('_factor', '')}  ({sob(names[j])})"
                             for j in order], fontsize=9)
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=10)
    fig.suptitle("Twin experiments: true vs. fitted parameters (4 variables, Student-t, 40 starts)",
                 x=0.01, ha="left", fontsize=13.5, weight="bold", color=INK)
    fig.text(0.01, 0.935, "Rows ordered by the largest total-order Sobol index of the outputs over the "
             "4 variables (in brackets; the lambdas were held fixed in that analysis); a line joins each true value to the best fit.",
             fontsize=9, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
