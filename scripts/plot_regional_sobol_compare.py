"""
Regional Sobol total-order indices of several surrogates in one figure: one row per
surrogate (regional_sobol.npz from scripts/regional_sobol.py), one column per
variable, all boxes. Per parameter: median (dot), --pct range (bar) and maximum
(tick, with its value). Parameters are ordered by the first file's all-box mean ST
and each column shares its x-axis across rows. Needs only numpy/matplotlib.

    python scripts/plot_regional_sobol_compare.py \\
        --npz .../regional_sobol_n85/regional_sobol.npz \\
              .../regional_sobol_n85_ne128ne32/regional_sobol.npz \\
              .../regional_sobol_n85_ne32/regional_sobol.npz \\
        --labels "3-level" "2-level (ne128 + ne32)" "1-level (ne32 only)" \\
        --out docs/regional_sobol_n85_compare_index_allboxes.png
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--pct", type=float, nargs=2, default=[5, 95])
    ap.add_argument("--overlay", action="store_true",
                    help="One row: the surrogates side by side per parameter, colour-coded")
    ap.add_argument("--title", default="Regional Sobol total-order indices, all boxes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.npz) == len(args.labels)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [np.load(p) for p in args.npz]
    params = [str(p) for p in runs[0]["params"]]
    var_names = [str(v) for v in runs[0]["variables"]]
    for r in runs:
        assert [str(p) for p in r["params"]] == params and r["done"].all()
    order = np.argsort(-runs[0]["ST"].mean(axis=(0, 1)))
    xmax = [1.16 * max(r["ST"][:, vi].max() for r in runs) for vi in range(len(var_names))]
    y = np.arange(len(params))

    if args.overlay:
        overlay(runs, args, params, var_names, order, xmax, y, plt)
        return
    nr, nc = len(runs), len(var_names)
    fig, axes = plt.subplots(nr, nc, figsize=(4.1 * nc, 5.6 * nr), sharey=True, squeeze=False)
    for ri, (r, lab) in enumerate(zip(runs, args.labels)):
        for vi, v in enumerate(var_names):
            ax = axes[ri, vi]
            a = r["ST"][:, vi][:, order]
            lo, med, hi = np.percentile(a, [args.pct[0], 50, args.pct[1]], axis=0)
            mx = a.max(axis=0)
            ax.hlines(y, lo, hi, color="#8a8f98", lw=2.2)
            ax.plot(mx, y, "|", ms=6, mew=1.4, color="#6b6f76")
            ax.plot(med, y, "o", ms=4.5, color="#4a4f57", mec="white", mew=0.6)
            for yy, m_ in zip(y, mx):
                ax.text(m_, yy, f" {m_:.2f}", va="center", ha="left", fontsize=6.5, color="#4a4f57")
            ax.set_xlim(0, xmax[vi])
            ax.grid(axis="x", color="#e3e5e8", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if ri == 0:
                ax.set_title(v, fontsize=12)
            if ri == nr - 1:
                ax.set_xlabel("total-order index ST", fontsize=9)
        axes[ri, 0].set_ylabel(lab, fontsize=11, labelpad=8)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels([params[i] for i in order])
    axes[0, 0].invert_yaxis()
    for ax in axes[:, 0]:
        ax.tick_params(axis="y", labelsize=8.5, labelleft=True)
    fig.suptitle(args.title, y=0.997, fontsize=13)
    n_boxes, n_bins = len(runs[0]["done"]), int(runs[0]["n_bins"])
    fig.text(0.01, 0.004, f"{n_boxes} boxes per surrogate, each 1/{n_bins} of every parameter's range "
             f"(same boxes for all surrogates); dot = median over boxes, bar = {args.pct[0]:g}-"
             f"{args.pct[1]:g}%, tick and number = maximum. Parameters in the order of the first row.",
             fontsize=8, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.012, 1, 0.985))
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


COLORS = ["#2a6fdb", "#d9480f", "#2b8a3e", "#7048e8"]


def overlay(runs, args, params, var_names, order, xmax, y, plt):
    nc, n = len(var_names), len(runs)
    off = (np.arange(n) - (n - 1) / 2) * 0.24
    fig, axes = plt.subplots(1, nc, figsize=(4.4 * nc, 0.62 * len(params) + 2.2), sharey=True)
    for vi, v in enumerate(var_names):
        ax = axes[vi]
        for j, (r, lab) in enumerate(zip(runs, args.labels)):
            a = r["ST"][:, vi][:, order]
            lo, med, hi = np.percentile(a, [args.pct[0], 50, args.pct[1]], axis=0)
            mx = a.max(axis=0)
            yy = y + off[j]
            ax.hlines(yy, lo, hi, color=COLORS[j], lw=2, alpha=0.9)
            ax.plot(mx, yy, "|", ms=5, mew=1.4, color=COLORS[j])
            ax.plot(med, yy, "o", ms=4, color=COLORS[j], mec="white", mew=0.6,
                    label=lab if vi == 0 else None)
            for y_, m_ in zip(yy, mx):
                ax.text(m_, y_, f" {m_:.2f}", va="center", ha="left", fontsize=5.5, color="#4a4f57")
        ax.set_xlim(0, xmax[vi])
        ax.set_title(v, fontsize=12)
        ax.set_xlabel("total-order index ST", fontsize=9)
        ax.grid(axis="x", color="#e3e5e8", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for k in range(1, len(params)):
            ax.axhline(k - 0.5, color="#f0f1f3", lw=0.6, zorder=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([params[i] for i in order], fontsize=9)
    axes[0].set_ylim(len(params) - 0.5, -0.5)
    fig.legend(loc="upper center", ncol=n, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(args.title, y=0.995, fontsize=13)
    n_boxes, n_bins = len(runs[0]["done"]), int(runs[0]["n_bins"])
    fig.text(0.01, 0.006, f"{n_boxes} boxes per surrogate, each 1/{n_bins} of every parameter's range "
             f"(same boxes for all surrogates); dot = median over boxes, bar = {args.pct[0]:g}-"
             f"{args.pct[1]:g}%, tick and number = maximum. Parameters in the order of "
             f"{args.labels[0]}.", fontsize=8, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
