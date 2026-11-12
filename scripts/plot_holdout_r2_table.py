"""
Holdout R^2 at the top fidelity as a PNG table: the 3-level MF surrogate vs a
GP trained on the top-fidelity members alone, per variable, in normalized and
physical units. Both holdouts must use the same folds over the same members.

    python scripts/plot_holdout_r2_table.py \\
        --mf /pscratch/sd/r/rebassoo/autotune/results_ne256_n85_3level_holdout_ISO/nfidelity_holdout.pkl \\
        --alone /pscratch/sd/r/rebassoo/autotune/results_ne256_n85_alone_holdout/ne256_alone_holdout.pkl \\
        --out docs/holdout_r2_n85.png
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, RULE, HEAD = "#1a1a1a", "#6b6f76", "#d0d4da", "#eef2f8"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mf", required=True, help="nfidelity_holdout.pkl")
    ap.add_argument("--alone", required=True, help="ne256_alone_holdout.pkl")
    ap.add_argument("--title", default="Top-fidelity (ne256) holdout skill, n = 85")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mf = pickle.load(open(args.mf, "rb"))
    al = pickle.load(open(args.alone, "rb"))
    if (mf["folds"], mf["seed"]) != (al["folds"], al["seed"]) or \
            not np.allclose(mf["Y_true"], al["truth"], equal_nan=True):
        raise SystemExit("The two holdouts do not share folds and members; not comparable.")
    rows_mf = {r[0]: r for r in mf["rows_mf"]}
    rows_al = {r[0]: r for r in al["rows"]}
    names = list(mf["var_names"])

    # columns: variable | normalized (MF, alone) | physical (MF, alone)
    body = []
    for v in names:
        m, a = rows_mf[v], rows_al[v]
        body.append([v, m[1], a[1], m[3], a[3]])
    body.append(["Mean"] + [float(np.mean([r[k] for r in body])) for k in range(1, 5)])

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.set_axis_off()
    x = [0.03, 0.30, 0.50, 0.72, 0.92]          # column centres (variable left-aligned)
    y0, dy = 0.70, 0.105
    ax.text(0.01, 1.0, args.title, fontsize=13, weight="bold", color=INK, va="top",
            transform=ax.transAxes)
    ax.text(0.01, 0.92, f"R² on held-out members, {mf['folds']}-fold cross-validation "
            f"(17 held out per fold), predictions pooled over folds", fontsize=9, color=MUTED,
            va="top", transform=ax.transAxes)
    # group headers
    for xc, lab in ((0.40, "normalized units"), (0.82, "physical units")):
        ax.text(xc, y0 + 0.11, lab, ha="center", fontsize=9.5, color=MUTED,
                transform=ax.transAxes)
        ax.plot([xc - 0.17, xc + 0.17], [y0 + 0.095] * 2, color=RULE, lw=1,
                transform=ax.transAxes)
    heads = ["Variable", "3-level MF", "ne256 alone", "3-level MF", "ne256 alone"]
    ax.add_patch(plt.Rectangle((0.0, y0 - 0.045), 1.0, 0.09, color=HEAD, lw=0,
                               transform=ax.transAxes))
    for k, h in enumerate(heads):
        ax.text(x[k], y0, h, ha="left" if k == 0 else "center", va="center", fontsize=10,
                weight="bold", color=INK, transform=ax.transAxes)
    for i, row in enumerate(body):
        y = y0 - (i + 1) * dy
        last = i == len(body) - 1
        if last:
            ax.plot([0, 1], [y + dy / 2] * 2, color=INK, lw=0.8, transform=ax.transAxes)
        for k, val in enumerate(row):
            txt = val if k == 0 else f"{val:.2f}"
            ax.text(x[k], y, txt, ha="left" if k == 0 else "center", va="center",
                    fontsize=10.5, color=INK,
                    weight="bold" if (last or k in (1, 3)) else "normal",
                    transform=ax.transAxes)
        if not last:
            ax.plot([0, 1], [y - dy / 2] * 2, color=RULE, lw=0.6, transform=ax.transAxes)
    ax.text(0.01, 0.10,
            "3-level MF: ne32 (984) → ne128 (227) → ne256 AR GP, iso top kernel, lower levels frozen; "
            "the held-out\nne256 members' parameters can still appear in the lower levels. "
            "ne256 alone: single-fidelity GP on the ne256 training members.",
            fontsize=7.8, color=MUTED, va="top", transform=ax.transAxes, linespacing=1.5)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
