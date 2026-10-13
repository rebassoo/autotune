"""
Summary table of surrogate skill (R^2) across resolutions and configurations.

Numbers are transcribed from the runs named in PROVENANCE below; each row says
which job produced it, so any figure can be traced back to a log.

Two kinds of number appear here and they are NOT interchangeable:
  k-fold  : members of that one ensemble split 5 ways; the surrogate predicts
            held-out members of the SAME resolution it was trained on.
  holdout : only the top (ne256) level is split; lower levels stay in training.
            'strict' additionally removes the ne32/ne128 members sitting at the
            held-out parameter point, so no level has seen it -- the regime the
            optimizer actually works in.

Usage:
    python scripts/make_surrogate_performance_table.py --out docs/surrogate_performance.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
HEADER = "#D9D9D9"
# Sequential single-hue blue ramp (dataviz reference palette), light -> dark.
SEQ = LinearSegmentedColormap.from_list("blue_seq", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
])

VARS = ["PCP", "OSR", "OLR", "TLWP"]

# (label, sub-label, [R2 per variable], provenance)
ROWS = [
    ("ne32 alone",  "single-fidelity, 984 members, 5-fold",
     [0.8702, 0.9268, 0.8786, 0.9126], "job 58879571"),
    ("ne128 alone", "single-fidelity, 218 members, 5-fold",
     [0.7960, 0.8401, 0.6807, 0.8204], "job 58172281"),
    ("ne256 alone", "single-fidelity GPy, 47 members, same folds as the holdouts",
     [0.4091, 0.4493, 0.1966, 0.3215], "matched-fold rerun"),
    ("ne32 + ne128", "2-level MF, skill at ne128, 5-fold",
     [0.8797, 0.9584, 0.8961, 0.9499], "job 58172259"),
    ("ne32 + ne128 + ne256", "3-level MF, skill at ne256, holdout",
     [0.8284, 0.9297, 0.9279, 0.9564], "job 58871667"),
    ("ne32 + ne128 + ne256", "3-level MF, skill at ne256, STRICT holdout",
     [0.7837, 0.8761, 0.7617, 0.9079], "job 58879570"),
    ("ne128 run as a proxy", "no surrogate: the ne128 simulation at the same params",
     [0.6807, 0.8150, 0.8187, 0.9237], "job 58879570 baseline"),
]


COMPARE = [
    ("ne32 + ne128 + ne256", "3-level MF, STRICT holdout (no level has seen the point)",
     [0.7837, 0.8761, 0.7617, 0.9079], "job 58879570"),
    ("ne32 + ne128 + ne256", "3-level MF, standard holdout (lower levels keep the point)",
     [0.8284, 0.9297, 0.9279, 0.9564], "job 58871667"),
    ("ne256 alone", "single-fidelity GPy on the 47 ne256 members only",
     [0.4091, 0.4493, 0.1966, 0.3215], "matched-fold rerun"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/surrogate_performance.png")
    ap.add_argument("--compare", action="store_true",
                    help="Three-way ne256 comparison instead of the full table.")
    args = ap.parse_args()

    rows = COMPARE if args.compare else ROWS
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13.2, 0.72 * n + 3.0), facecolor=SURFACE)
    ax.axis("off")

    x0, w_lab, w_col = 0.012, 0.44, 0.115
    y_top, h = 0.80, 0.088

    title = ("Predicting ne256: what the extra levels buy"
             if args.compare else "Surrogate skill by resolution and configuration")
    ax.text(x0, y_top + 0.115, title,
            fontsize=16, weight="bold", transform=ax.transAxes)
    ax.text(x0, y_top + 0.062,
            "R² on held-out members (normalised space). Deeper blue = better.",
            fontsize=10.5, color=TEXT2, transform=ax.transAxes)

    for j, v in enumerate(VARS):
        ax.text(x0 + w_lab + (j + 0.5) * w_col, y_top + 0.012, v, ha="center",
                fontsize=12, weight="bold", transform=ax.transAxes)

    for i, (lab, sub, vals, prov) in enumerate(rows):
        y = y_top - (i + 1) * h
        strict = "STRICT" in sub
        ax.text(x0, y + h * 0.60, lab, fontsize=11.5,
                weight="bold" if strict else "normal", transform=ax.transAxes)
        ax.text(x0, y + h * 0.22, sub, fontsize=8.8, color=TEXT2, transform=ax.transAxes)
        for j, r in enumerate(vals):
            cx = x0 + w_lab + j * w_col
            ax.add_patch(plt.Rectangle((cx, y + 0.008), w_col - 0.008, h - 0.016,
                                       facecolor=SEQ(float(np.clip(r, 0, 1))),
                                       edgecolor=SURFACE, lw=1.5,
                                       transform=ax.transAxes, clip_on=False))
            ax.text(cx + (w_col - 0.008) / 2, y + h * 0.42, f"{r:.3f}", ha="center",
                    va="center", fontsize=11,
                    color="white" if r > 0.62 else TEXT, transform=ax.transAxes)
        ax.text(x0 + w_lab + 4 * w_col + 0.012, y + h * 0.42, prov, fontsize=7.6,
                color=MUTED, va="center", transform=ax.transAxes)

    if args.compare:
        note = (
            "All three rows use the SAME 5 folds over the same 47 ne256 members "
            "(9-10 held out per fold,\n37-38 in training) and the same GPy/AR machinery, "
            "so the only difference is what each model is given.\n\n"
            "STRICT is the regime the optimizer works in: the parameter point being "
            "predicted was never\nsimulated at any resolution. The gap to the standard row "
            "is how much of the skill came from\nhaving cheap runs at exactly that point. "
            "The gap to ne256-alone is what the lower levels buy."
        )
    else:
        note = (
        "k-fold rows split one ensemble and predict held-out members of the same resolution.\n"
        "Holdout rows split only ne256; ne32 and ne128 stay in training. STRICT also removes the\n"
        "ne32/ne128 members at each held-out parameter point, so no level has seen it — this is the\n"
        "regime the optimizer works in, and the row to judge optimization results by.\n"
        "The last row is not a surrogate: it asks how well an ne128 run predicts ne256 at the same\n"
        "parameters. It needs a run at that point, so it is an oracle, unavailable during optimization."
    )
    ax.text(x0, y_top - (n + 1) * h - 0.035, note, fontsize=9.2, color=TEXT2,
            va="top", linespacing=1.55, transform=ax.transAxes)

    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
