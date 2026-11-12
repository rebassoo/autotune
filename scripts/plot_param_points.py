"""
The 19 parameters as a dot plot: 4-variable optimum, 10-variable best optimum and
10-variable second optimum (cost ranks 5-32), on the normalized scale, with the
default and the range the PPE sampled. Reads the CSV from plot_param_table.py.

    python scripts/plot_param_points.py \\
        --csv docs/param_table_4var_10var.csv --out docs/param_points_4var_10var.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/perlmutter_mf3_ne256_n85_10var_annual.yaml"
INK, MUTED = "#1a1a1a", "#6b6f76"
FITS = [("4-var", "4-variable optimum", "#2a6fdb", "o", -0.22),
        ("10-var best", "10-variable best (rank 1–4)", "#e8833a", "s", 0.0),
        ("10-var rank 5–32", "10-variable second (rank 5–32)", "#2e9e5b", "^", 0.22)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    bounds = yaml.safe_load(open(CONFIG))["optimize"]["param_physical_bounds"]
    n = len(rows)

    fig, ax = plt.subplots(figsize=(13, 10))
    for i, r in enumerate(rows):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f7f8", lw=0, zorder=0)
        ax.plot([float(r["sampled_low_norm"]), float(r["sampled_high_norm"])], [y, y], color="#c9cdd2",
                lw=9, solid_capstyle="butt", zorder=1, label="range the PPE sampled" if i == 0 else None)
        ax.plot(float(r["Default_norm"]), y, marker="x", ms=9, mew=2, color=INK, ls="", zorder=2,
                label="default" if i == 0 else None)
        for key, lab, c, m, dy in FITS:
            ax.plot(float(r[f"{key}_norm"]), y + dy, marker=m, ms=9, color=c, mec="white", mew=1, ls="",
                    zorder=3, label=lab if i == 0 else None)
        lo, hi = bounds[r["parameter"]]
        ax.text(1.035, y, f"{lo:g} – {hi:g}", va="center", fontsize=9, color=MUTED)
    ax.text(1.035, n - 0.35, "declared range", va="bottom", fontsize=9, color=MUTED, weight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["parameter"] + (" (log)" if r["log_scaled"] == "True" else "")
                        for r in reversed(rows)], fontsize=10.5, color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("normalized value (0–1 over the declared range; log space for (log) parameters)", fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e3e5e8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("The 19 parameters at the 4-variable and 10-variable optima", x=0.012, y=0.985, ha="left",
                 fontsize=15, weight="bold", color=INK)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.96), ncol=5,
               frameon=False, fontsize=10, handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.235, right=0.875, top=0.91, bottom=0.07)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
