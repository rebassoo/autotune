"""
Per-variable Pearson correlation heatmaps: ZRG locations on x, parameters on y,
one panel per resolution.

Reads the CSV written by plot_param_output_correlation.py (which holds every
location, not just the global mean) so nothing is recomputed.

Usage:
    python scripts/plot_correlation_by_location.py \
        --csv docs/param_output_correlation_3levels.csv \
        --labels-from /path/to/results_dir/diagnostics/projection_data_seed50.pkl \
        --out-dir docs
"""
from __future__ import annotations

import argparse
import csv
import glob
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import stats

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [
    "#184f95", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#9e2b2a",
])
SHORT = {
    "poles": "poles", "extratropical_land": "extratrop land",
    "extratropical_ocean": "extratrop ocean", "tropical_land": "trop land",
    "ascending_tropical_ocean": "asc trop ocn",
    "descending_tropical_ocean": "desc trop ocn", "global": "GLOBAL",
}


def _label(raw: str) -> str:
    try:
        c = float(raw)
        return f"{abs(c):g}°{'S' if c < 0 else 'N'}"
    except ValueError:
        return SHORT.get(raw, raw)


def _r_crit(n: int, alpha: float = 0.05) -> float:
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(t / np.sqrt(t ** 2 + n - 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--labels-from", required=True,
                    help="A projection_data*.pkl, for the ZRG location labels.")
    ap.add_argument("--levels", default="ne32,ne128,ne256")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="correlation_by_location")
    args = ap.parse_args()

    with open(sorted(glob.glob(args.labels_from))[-1], "rb") as f:
        labels = [_label(l) for l in pickle.load(f)["zrg_labels"]]

    rows = list(csv.DictReader(open(args.csv)))
    levels = [l.strip() for l in args.levels.split(",")]
    params, variables = [], []
    for r in rows:
        if r["parameter"] not in params:
            params.append(r["parameter"])
        if r["variable"] not in variables:
            variables.append(r["variable"])
    n_loc = len(labels)

    R = {(lv, v): np.full((len(params), n_loc), np.nan) for lv in levels for v in variables}
    P = {k: np.full_like(v, np.nan) for k, v in R.items()}
    n_mem = {}
    pidx = {p: i for i, p in enumerate(params)}
    for r in rows:
        lv = r["level"]
        if lv not in levels:
            continue
        n_mem[lv] = int(r["n_members"])
        R[(lv, r["variable"])][pidx[r["parameter"]], int(r["zrg_index"])] = float(r["pearson_r"])
        P[(lv, r["variable"])][pidx[r["parameter"]], int(r["zrg_index"])] = float(r["p_value"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in variables:
        fig, axes = plt.subplots(1, len(levels),
                                 figsize=(2.0 + 5.0 * len(levels), 0.42 * len(params) + 3.4),
                                 facecolor=SURFACE, squeeze=False)
        fig.subplots_adjust(wspace=0.10)
        for li, (ax, lv) in enumerate(zip(axes[0], levels)):
            vals, sig = R[(lv, v)], P[(lv, v)] < 0.05
            ax.imshow(np.where(np.isfinite(vals), vals, 0.0), cmap=DIVERGING,
                      vmin=-1, vmax=1, aspect="auto")
            # fade the cells that do not reach p < 0.05
            ax.imshow(np.where(sig, np.nan, 1.0),
                      cmap=matplotlib.colors.ListedColormap([SURFACE]),
                      alpha=0.66, aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(n_loc))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels(params if li == 0 else [""] * len(params), fontsize=8.5)
            n = n_mem.get(lv, 0)
            ax.set_title(f"{lv}   n = {n}   |r| > {_r_crit(n):.2f} for p < 0.05",
                         fontsize=11, pad=8)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_color(MUTED)
            # separator between the zonal bands and the named regions
            n_zonal = sum(1 for l in labels if l.endswith("S") or l.endswith("N"))
            ax.axvline(n_zonal - 0.5, color=TEXT, lw=1.4)
            ax.axvline(n_loc - 1.5, color=TEXT, lw=1.0, ls=":")

        sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
        cb = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.016, pad=0.015)
        cb.set_label("Pearson r", fontsize=10, color=TEXT2)
        cb.outline.set_edgecolor(MUTED)
        fig.suptitle(f"{v}: Pearson correlation between each parameter and each ZRG location"
                     "\nsolid line separates zonal bands from named regions; "
                     "dotted line marks GLOBAL; faded = not significant at p < 0.05",
                     fontsize=13, y=0.99)
        out = out_dir / f"{args.prefix}_{v}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
