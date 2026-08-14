"""
Combined ZRG value table across scenarios, regional+global rows only (no
zonal bands), all variables stacked into a single PNG.

Reuses run_diagnostics' saved projection_data*.pkl (same source as
plot_scenario_comparison.py's plot_zrg_table), but:
  - drops the zonal-band rows, keeping only regions + global -- useful for
    runs where zonal wasn't part of the optimized cost (weights.zrg.zonal=0),
    where the zonal rows only reflect wherever the optimizer incidentally
    landed rather than something it was actually fit to.
  - stacks all variables' tables into one figure/PNG instead of one PNG per
    variable, for easier side-by-side reading.

Cell colouring per row: green = closest to obs, red = furthest (same
convention as plot_scenario_comparison.plot_zrg_table).

Usage:
    python scripts/plot_zrg_table_combined.py \
        --dir /path/to/results_a --label "Scenario A" \
        --dir /path/to/results_b --label "Scenario B" \
        --dir /path/to/results_c --label "Scenario C" \
        --out-dir /path/to/out_dir
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN  = "#90EE90"
YELLOW = "#FFD700"
RED    = "#FF9999"
WHITE  = "white"


def _load_projection(results_dir: str):
    diag_dir = Path(results_dir) / "diagnostics"
    pkls = sorted(glob.glob(str(diag_dir / "projection_data*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No projection_data*.pkl found in {diag_dir}")
    with open(pkls[-1], "rb") as f:
        return pickle.load(f)


def _n_zonal(zrg_labels: list[str]) -> int:
    """Zonal-band labels are numeric (latitude); regions/global are not."""
    n = 0
    for lab in zrg_labels:
        try:
            float(lab)
        except ValueError:
            break
        n += 1
    return n


def plot_zrg_table_combined(scenarios: list[dict], out_dir: Path,
                            vars_to_plot: tuple, out_name: str = "comparison_table_regionsglobal.png"):
    """
    scenarios: list of dicts with keys: label, proj (projection_data dict).
    One combined PNG, all variables stacked, regional+global rows only.
    """
    var_names  = scenarios[0]["proj"]["var_names"]
    zrg_labels = scenarios[0]["proj"]["zrg_labels"]
    obs        = scenarios[0]["proj"]["obs"]        # (n_feat, n_vars)
    default    = scenarios[0]["proj"]["hf_default"]

    n_skip = _n_zonal(zrg_labels)
    keep_idx = list(range(n_skip, len(zrg_labels)))  # regions + global only
    labels_kept = [zrg_labels[i] for i in keep_idx]
    n_rows_per_var = len(keep_idx)

    col_labels = ["Feature", "Obs", "Default (m0)"] + [sc["label"] for sc in scenarios]
    n_sc = len(scenarios)
    n_cols = len(col_labels)

    vars_to_plot = [v for v in vars_to_plot if v in var_names]
    n_vars = len(vars_to_plot)

    fig_w = max(10, n_cols * 1.6)
    fig_h = n_vars * (n_rows_per_var * 0.32 + 0.9) + 0.5
    fig, axes = plt.subplots(n_vars, 1, figsize=(fig_w, fig_h))
    if n_vars == 1:
        axes = [axes]

    for ax, var in zip(axes, vars_to_plot):
        j = var_names.index(var)
        obs_vals = obs[keep_idx, j]
        def_vals = default[keep_idx, j]
        preds    = [sc["proj"]["hf_optimal"][keep_idx, j] for sc in scenarios]

        cell_text   = []
        cell_colors = []
        for i in range(n_rows_per_var):
            diffs = np.array([abs(p[i] - obs_vals[i]) for p in preds])
            order = np.argsort(diffs)

            pred_colors = [WHITE] * n_sc
            pred_colors[order[0]]  = GREEN
            pred_colors[order[-1]] = RED
            for k in range(1, n_sc - 1):
                pred_colors[order[k]] = YELLOW

            cell_colors.append([WHITE, WHITE, WHITE] + pred_colors)
            cell_text.append(
                [labels_kept[i], f"{obs_vals[i]:.3g}", f"{def_vals[i]:.3g}"]
                + [f"{preds[s][i]:.3g}" for s in range(n_sc)]
            )

        ax.axis("off")
        tbl = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=col_labels,
            colColours=[WHITE] * n_cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(n_cols)))
        ax.set_title(f"{var} — regions+global only (zonal dropped)", fontsize=10, pad=6)

    fig.suptitle("ZRG values by scenario  (green=closest to obs, red=furthest)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    path = out_dir / out_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", action="append", required=True, dest="dirs",
                   help="Results dir (repeatable, paired by position with --label).")
    p.add_argument("--label", action="append", required=True, dest="labels",
                   help="Label for the corresponding --dir (repeatable).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-name", default="comparison_table_regionsglobal.png")
    p.add_argument("--vars", default="PCP,OSR,OLR,TLWP",
                   help="Comma-separated variable order (default: PCP,OSR,OLR,TLWP).")
    args = p.parse_args()

    if len(args.dirs) != len(args.labels):
        p.error("--dir and --label must be given the same number of times.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for d, lbl in zip(args.dirs, args.labels):
        print(f"Loading projection data for {lbl!r} ...")
        scenarios.append(dict(label=lbl, proj=_load_projection(d)))

    plot_zrg_table_combined(scenarios, out_dir, tuple(args.vars.split(",")), args.out_name)
    print("Done.")


if __name__ == "__main__":
    main()
