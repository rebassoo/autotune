"""
Best-fit parameters of the full optimization next to runs that froze some
parameters, to show whether the free ones move when the others are fixed.

With no --frozen it is a plain comparison of the runs' optima. Each run contributes the best point from its diagnostics/projection_data_seed50.pkl;
x is the normalized parameter (0-1 over the scaler range), the shaded band each
parameter's search box. Frozen parameters are labelled in the y axis and drawn
hollow for the frozen runs. The legend gives each run's best cost, comparable only
when the runs share the cost, weights and normalization; --no-cost omits it.

    python scripts/plot_freeze_optima.py \\
        --run "all 19 free=/pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_studentt_dof4" \\
        --run "6 frozen at the fitted values=/pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_freeze6_fitted" \\
        --run "6 frozen at the control-run defaults=/pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_freeze6_default" \\
        --frozen deposition_nucleation_exponent thl2tune rain_selfcollection_breakup_diameter \\
                 autoconversion_radius rain_to_ice_collection_factor c_diag_3rd_mom \\
        --bounds-log logs/stage2b_mf3_ne256_n85_studentt_dof4_60176999.log \\
        --out docs/freeze6_optima_n85.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_calibrated_optima import _box_from_log   # noqa: E402

INK, GREY = "#1a1a1a", "#9aa0a6"
COLORS = [INK, "#2a6fdb", "#e8833a", "#8e44ad"]
MARKERS = ["o", "s", "D", "^"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=results dir; the first is the reference (nothing frozen)")
    ap.add_argument("--frozen", nargs="*", default=[],
                    help="Parameters held fixed in the runs after the first (none: plain comparison).")
    ap.add_argument("--names-config", default="configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml")
    ap.add_argument("--bounds-log", default=None)
    ap.add_argument("--no-cost", action="store_true",
                    help="Leave the best cost out of the legend (runs not comparable).")
    ap.add_argument("--title", default="Best fit with all parameters free vs. 6 low-Sobol parameters frozen")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    names = list(yaml.safe_load(open(args.names_config))["optimize"]["param_physical_bounds"])
    runs = []
    for r in args.run:
        label, d = r.split("=", 1)
        p = pickle.load(open(Path(d) / "diagnostics" / "projection_data_seed50.pkl", "rb"))
        runs.append((label, np.asarray(p["best_params_norm"]), float(p["best_cost"])))
    frozen = set(args.frozen)
    unknown = frozen - set(names)
    if unknown:
        raise SystemExit(f"--frozen names not in the parameter set: {sorted(unknown)}")

    # free parameters on top, frozen ones below a divider
    order = [n for n in names if n not in frozen] + [n for n in names if n in frozen]
    idx = [names.index(n) for n in order]
    n = len(order)
    y = np.arange(n)[::-1].astype(float)
    off = np.linspace(-0.25, 0.25, len(runs))
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * n + 2.2))
    if args.bounds_log:
        box = _box_from_log(args.bounds_log, order)
        for i in range(n):
            ax.fill_betweenx([y[i] - 0.42, y[i] + 0.42], box[i, 0], box[i, 1], color="#dbe7f7",
                             lw=0, zorder=0, label="search box (sampled range)" if i == 0 else None)
    x0 = runs[0][1][idx]
    for k in range(1, len(runs)):
        xk = runs[k][1][idx]
        for i in range(n):
            ax.plot([x0[i], xk[i]], [y[i] + off[0], y[i] + off[k]], color=GREY, lw=0.6, zorder=1)
    is_frozen = np.array([o in frozen for o in order])
    for k, (label, x, cost) in enumerate(runs):
        xx, yy = x[idx], y + off[k]
        hollow = is_frozen & (k > 0)
        ax.scatter(xx[~hollow], yy[~hollow], s=36, marker=MARKERS[k], color=COLORS[k],
                   edgecolor="white", lw=0.6, zorder=3, label=label if args.no_cost else f"{label} (cost {cost:.4f})")
        ax.scatter(xx[hollow], yy[hollow], s=36, marker=MARKERS[k], facecolor="white",
                   edgecolor=COLORS[k], lw=1.4, zorder=3)
    n_free = int((~is_frozen).sum())
    if frozen:
        ax.axhline(y[n_free - 1] - 0.5, color=INK, lw=0.8)
        ax.text(1.0, y[n_free - 1] - 0.55, "frozen in the freeze runs (Sobol ST < 0.01)",
                ha="right", va="top", fontsize=8, color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("normalized parameter")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=9, frameon=False)
    ax.set_title(args.title, pad=78 if args.bounds_log else 62, fontsize=11)
    fig.text(0.01, 0.003, ("Hollow = held fixed in that run. " if frozen else "")
             + "x: 0-1 over the scaler range.",
             fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(args.out, dpi=150)

    free = ~is_frozen
    for label, x, cost in runs[1:]:
        d = x[idx][free] - x0[free]
        print(f"{label}: free-parameter shift RMS {np.sqrt((d ** 2).mean()):.3f}, "
              f"max {np.abs(d).max():.3f} ({np.array(order)[free][np.abs(d).argmax()]})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
