"""
Optimized parameter values from every multi-start, for several runs side by side.

Each row is one parameter; within a row each run gets its own lane. Every dot
is the endpoint of one optimizer start (read from the per-start checkpoints,
[19 normalized params, cost]). The x axis is the position within the range the
PPE sampled for that parameter (the optimizer's bounds_from_data box), with the
physical end values printed on either side, so all 19 parameters share one axis.

Dots within --delta of that run's best cost are drawn solid, the rest faded;
the best start is a black-edged diamond. Where the runs' solid dots sit apart
for a parameter, the runs disagree about it; where one run's solid dots spread
along a lane, that run's cost is flat in that parameter.

Usage:
    python scripts/plot_start_endpoints_compare.py \
        --run "warm=/path/results_mf3_ne256_studentt_dof4/optimize_checkpoints" \
        --run "cold=/path/results_mf3_ne256_studentt_dof4_COLD/optimize_checkpoints" \
        --scalers .../preprocess_output_ne128_prod_v2/scalers.pkl \
        --names-from /path/results_mf3_ne256_studentt_dof4/diagnostics/parameter_uncertainty_seed50.pkl \
        --out docs/warm_vs_cold_endpoints.png
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

SURFACE = "#fcfcfb"
TEXT2 = "#52514e"
COLORS = ["#2f6db3", "#d9792b", "#3a9a5b", "#9b4fb0"]


def _load(ckpt_dir):
    m = pickle.load(open(Path(ckpt_dir) / "manifest.pkl", "rb"))
    A = np.array([pickle.load(open(f, "rb"))
                  for f in sorted(glob.glob(str(Path(ckpt_dir) / "start_*.pkl")))], dtype=float)
    if not len(A):
        raise SystemExit(f"No start_*.pkl in {ckpt_dir}")
    return A[:, :-1], A[:, -1], np.asarray(m["bounds_low"]), np.asarray(m["bounds_high"])


def _fmt(v):
    a = abs(v)
    return f"{v:.2e}" if (a and (a < 1e-2 or a >= 1e4)) else f"{v:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=checkpoint_dir")
    ap.add_argument("--scalers", default=None, help="scalers.pkl whose X_pipeline maps to physical units")
    ap.add_argument("--names-from", default=None, help="parameter_uncertainty*.pkl holding param_names")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="starts within this cost of their run's best are drawn solid")
    ap.add_argument("--best-only", action="store_true",
                    help="draw only each run's best start (filled diamond)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, d = spec.split("=", 1)
        runs.append((label, *_load(d)))
    n_params = runs[0][1].shape[1]
    lo = np.min([r[3] for r in runs], axis=0)       # common box over all runs
    hi = np.max([r[4] for r in runs], axis=0)

    names = (list(pickle.load(open(args.names_from, "rb"))["param_names"])
             if args.names_from else [f"p{i}" for i in range(n_params)])
    phys_lo = phys_hi = None
    if args.scalers:
        xs = pickle.load(open(args.scalers, "rb"))["X_pipeline"]
        phys_lo = xs.inverse_transform(lo[None])[0]
        phys_hi = xs.inverse_transform(hi[None])[0]

    nr = len(runs)
    lane = 0.62 / nr
    fig, ax = plt.subplots(figsize=(12, 0.55 * n_params + 2.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for k, (label, P, c, _, _) in enumerate(runs):
        U = (P - lo) / np.where(hi > lo, hi - lo, 1)
        best = int(np.argmin(c))
        near = c <= c[best] + args.delta
        if args.best_only:
            ys = np.arange(n_params) - 0.31 + lane * (k + 0.5)
            ax.scatter(U[best], ys, s=90, marker="D", color=COLORS[k],
                       edgecolor="black", lw=0.9, zorder=5)
            ax.scatter([], [], s=50, marker="D", color=COLORS[k], edgecolor="black",
                       label=f"{label}: best start (cost {c[best]:.4f})")
            continue
        for i in range(n_params):
            y = i - 0.31 + lane * (k + 0.5)
            jit = rng.uniform(-lane * 0.3, lane * 0.3, len(c))
            ax.scatter(U[~near, i], y + jit[~near], s=14, color=COLORS[k], alpha=0.18, lw=0)
            ax.scatter(U[near, i], y + jit[near], s=22, color=COLORS[k], alpha=0.85, lw=0,
                       zorder=6)
            # Hollow and behind the dots: starts that converge to the same
            # point would otherwise vanish under a filled marker.
            ax.scatter(U[best, i], y, s=110, marker="D", facecolor="none",
                       edgecolor="black", lw=1.1, zorder=5)
        ax.scatter([], [], s=40, color=COLORS[k],
                   label=f"{label}: best {c[best]:.4f}, {near.sum()}/{len(c)} starts within Δ={args.delta}")

    for i in range(n_params + 1):
        ax.axhline(i - 0.5, color="k", lw=0.4, alpha=0.2)
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(n_params - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Position within the sampled range (0 = PPE minimum, 1 = PPE maximum)")
    if phys_lo is not None:
        for i in range(n_params):
            ax.text(1.035, i, f"{_fmt(phys_lo[i])} – {_fmt(phys_hi[i])}", ha="left", va="center",
                    fontsize=7.5, color=TEXT2, transform=ax.get_yaxis_transform())
        ax.text(1.035, -0.75, "sampled range", ha="left", va="bottom", fontsize=7.5,
                color=TEXT2, transform=ax.get_yaxis_transform(), style="italic")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, frameon=False, fontsize=9)
    ax.set_title(args.title or "Optimized parameter values for every start",
                 fontsize=12, pad=10)
    note = ("Diamond = each run's best start (lowest cost over all its starts)."
            if args.best_only else "Hollow diamond = best start. Solid dots = starts within Δ of that run's best "
             "cost; faded = the rest.")
    fig.text(0.5, 0.005, note + " Right: physical range the PPE sampled (maps to 0–1).",
             ha="center", fontsize=8, color=TEXT2)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
