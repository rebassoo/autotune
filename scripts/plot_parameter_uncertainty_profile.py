"""
Parameter-uncertainty plot using profile-likelihood 68% bounds (Wilks'
theorem, all 18 other params re-optimized at each step -- see
scripts/profile_likelihood.py) instead of the inverse-Hessian/Laplace
approximation (autotune_gp.uncertainty). Unlike that approximation, this
works for parameters pinned at a bound (no NaNs) and makes no local-quadratic
assumption -- but a direction that never crosses the Wilks threshold before
hitting the search-box edge is reported as an OPEN interval (a dashed arrow),
not a fabricated finite number.

Same per-parameter-box rescaling convention as
autotune_gp.diagnostics.plot_parameter_uncertainty: y=0 is that parameter's
own sampled minimum, y=1 its own sampled maximum, so every column's box is
exactly [0,1].

Usage:
    python scripts/plot_parameter_uncertainty_profile.py \
        --profile-pkl /path/to/results_..._laplace/diagnostics/profile_likelihood.pkl \
        --uncertainty-pkl /path/to/results_..._laplace/diagnostics/parameter_uncertainty_seed50.pkl \
        --out /path/to/results_..._laplace/diagnostics/parameter_uncertainty_profile_seed50.png
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile-pkl", required=True)
    p.add_argument("--uncertainty-pkl", required=True,
                    help="Original Hessian-based parameter_uncertainty pkl, used only "
                         "to flag which params were NaN/pinned there for comparison.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.profile_pkl, "rb") as f:
        prof = pickle.load(f)
    with open(args.uncertainty_pkl, "rb") as f:
        hess = pickle.load(f)

    names = prof["param_names"]
    x_best = np.asarray(prof["x_best"], dtype=float)
    bl = np.asarray(prof["bounds_low"], dtype=float)
    bh = np.asarray(prof["bounds_high"], dtype=float)
    results = prof["results"]
    n = len(names)

    hess_nan = np.isnan(np.asarray(hess["stderr"], dtype=float))
    hess_at_bound = np.asarray(hess.get("at_bound", np.zeros(n, dtype=bool)), dtype=bool)
    hess_pinned = hess_nan & hess_at_bound

    span = np.where(bh > bl, bh - bl, 1.0)
    y_best = (x_best - bl) / span

    lo_val = np.array([results[i][-1]["bound_val"] for i in range(n)])
    hi_val = np.array([results[i][+1]["bound_val"] for i in range(n)])
    lo_edge = np.array([results[i][-1]["hit_edge"] for i in range(n)])
    hi_edge = np.array([results[i][+1]["hit_edge"] for i in range(n)])
    y_lo = (lo_val - bl) / span
    y_hi = (hi_val - bl) / span

    n_edge_total = int(np.sum(lo_edge) + np.sum(hi_edge))
    n_cross_total = int(np.sum(~lo_edge) + np.sum(~hi_edge))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 6.2))
    xpos = np.arange(n)
    ax.axhspan(0, 1, color="0.9", zorder=0)

    OPEN_COLOR = "tab:orange"
    BOUND_COLOR = "tab:blue"
    cap_w = 0.14

    for i in range(n):
        # lower side
        if lo_edge[i]:
            ax.annotate("", xy=(xpos[i], -0.09), xytext=(xpos[i], y_best[i]),
                        arrowprops=dict(arrowstyle="-|>", color=OPEN_COLOR, ls="--", lw=1.5),
                        zorder=2)
        else:
            ax.plot([xpos[i], xpos[i]], [y_lo[i], y_best[i]], color=BOUND_COLOR, lw=1.5, zorder=2)
            ax.plot([xpos[i] - cap_w, xpos[i] + cap_w], [y_lo[i], y_lo[i]],
                    color=BOUND_COLOR, lw=1.5, zorder=2)
        # upper side
        if hi_edge[i]:
            ax.annotate("", xy=(xpos[i], 1.09), xytext=(xpos[i], y_best[i]),
                        arrowprops=dict(arrowstyle="-|>", color=OPEN_COLOR, ls="--", lw=1.5),
                        zorder=2)
        else:
            ax.plot([xpos[i], xpos[i]], [y_best[i], y_hi[i]], color=BOUND_COLOR, lw=1.5, zorder=2)
            ax.plot([xpos[i] - cap_w, xpos[i] + cap_w], [y_hi[i], y_hi[i]],
                    color=BOUND_COLOR, lw=1.5, zorder=2)

    ax.scatter(xpos, y_best, color="black", s=22, zorder=3, label="x_best")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="black", marker="o", ls="none", markersize=5, label="x_best"),
        Line2D([0], [0], color=BOUND_COLOR, lw=1.5, label="genuine 68% bound (crossed threshold)"),
        Line2D([0], [0], color=OPEN_COLOR, lw=1.5, ls="--",
               label="open (hit search-box edge before threshold)"),
    ]
    fig.legend(handles=legend_elems, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)

    labels = [nm + (" *" if hess_pinned[k] else "") for k, nm in enumerate(names)]
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylim(-0.22, 1.22)
    ax.set_ylabel("Fraction of parameter's own sampled range\n(0 = its min, 1 = its max)")
    title = (f"Best-fit parameters with profile-likelihood 68% bounds "
             f"(Wilks' theorem, all 18 other params re-optimized per step)\n"
             f"{n_edge_total}/{n_edge_total + n_cross_total} directions hit the search-box edge "
             f"(open interval) rather than crossing the threshold; "
             f"{n_cross_total}/{n_edge_total + n_cross_total} show a genuine bound.\n"
             f"* = pinned at bound under the Hessian/Laplace approximation (NaN there)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
