"""
Posterior coverage of a twin experiment run with the PCA likelihood: for every
parameter, the posterior's 5-95% and 25-75% ranges (MCMC, second half of the chain),
the true value and the best fit. One figure per twin.

x is the normalized parameter (0-1 over the declared range); the grey band is the
range the PPE sampled (the prior box). Rows ordered by the largest total-order Sobol
index of the outputs over the 4 variables, as in plot_twin_params.py.

    python scripts/plot_twin_posterior.py --twin m060_3level \\
        --posterior posterior_2000steps.pkl --out docs/twin_posterior_m060.png
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
INK, BLUE, RED, BAND, GREY = "#1a1a1a", "#2a6fdb", "#d9542b", "#eceef1", "#9aa0a6"
POST_LIGHT, POST_DARK = "#c9d8f2", "#7fa3e0"
LAMBDAS = ("lambda_low", "lambda_high")


def hist_grid(s, xt, best, lo, hi, q5, q95, names, order, imp, inside, member, footer, out):
    """4 x 5 grid: each parameter's 1-D posterior histogram over its sampled (prior) range."""
    fig, axes = plt.subplots(4, 5, figsize=(17, 12.5))
    for a, j in zip(axes.flat, order):
        bins = np.linspace(lo[j], hi[j], 41)
        a.axhline(1 / (hi[j] - lo[j]), color=GREY, lw=1.2, ls=":", zorder=1)
        a.axvspan(q5[j], q95[j], ymax=0.035, color=INK, alpha=0.35, lw=0, zorder=1)
        a.hist(s[:, j], bins=bins, density=True, color=POST_DARK, alpha=0.85, lw=0, zorder=2)
        a.axvline(best[j], color=BLUE, lw=1.6, ls="--", zorder=3)
        a.axvline(xt[j], color=INK if inside[j] else RED, lw=2.2, zorder=4)
        n = names[j]
        sob = f"Sobol {imp[n]:.2f}" if n in imp and n not in LAMBDAS else "no Sobol index"
        a.set_title(f"{n.replace('_factor', '')}\n({sob})", fontsize=9.5,
                    color=INK if inside[j] else RED)
        a.set_xlim(lo[j], hi[j])
        a.set_yticks([])
        a.tick_params(labelsize=8)
        a.spines[["top", "right", "left"]].set_visible(False)
    for a in axes.flat[len(order):]:
        a.axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=POST_DARK, alpha=0.85, label="posterior (MCMC samples)"),
               Patch(color=INK, alpha=0.35, label="posterior 5–95% (bar on the axis)"),
               Line2D([], [], color=GREY, lw=1.2, ls=":", label="flat prior (no information)"),
               Line2D([], [], color=INK, lw=2.2, label="true value"),
               Line2D([], [], color=RED, lw=2.2, label="true value outside 5–95%"),
               Line2D([], [], color=BLUE, lw=1.6, ls="--", label="best fit (lowest cost)")]
    axes.flat[-1].legend(handles=handles, loc="center", frameon=False, fontsize=10)
    fig.suptitle(f"{member} twin (3-level surrogate, PCA likelihood): 1-D posteriors and true parameters — "
                 f"true value inside 5–95% for {int(inside.sum())} of {len(names)}",
                 x=0.01, ha="left", fontsize=13, weight="bold", color=INK)
    fig.text(0.01, 0.005, footer + "\nx: normalized value (0–1 over the declared range), each panel spanning the range the PPE "
             "sampled (the prior box). Panels ordered by "
             "largest output Sobol index.", fontsize=8.5, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Saved {out}: inside {int(inside.sum())}/{len(names)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", required=True, help="e.g. m060_3level")
    ap.add_argument("--posterior", default="posterior_2000steps.pkl")
    ap.add_argument("--sobol", default=str(S / "sobol_n85" / "sobol_summary.csv"))
    ap.add_argument("--kind", choices=["ranges", "hist"], default="ranges",
                    help="ranges: one row per parameter; hist: a 4x5 grid of 1-D posterior histograms")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L = lambda p: pickle.load(open(p, "rb"))
    tw, rd = S / "twin_preprocess" / args.twin, S / f"results_twin_{args.twin}_pca"
    truth = L(tw / "truth.pkl")
    names = list(truth["params"])
    xt = L(tw / "scalers.pkl")["X_pipeline"].transform(truth["x_phys"][None, :])[0]
    post = L(rd / "diagnostics" / args.posterior)
    if list(post["param_names"]) != names:
        raise SystemExit("posterior parameters differ from truth.pkl")
    ch = post["chain"]
    n_steps = ch.shape[0]
    s = ch[n_steps // 2:].reshape(-1, ch.shape[2])
    try:
        import emcee
        tau = float(np.max(emcee.autocorr.integrated_time(ch, quiet=True)))
    except Exception:
        tau = float("nan")
    q5, q25, q50, q75, q95 = np.percentile(s, [5, 25, 50, 75, 95], axis=0)
    lo, hi = (np.asarray(b) for b in post["bounds"])
    rows = [np.asarray(L(f), float) for f in sorted((rd / "optimize_checkpoints").glob("start_*.pkl"))]
    best = clusters(rows)[0][0][:-1]

    imp = {r["parameter"]: max(float(v) for k, v in r.items() if k.startswith("ST_") and "misfit" not in k)
           for r in csv.DictReader(open(args.sobol))}
    order = sorted(range(len(names)), key=lambda j: imp.get(names[j], 0.0) if names[j] not in LAMBDAS else -1)
    y = np.arange(len(order))
    inside = (xt >= q5) & (xt <= q95)

    member = args.twin.split("_")[0]
    footer = (f"MCMC: 64 walkers × {n_steps} steps, second half kept ({len(s):,} samples); longest "
              f"autocorrelation time {tau:.0f} steps ({n_steps / tau:.0f} × τ; ≥ 50 × τ is the usual target).")
    if args.kind == "hist":
        hist_grid(s, xt, best, lo, hi, q5, q95, names, order[::-1], imp, inside, member, footer, args.out)
        return

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(y, (hi - lo)[order], left=lo[order], height=0.7, color=BAND, lw=0, zorder=0,
            label="range sampled by the PPE (prior)")
    ax.barh(y, (q95 - q5)[order], left=q5[order], height=0.42, color=POST_LIGHT, lw=0, zorder=1,
            label="posterior 5–95%")
    ax.barh(y, (q75 - q25)[order], left=q25[order], height=0.42, color=POST_DARK, lw=0, zorder=2,
            label="posterior 25–75%")
    ax.vlines(q50[order], y - 0.21, y + 0.21, color=BLUE, lw=1.6, zorder=3, label="posterior median")
    ax.plot(best[order], y, ls="", marker="o", ms=6.5, mfc="white", mec=BLUE, mew=1.6, zorder=4,
            label="best fit (lowest cost)")
    ax.plot(xt[order], y, ls="", marker="D", ms=7, color=INK, zorder=5, label="true value")
    miss = [k for k, j in enumerate(order) if not inside[j]]
    ax.plot(xt[order][miss], y[miss], ls="", marker="D", ms=7, color=RED, zorder=6,
            label="true value outside 5–95%")
    sob = lambda n: f"{imp[n]:.2f}" if n in imp and n not in LAMBDAS else "no Sobol index"
    ax.set_yticks(y)
    ax.set_yticklabels([f"{names[j].replace('_factor', '')}  ({sob(names[j])})" for j in order], fontsize=9)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("normalized value (0–1 over declared range)")
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=3)
    ax.set_title(f"{member} twin (3-level surrogate, PCA likelihood): posterior vs. true parameters\n"
                 f"true value inside the posterior 5–95% range for {int(inside.sum())} of {len(names)} "
                 "parameters", loc="left", fontsize=11)
    fig.text(0.01, 0.005, footer + " In brackets: largest output Sobol index.", fontsize=8, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}: inside {int(inside.sum())}/{len(names)}")


if __name__ == "__main__":
    main()
