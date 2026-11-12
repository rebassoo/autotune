"""
How the per-feature costs are combined into the one number the optimizer
minimizes: the weighted sum over variables, the averaging within each block,
and the resulting effective weight of each of the 4 x 22 = 88 features.

Reads the weights from a config (weights.variables, weights.zrg); per-variable
block overrides and feature masks (weights.zrg_per_variable) are applied.

    python scripts/plot_cost_aggregation.py \\
        --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \\
        --out docs/cost_aggregation_n85.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from autotune_gp.config import load_config   # noqa: E402

ZONAL = ["-65", "-55", "-45", "-35", "-25", "-15", "-5", "5", "15", "25", "35", "45", "55", "65", "75"]
REGIONS = ["poles", "extratrop\nland", "extratrop\nocean", "tropical\nland", "asc. trop\nocean",
           "desc. trop\nocean"]
INK, MUTED, ACCENT = "#1a1a1a", "#6b6f76", "#1b4f9c"


def weights(cfg):
    """(n_vars, 22) effective weight of each feature, as the cost applies them."""
    nz, nr = len(ZONAL), len(REGIONS)
    zrg = cfg.weights.zrg
    per = zrg.get("per_variable", {})
    names = list(cfg.data.variables)
    W = np.zeros((len(names), nz + nr + 1))
    for i, v in enumerate(names):
        b = {**{k: zrg[k] for k in ("zonal", "regional", "global")}, **per.get(v, {})}
        zw = np.asarray(b.get("zonal_weights") or cfg.weights.zonal_weights or np.ones(nz), float)
        rw = np.asarray(b.get("regional_weights") or cfg.weights.regional_weights or np.ones(nr), float)
        wv = float(cfg.weights.variables.get(v, 0.0))
        W[i, :nz] = wv * b["zonal"] * zw / zw.sum()
        W[i, nz:nz + nr] = wv * b["regional"] * rw / rw.sum()
        W[i, -1] = wv * b["global"]
    return names, W


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--title", default="How the 88 features are combined into one cost")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    names, W = weights(cfg)
    nv, nf = W.shape
    nz, nr = len(ZONAL), len(REGIONS)
    zrg = cfg.weights.zrg

    fig = plt.figure(figsize=(16, 9.6))
    fig.text(0.03, 0.97, args.title.replace("88", str(nv * nf)), fontsize=17, color=INK,
             weight="bold", va="top")

    # ---- equations ------------------------------------------------------------
    fig.text(0.03, 0.905, "1  Average within each block, then weight the blocks, then weight the variables",
             fontsize=12, color=ACCENT, weight="bold", va="top")
    fig.text(0.05, 0.835,
             r"$J(\theta) \;=\; \sum_{v=1}^{%d} w_v \left[\; w_Z\,\frac{1}{%d}\sum_{s=1}^{%d}"
             r" \ell_{v,s}^{\,\mathrm{zonal}} \;+\; w_R\,\frac{1}{%d}\sum_{s=1}^{%d}"
             r" \ell_{v,s}^{\,\mathrm{region}} \;+\; w_G\, \ell_{v}^{\,\mathrm{global}} \;\right]$"
             % (nv, nz, nz, nr, nr), fontsize=19, color=INK, va="center")
    fig.text(0.03, 0.76, f"2  The same thing as one weighted sum over all {nv * nf} features",
             fontsize=12, color=ACCENT, weight="bold", va="top")
    fig.text(0.05, 0.70,
             r"$J(\theta) \;=\; \sum_{v,s} \omega_{v,s}\,\ell_{v,s}(\theta), \qquad"
             r"\omega_{v,s} = \frac{w_v\, w_Z}{%d}\ \mathrm{(zonal)},\quad"
             r"\frac{w_v\, w_R}{%d}\ \mathrm{(region)},\quad w_v\, w_G\ \mathrm{(global)},"
             r"\qquad \sum_{v,s}\omega_{v,s} = %.2g$" % (nz, nr, W.sum()),
             fontsize=16, color=INK, va="center")
    lines = [r"$\ell_{v,s}$ = the cost of one feature (Student-t, in PPE-sd units).   "
             + r"$w_Z = w_R = w_G$ = " + f"{zrg['zonal']:.3g}"
             + (" unless set per variable below." if zrg.get("per_variable") else "."),
             r"$w_v$: " + ", ".join(f"{n} {cfg.weights.variables.get(n, 0):g}" for n in names) + "."]
    for v, o in zrg.get("per_variable", {}).items():
        bits = []
        if "global" in o:
            bits.append(f"zonal {o['zonal']:g}, regional {o['regional']:g}, global {o['global']:g}")
        if "zonal_weights" in o or "regional_weights" in o:
            nz0 = sum(1 for w in o.get("zonal_weights", [1] * len(ZONAL)) if w == 0)
            nr0 = sum(1 for w in o.get("regional_weights", [1] * len(REGIONS)) if w == 0)
            bits.append(f"{nz0} bands and {nr0} regions left out, the block averages over the rest")
        lines.append(f"{v}: " + "; ".join(bits) + ".")
    for k, t in enumerate(lines):
        fig.text(0.05, 0.655 - 0.024 * k, t, fontsize=10.5, color=MUTED, va="center")

    # ---- per-feature weight grid ----------------------------------------------
    ax = fig.add_axes([0.075, 0.1, 0.83, 0.42])
    x = np.arange(nf).astype(float)
    x[nz:] += 0.5
    x[-1] += 0.5
    pos = W[W > 0]
    norm = LogNorm(vmin=pos.min() / 1.5, vmax=pos.max())
    cmap = plt.get_cmap("Blues")
    for i in range(nv):
        for k in range(nf):
            w = W[i, k]
            yy = nv - 1 - i
            if w > 0:
                c = cmap(0.15 + 0.85 * norm(w))
                ax.add_patch(plt.Rectangle((x[k] - 0.46, yy - 0.44), 0.92, 0.88, color=c, lw=0))
                ax.text(x[k], yy, f"{1e3 * w:.1f}", ha="center", va="center", fontsize=8.3,
                        color="white" if norm(w) > 0.55 else INK)
            else:
                ax.add_patch(plt.Rectangle((x[k] - 0.46, yy - 0.44), 0.92, 0.88, fc="white",
                                           ec="#c4c8ce", lw=0.8, hatch="////"))
    for i, n in enumerate(names):
        yy = nv - 1 - i
        ax.text(-0.9, yy, f"{n}", ha="right", va="center", fontsize=11, color=INK, weight="bold")
        ax.text(x[-1] + 0.75, yy, f"= {W[i].sum():.2f}", ha="left", va="center", fontsize=10.5,
                color=INK)
    ax.text(x[-1] + 0.75, nv - 0.35, "row sum", ha="left", va="bottom", fontsize=9, color=MUTED)
    # block brackets and totals
    blocks = [("15 zonal bands, averaged", 0, nz), ("6 regions, averaged", nz, nz + nr), ("global", nz + nr, nf)]
    for lab, a, b in blocks:
        x0, x1 = x[a] - 0.46, x[b - 1] + 0.46
        ax.plot([x0, x0, x1, x1], [nv - 0.3, nv - 0.22, nv - 0.22, nv - 0.3], color=INK, lw=1)
        ax.text((x0 + x1) / 2, nv - 0.12, lab, ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{z}°" for z in ZONAL] + REGIONS + ["global"], fontsize=8.5,
                       rotation=0)
    for t in ax.get_xticklabels()[nz:]:
        t.set_fontsize(7.8)
    ax.tick_params(length=0)
    ax.set_yticks([])
    ax.set_xlim(-1.0, x[-1] + 0.6)
    ax.set_ylim(-0.6, nv + 0.35)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.03, 0.56, r"3  Effective weight $\omega_{v,s}$ of every feature ($\times 10^{-3}$); darker = counts more",
             fontsize=12, color=ACCENT, weight="bold", va="top")

    # one-line reading guide, computed from the weights
    i0 = int(np.argmax([cfg.weights.variables.get(n, 0) for n in names]))
    gz, gr, gg = W[i0, 0], W[i0, nz], W[i0, -1]
    note = (f"Each block gets the same total weight, so a feature's weight falls with the number of features "
            f"in its block: for {names[i0]}, one zonal band = {1e3 * gz:.1f}, one region = {1e3 * gr:.1f}, "
            f"the global mean = {1e3 * gg:.1f} ({gg / gz:.0f}x a band).")
    if (W == 0).any():
        note += "\nHatched = weight 0 (left out)."
    fig.text(0.03, 0.05, note, fontsize=10, color=INK, va="top", linespacing=1.5)
    fig.text(0.03, 0.008, f"Weights from {args.config}.", fontsize=8.5, color=MUTED, va="center")
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}; total weight {W.sum():.4f}")


if __name__ == "__main__":
    main()
