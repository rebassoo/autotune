"""
One-page explainer PNG: how the 3-level ne256 GP warm-starts from the trained
ne32+ne128 model, and which hyperparameters are reused rather than refitted.

Everything here is transcribed from autotune_gp.gp_multifidelity.NFidelityGPWrapper
(_apply_warm_start / _freeze_lower_levels / _build_kernels) and the run log of
the 3-level training job, so the counts and timings are the real ones.

Usage:
    python scripts/make_warm_start_figure.py --out docs/warm_start_explained.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
PANEL = "#f3f2ee"
BLUE = "#2a78d6"      # ne32
ORANGE = "#eb6834"    # ne128
AQUA = "#1baf7a"      # ne256
FROZEN_BG = "#dce7f2"
FITTED_BG = "#d6f0e4"

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
})

# block, n params, source, frozen?
BLOCKS = [
    ("ne32 base kernel  $k_0$ — ARD RBF", "19 lengthscales + variance", 20,
     "copied from the trained 2-level GP", True),
    ("ne128 discrepancy kernel  $k_1$ — ARD RBF", "19 lengthscales + variance", 20,
     "copied from the trained 2-level GP", True),
    ("$\\rho_1$  (ne32 $\\rightarrow$ ne128)", "scaling factor", 1,
     "copied from the trained 2-level GP", True),
    ("noise, ne32 and ne128 levels", "one variance each", 2,
     "copied from the trained 2-level GP", True),
    ("ne256 discrepancy kernel  $k_2$ — isotropic RBF", "1 shared lengthscale + variance", 2,
     "FITTED from the 47 ne256 members", False),
    ("$\\rho_2$  (ne128 $\\rightarrow$ ne256)", "scaling factor", 1,
     "FITTED from the 47 ne256 members", False),
    ("noise, ne256 level", "one variance", 1,
     "FITTED from the 47 ne256 members", False),
]


def panel(fig, x, y, w, h, face=PANEL, edge=GRID):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.007",
                                  transform=fig.transFigure, facecolor=face,
                                  edgecolor=edge, lw=1.0, zorder=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/warm_start_explained.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(17, 13.6), facecolor=SURFACE)

    fig.text(0.5, 0.978, "Warm-starting the ne256 level from the ne32 + ne128 model",
             ha="center", va="top", fontsize=21, weight="bold")
    fig.text(0.5, 0.953,
             "autotune_gp.gp_multifidelity.NFidelityGPWrapper  "
             "(warm_start=<trained 2-level GP>, freeze_lower=True)",
             ha="center", va="top", fontsize=11.5, color=TEXT2)

    # ---- A: the model ------------------------------------------------------
    panel(fig, 0.03, 0.745, 0.94, 0.185)
    fig.text(0.045, 0.918, "The model:  each level is the one below it, rescaled, "
             "plus its own discrepancy GP", va="top", fontsize=13, weight="bold")
    eqs = [
        (BLUE,   r"$f_{\mathrm{ne32}}(x) = g_0(x)$",
         r"$g_0 \sim \mathcal{GP}(0,\ k_0)$", "984 members"),
        (ORANGE, r"$f_{\mathrm{ne128}}(x) = \rho_1\, f_{\mathrm{ne32}}(x) + g_1(x)$",
         r"$g_1 \sim \mathcal{GP}(0,\ k_1)$", "218 members"),
        (AQUA,   r"$f_{\mathrm{ne256}}(x) = \rho_2\, f_{\mathrm{ne128}}(x) + g_2(x)$",
         r"$g_2 \sim \mathcal{GP}(0,\ k_2)$", "47 members"),
    ]
    for i, (c, eq, kern, n) in enumerate(eqs):
        y = 0.876 - i * 0.042
        fig.add_artist(plt.Rectangle((0.045, y - 0.012), 0.005, 0.028,
                                     transform=fig.transFigure, facecolor=c, edgecolor="none"))
        fig.text(0.060, y, eq, va="center", fontsize=15.5)
        fig.text(0.44, y, kern, va="center", fontsize=12, color=TEXT2)
        fig.text(0.60, y, n, va="center", fontsize=11, color=TEXT2)
    fig.text(0.045, 0.762,
             "One such model per output feature — 88 in total (22 ZRG locations × 4 variables). "
             "Fitted values here: ρ₁ ≈ 0.92, ρ₂ ≈ 0.88 (means over the 88 models), "
             "i.e. the resolutions track each other closely.",
             va="center", fontsize=10.5, color=TEXT2)

    # ---- B: where each hyperparameter comes from ---------------------------
    panel(fig, 0.03, 0.315, 0.94, 0.405)
    fig.text(0.045, 0.705, "Where each hyperparameter comes from",
             va="top", fontsize=13, weight="bold")
    fig.text(0.045, 0.679,
             "47 free hyperparameters per model. The warm start supplies 43 of them and "
             "freezes them; only 4 are fitted from the ne256 data.",
             va="top", fontsize=10.5, color=TEXT2)

    hy = 0.645
    fig.text(0.052, hy, "block", fontsize=10.5, weight="bold", color=TEXT2)
    fig.text(0.415, hy, "what it is", fontsize=10.5, weight="bold", color=TEXT2)
    fig.text(0.625, hy, "#", fontsize=10.5, weight="bold", color=TEXT2)
    fig.text(0.665, hy, "source", fontsize=10.5, weight="bold", color=TEXT2)

    for i, (name, what, n, src, frozen) in enumerate(BLOCKS):
        y = hy - 0.030 - i * 0.042
        fig.add_artist(FancyBboxPatch((0.045, y - 0.018), 0.912, 0.035,
                                      boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure,
                                      facecolor=FROZEN_BG if frozen else FITTED_BG,
                                      edgecolor="none", zorder=0))
        fig.text(0.052, y, name, va="center", fontsize=11.5)
        fig.text(0.415, y, what, va="center", fontsize=10, color=TEXT2)
        fig.text(0.625, y, str(n), va="center", fontsize=11.5, weight="bold")
        fig.text(0.665, y, src, va="center", fontsize=10.5,
                 color=TEXT if not frozen else TEXT2,
                 weight="bold" if not frozen else "normal")

    fig.text(0.052, 0.328, "43 frozen", fontsize=12, weight="bold", color="#1c5cab")
    fig.text(0.145, 0.328, "(blue rows — reused as-is, never re-estimated)",
             fontsize=10.5, color=TEXT2)
    fig.text(0.52, 0.328, "4 fitted", fontsize=12, weight="bold", color="#0f7a55")
    fig.text(0.60, 0.328, "(green rows — everything the ne256 data actually determines)",
             fontsize=10.5, color=TEXT2)

    # ---- C: what it does and does not change -------------------------------
    panel(fig, 0.03, 0.135, 0.46, 0.165, face="white")
    fig.text(0.045, 0.285, "What the warm start does NOT change",
             va="top", fontsize=12.5, weight="bold")
    for i, line in enumerate([
        "All the data is still used: the likelihood is computed over",
        "984 + 218 + 47 = 1249 points at every iteration.",
        "Only the hyperparameters are held fixed, not the training set.",
        "",
        "Predictions still flow up the whole chain, so ne256 predictions",
        "depend on the ne32 and ne128 members as much as ever.",
    ]):
        fig.text(0.045, 0.252 - i * 0.021, line, va="top", fontsize=10.5,
                 color=TEXT2 if line else TEXT)

    panel(fig, 0.51, 0.135, 0.46, 0.165, face="white")
    fig.text(0.525, 0.285, "Why freeze them", va="top", fontsize=12.5, weight="bold")
    for i, line in enumerate([
        "47 ne256 members are ~3% of the training data and carry no real",
        "information about the ne32↔ne128 relationship. A joint fit would",
        "still let them move those 40 well-determined hyperparameters,",
        "possibly into a different local optimum.",
        "",
        "It is also far faster: 27 min to train all 88 models warm+frozen,",
        "against ~7.5 h per fold when fitted cold (strict-holdout job).",
    ]):
        fig.text(0.525, 0.252 - i * 0.021, line, va="top", fontsize=10.5,
                 color=TEXT2 if line else TEXT)

    # ---- D: caveats --------------------------------------------------------
    panel(fig, 0.03, 0.022, 0.94, 0.095)
    fig.text(0.045, 0.100, "Two caveats", va="top", fontsize=12.5, weight="bold")
    fig.text(0.045, 0.074,
             "1.  The predictive variance treats the 43 frozen hyperparameters as exactly known, "
             "so it ignores their uncertainty and is mildly optimistic. Since the cost functions "
             "weight each location by that variance,\n     the parameter error bars inherit this. "
             "Run with --no-freeze-lower once to check how much it matters.",
             va="top", fontsize=10.5, color=TEXT2, linespacing=1.45)
    fig.text(0.045, 0.040,
             "2.  The top discrepancy kernel k₂ uses ONE shared lengthscale, not one per parameter: "
             "19 ARD lengthscales cannot be identified from 47 points. NFidelityGPWrapper switches to "
             "per-parameter lengthscales\n     automatically once the top level reaches 3 × 19 = 57 members.",
             va="top", fontsize=10.5, color=TEXT2, linespacing=1.45)

    # arrow from the 2-level source into the frozen block
    fig.add_artist(FancyArrowPatch((0.905, 0.745), (0.905, 0.612),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=18, color="#1c5cab", lw=2.0, zorder=5))
    fig.text(0.893, 0.690, "trained 2-level GP\n(mf_gp_trained.pkl)\nsupplies the blue rows",
             fontsize=10, color="#1c5cab", va="center", ha="right", weight="bold")

    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
