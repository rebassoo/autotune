"""
PNG of the 3-level AR covariance matrix, block by block, showing which blocks
the warm start leaves untouched and which ones the ne256 fit actually moves.

Blocks follow from the chain f0 = g0, f1 = rho1 f0 + g1, f2 = rho2 f1 + g2 as
implemented by emukit's LinearMultiFidelityKernel, with k0/k1/rho1 and the two
lower noises frozen by NFidelityGPWrapper(freeze_lower=True).

Usage:
    python scripts/make_covariance_blocks_figure.py --out docs/covariance_blocks.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
PANEL = "#f3f2ee"
FROZEN = "#dce7f2"     # unchanged by the ne256 fit
SCALED = "#eef6ef"     # fixed matrix, one varying scalar (rho2)
FITTED = "#d6f0e4"     # genuinely new structure

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Sans",
                     "text.color": TEXT})

LEV = [("ne32", 984), ("ne128", 218), ("ne256", 47)]

# formula, category
CELLS = [
    [(r"$k_0 + \sigma_0^2 I$", "frozen"),
     (r"$\rho_1 k_0$", "frozen"),
     (r"$\rho_2\rho_1 k_0$", "scaled")],
    [(r"$\rho_1 k_0$", "frozen"),
     (r"$\rho_1^2 k_0 + k_1 + \sigma_1^2 I$", "frozen"),
     (r"$\rho_2(\rho_1^2 k_0 + k_1)$", "scaled")],
    [(r"$\rho_2\rho_1 k_0$", "scaled"),
     (r"$\rho_2(\rho_1^2 k_0 + k_1)$", "scaled"),
     (r"$\rho_2^2(\rho_1^2 k_0 + k_1) + k_2 + \sigma_2^2 I$", "fitted")],
]
FACE = {"frozen": FROZEN, "scaled": SCALED, "fitted": FITTED}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/covariance_blocks.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(16.5, 11.6), facecolor=SURFACE)
    fig.text(0.5, 0.972, "The 3-level covariance matrix: what the ne256 fit changes",
             ha="center", va="top", fontsize=20, weight="bold")
    fig.text(0.5, 0.943,
             "Joint covariance over the stacked 984 + 218 + 47 = 1249 training points, "
             "with the lower levels warm-started and frozen",
             ha="center", va="top", fontsize=11.5, color=TEXT2)

    # ---- the matrix --------------------------------------------------------
    x0, y0, w, h = 0.145, 0.315, 0.262, 0.113
    for j, (lab, n) in enumerate(LEV):
        fig.text(x0 + (j + 0.5) * w, y0 + 3 * h + 0.030, lab, ha="center",
                 fontsize=14, weight="bold")
        fig.text(x0 + (j + 0.5) * w, y0 + 3 * h + 0.008, f"{n} points", ha="center",
                 fontsize=10, color=TEXT2)
    for i, (lab, n) in enumerate(LEV):
        y = y0 + (2 - i) * h
        fig.text(x0 - 0.012, y + h / 2 + 0.008, lab, ha="right", va="center",
                 fontsize=14, weight="bold")
        fig.text(x0 - 0.012, y + h / 2 - 0.016, f"{n} points", ha="right", va="center",
                 fontsize=10, color=TEXT2)

    for i in range(3):
        for j in range(3):
            eq, cat = CELLS[i][j]
            x, y = x0 + j * w, y0 + (2 - i) * h
            fig.add_artist(FancyBboxPatch((x + 0.004, y + 0.004), w - 0.008, h - 0.008,
                                          boxstyle="round,pad=0,rounding_size=0.006",
                                          transform=fig.transFigure, facecolor=FACE[cat],
                                          edgecolor=GRID, lw=1.2, zorder=1))
            fs = 13.5 if cat != "fitted" else 12.0
            fig.text(x + w / 2, y + h / 2 + 0.012, eq, ha="center", va="center",
                     fontsize=fs, zorder=2)
            tag = {"frozen": "unchanged",
                   "scaled": "fixed matrix × scalar $\\rho_2$",
                   "fitted": "new structure"}[cat]
            fig.text(x + w / 2, y + h / 2 - 0.030, tag, ha="center", va="center",
                     fontsize=9.5, color=TEXT2, style="italic", zorder=2)

    # ---- legend ------------------------------------------------------------
    ly = 0.262
    for k, (cat, label) in enumerate([
        ("frozen", "unchanged — every hyperparameter in it is frozen"),
        ("scaled", "changes only through the single scalar $\\rho_2$"),
        ("fitted", "genuinely new — contains $k_2$ and $\\sigma_2^2$"),
    ]):
        x = 0.075 + k * 0.30
        fig.add_artist(FancyBboxPatch((x, ly), 0.022, 0.020,
                                      boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, facecolor=FACE[cat],
                                      edgecolor=GRID, lw=1.0))
        fig.text(x + 0.030, ly + 0.010, label, va="center", fontsize=10.5, color=TEXT2)

    # ---- the chain ---------------------------------------------------------
    fig.add_artist(FancyBboxPatch((0.03, 0.715), 0.94, 0.205,
                                  boxstyle="round,pad=0,rounding_size=0.007",
                                  transform=fig.transFigure, facecolor=PANEL,
                                  edgecolor=GRID, lw=1.0, zorder=0))
    fig.text(0.045, 0.908, "Where the blocks come from", va="top", fontsize=13,
             weight="bold")
    for i, eq in enumerate([
        r"$f_{\mathrm{ne32}} = g_0,\qquad g_0\sim\mathcal{GP}(0,k_0)$",
        r"$f_{\mathrm{ne128}} = \rho_1 f_{\mathrm{ne32}} + g_1,\qquad g_1\sim\mathcal{GP}(0,k_1)$",
        r"$f_{\mathrm{ne256}} = \rho_2 f_{\mathrm{ne128}} + g_2,\qquad g_2\sim\mathcal{GP}(0,k_2)$",
    ]):
        fig.text(0.055, 0.874 - i * 0.033, eq, va="center", fontsize=14)
    fig.text(0.53, 0.874, "FROZEN by the warm start (43 params):", va="center",
             fontsize=11.5, weight="bold", color="#1c5cab")
    fig.text(0.55, 0.848, r"$k_0$ (20),  $k_1$ (20),  $\rho_1$,  $\sigma_0^2$,  $\sigma_1^2$",
             va="center", fontsize=12.5)
    fig.text(0.53, 0.812, "FITTED from the 47 ne256 members (4 params):", va="center",
             fontsize=11.5, weight="bold", color="#0f7a55")
    fig.text(0.55, 0.786, r"$k_2$ (2),  $\rho_2$,  $\sigma_2^2$",
             va="center", fontsize=12.5)
    fig.text(0.045, 0.742,
             "Substituting the chain gives every block below. Only $\\rho_2$, $k_2$ and "
             "$\\sigma_2^2$ are free, so the entire top-left 1202 × 1202 corner "
             "(ne32 and ne128 with each other) is numerically identical to the trained "
             "2-level model.",
             va="center", fontsize=10.5, color=TEXT2)

    # ---- the catch ---------------------------------------------------------
    fig.add_artist(FancyBboxPatch((0.03, 0.020), 0.94, 0.215,
                                  boxstyle="round,pad=0,rounding_size=0.007",
                                  transform=fig.transFigure, facecolor="white",
                                  edgecolor=GRID, lw=1.0, zorder=0))
    fig.text(0.045, 0.222, "Frozen is not the same as free", va="top",
             fontsize=13, weight="bold")
    for i, line in enumerate([
        "GPy assembles and Cholesky-factorises the WHOLE 1249 × 1249 matrix at every likelihood evaluation, frozen blocks included.",
        "So the unchanged corner is recomputed from scratch each iteration — which is why the warm+frozen fit still took 27 minutes",
        "rather than seconds, and why a cold fit costs ~7.5 h per fold.",
        "",
        "Recursive (sequential) co-kriging is what actually exploits the structure: condition on the lower levels once, then fit the top",
        "level on the residuals as a 47 × 47 problem. For a nested design that is exact, not an approximation — the posterior is identical.",
        "",
        "Nesting here: ne256 ⊆ ne32 holds for all 47 members, but 6 of the 47 have no ne128 partner and 6 of the 218 ne128 members have",
        "no ne32 partner, so those steps would need the non-nested members dropped (or a small approximation accepted).",
    ]):
        fig.text(0.045, 0.192 - i * 0.021, line, va="top", fontsize=10.2,
                 color=TEXT2 if line else TEXT, linespacing=1.4)

    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
