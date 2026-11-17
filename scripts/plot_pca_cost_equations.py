"""
The PCA-likelihood cost as a PNG, in the style of plot_cost_function_equations.py:
the cost, the projection onto the PPE's leading EOFs, the error covariance, the
calibration check, a legend of every symbol and the settings of the n=85 basis
(/pscratch/sd/r/rebassoo/autotune/pca_basis_k20_n85.pkl). The formula follows
autotune_gp.pca_likelihood (build_basis, pca_cost).

    python scripts/plot_pca_cost_equations.py --out docs/pca_cost_function_n85.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, ACCENT, RULE = "#1a1a1a", "#6b6f76", "#1b4f9c", "#d0d4da"

EQS = [
    ("Cost, minimized over the 19 parameters (the MCMC posterior is proportional to exp(−J))",
     r"$J(\theta) \;=\; \frac{1}{2}\, \mathbf{c}(\theta)^{\mathsf{T}}\, \Sigma^{-1}\, \mathbf{c}(\theta)"
     r"\;=\; \frac{1}{2} \sum_{i=1}^{K}\;\sum_{j=1}^{K}\, c_i(\theta)\, \left(\Sigma^{-1}\right)_{ij} c_j(\theta)$"),
    ("Residual of all 88 features, projected onto the K = 20 leading PCA modes of the PPE",
     r"$\mathbf{c}(\theta) \;=\; V^{\mathsf{T}}\, \mathbf{r}(\theta), \qquad "
     r"\mathbf{r}(\theta) \;=\; \boldsymbol{\mu}(\theta) - \mathbf{z}$"),
    ("Error covariance: second moment of the surrogate's out-of-sample errors, in mode space",
     r"$\Sigma \;=\; \mathrm{LW}\!\left[\; \frac{1}{M}\sum_{m=1}^{M} \mathbf{e}_m \mathbf{e}_m^{\mathsf{T}} \;\right],"
     r"\qquad \mathbf{e}_m \;=\; V^{\mathsf{T}}\left(\boldsymbol{\mu}_{-m}(\theta_m) - \mathbf{y}_m\right)$"),
    ("Calibration check: at the true parameters, with only surrogate error",
     r"$2\,J(\theta_{\mathrm{true}}) \;\sim\; \chi^2_K \quad\Rightarrow\quad "
     r"\mathrm{E}\left[J(\theta_{\mathrm{true}})\right] \;=\; K/2 \;=\; 10$"),
]

SYMBOLS = [
    (r"$\theta$", "the 19 tuning parameters, scaled to 0-1 over the range the PPE sampled"),
    (r"$\mathbf{z}$", "the target, 88 numbers: 4 observables (PCP, OSR, OLR, TLWP) × 22 features "
     "(15 zonal bands, 6 regions, global mean)"),
    (r"$\boldsymbol{\mu}(\theta)$", "the 3-level surrogate's prediction of ne256 SCREAM at $\\theta$, same 88 features"),
    ("", "everything in normalized units: 1 unit = 1 standard deviation of the ne128 PPE at that feature"),
    (r"$V$", "88 × 20: the leading 20 PCA modes (EOFs) of the 85 ne256 PPE runs; 99.6% of the PPE variance"),
    (r"$\mathbf{c}(\theta)$", "the residual's 20 mode amplitudes"),
    (r"$\Sigma$", "20 × 20 error covariance; constant (does not depend on $\\theta$), off-diagonals kept"),
    (r"$\mathbf{e}_m$", "error of the 5-fold holdout prediction of ne256 run $m$ (surrogate trained without it),"
     " $M$ = 85"),
    (r"$\mathrm{LW}$", "Ledoit-Wolf shrinkage of the covariance (85 errors for 210 covariance entries)"),
    (r"$\chi^2_K$", "chi-squared with K = 20 degrees of freedom: 5-95% range 10.9-31.4 for $2J$"),
]

NOTES = [
    "Each mode is scored against how well the surrogate can predict it: modes 1-3 (cloud, OLR/TLWP, precip) are"
    " predicted to ~3% of their PPE variance,",
    "     modes beyond ~11 only to ≥100%, so they count much less.",
    "No log-variance term: $\\Sigma$ is fixed, so the cost cannot be lowered by moving to where the surrogate is"
    " uncertain.",
    "Features outside the 20 modes (0.4% of PPE variance) do not enter the cost.",
    "Not included: observation error and the model's structural error against observations. Against real"
    " observations",
    "     the best fit has $2J \\approx 113$ instead of ~20; a discrepancy term would add $V^{\\mathsf{T}} C_\\delta V$"
    " to $\\Sigma$.",
    "Posterior: emcee, 64 walkers, uniform prior over the box the PPE sampled.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    H = 16.8
    f = 11.5 / H
    fig = plt.figure(figsize=(13.5, H))
    fig.text(0.04, 0.975, "The PCA-likelihood cost", fontsize=17, color=INK, weight="bold", va="top")
    fig.text(0.04, 1 - 0.058 * f, "Gaussian likelihood on the leading PCA modes, with the error covariance "
             "measured from surrogate holdout errors (n = 85)", fontsize=11, color=MUTED, va="top")
    y = 1 - 0.11 * f
    for k, (title, eq) in enumerate(EQS, 1):
        fig.text(0.04, y, f"{k}  {title}", fontsize=12, color=ACCENT, weight="bold", va="top")
        fig.text(0.07, y - 0.06 * f, eq, fontsize=19 if k in (1, 2) else 17, color=INK, va="center")
        y -= 0.135 * f

    y += 0.02 * f
    fig.add_artist(plt.Line2D([0.04, 0.96], [y, y], color=RULE, lw=1))
    y -= 0.03 * f
    fig.text(0.04, y, "Symbols", fontsize=12, color=ACCENT, weight="bold", va="top")
    y -= 0.045 * f
    for sym, desc in SYMBOLS:
        fig.text(0.07, y, sym, fontsize=13, color=INK, va="center")
        fig.text(0.16, y, desc, fontsize=10.5, color=INK if sym else MUTED, va="center")
        y -= 0.034 * f

    y -= 0.008 * f
    fig.add_artist(plt.Line2D([0.04, 0.96], [y, y], color=RULE, lw=1))
    y -= 0.03 * f
    fig.text(0.04, y, "Notes", fontsize=12, color=ACCENT, weight="bold", va="top")
    y -= 0.045 * f
    for n in NOTES:
        fig.text(0.07, y, n, fontsize=10.5, color=INK, va="center")
        y -= 0.032 * f

    fig.text(0.04, 0.012, "Source: autotune_gp.pca_likelihood (build_basis, pca_cost); "
             "scripts/build_pca_likelihood.py; pca_basis_k20_n85.pkl.", fontsize=8.5, color=MUTED)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
