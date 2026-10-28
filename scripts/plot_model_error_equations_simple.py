"""
Two-equation version of plot_model_error_equations.py, with a Gaussian
likelihood for readability (the optimizer uses Student-t, dof=4; the 1/V
scaling is the same): the per-feature cost and its gradient, plus a legend
of every symbol.

    python scripts/plot_model_error_equations_simple.py --out docs/model_error_equations_simple.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, ACCENT = "#1a1a1a", "#6b6f76", "#1b4f9c"

EQS = [
    ("Cost of one feature (Gaussian negative log-likelihood)",
     r"$\ell_s(\theta) \;=\; \frac{r_s^2}{2\,V_s} \;+\; \frac{1}{2}\log\left(2\pi V_s\right),"
     r"\qquad r_s \;=\; z_s - \mu_{\mathrm{GP}}(s,\theta)$"),
    ("How hard that feature pulls on the parameters",
     r"$\frac{\partial \ell_s}{\partial \theta} \;=\; -\,\frac{r_s}{V_s}\,"
     r"\frac{\partial \mu_{\mathrm{GP}}(s,\theta)}{\partial \theta}"
     r"\qquad (V_s \mathrm{\ held\ fixed})$"),
]

SYMBOLS = [
    (r"$s$", "one feature: a zonal band, a region or the global mean of one variable"),
    (r"$\theta$", "the tuning parameters"),
    (r"$\ell_s$", "the cost of feature $s$; the total cost is a weighted sum of $\\ell_s$ over features and variables"),
    (r"$z_s$", "the observed value"),
    (r"$\mu_{\mathrm{GP}}(s,\theta)$", "the surrogate's prediction of SCREAM at parameters $\\theta$"),
    (r"$r_s$", "the residual, observation minus prediction"),
    (r"$\partial\mu_{\mathrm{GP}}/\partial\theta$", "how sensitive the prediction is to the parameters"),
    (r"$V_s$", "the total variance = $\\sigma^2_{\\mathrm{GP},s} + \\sigma^2_{\\mathrm{obs},s} + \\sigma^2_{\\delta,s}$"),
    (r"$\sigma^2_{\mathrm{GP},s}$", "surrogate variance: the GP's own uncertainty (depends on $\\theta$)"),
    (r"$\sigma^2_{\mathrm{obs},s}$", "observation-error variance (specified in advance; only the global terms in E2)"),
    (r"$\sigma^2_{\delta,s}$", "model-error variance from the calibration, one value per variable $\\times$ block"
     " (zonal / regional / global);"),
    ("", "it also absorbs any observation error not given in $\\sigma^2_{\\mathrm{obs}}$"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig = plt.figure(figsize=(12, 10))
    fig.text(0.04, 0.965, "Why a larger variance means a feature counts less", fontsize=16,
             color=INK, weight="bold", va="top")
    y = 0.89
    for k, (title, eq) in enumerate(EQS, 1):
        fig.text(0.04, y, f"{k}  {title}", fontsize=12, color=ACCENT, weight="bold", va="top")
        fig.text(0.07, y - 0.06, eq, fontsize=20, color=INK, va="center")
        y -= 0.15
    fig.text(0.07, y + 0.035,
             r"The pull is the residual divided by $V_s$: a larger $V_s$ makes that feature (variable/block) "
             "count less,\nso the optimum moves toward what the other variables want and the parameters are "
             "less tightly constrained.", fontsize=11, color=INK, va="top", linespacing=1.6)

    y -= 0.08
    fig.text(0.04, y, "Symbols", fontsize=12, color=ACCENT, weight="bold", va="top")
    y -= 0.05
    for sym, desc in SYMBOLS:
        fig.text(0.07, y, sym, fontsize=12, color=INK, va="center")
        fig.text(0.20, y, desc, fontsize=10.5, color=INK, va="center")
        y -= 0.034
    fig.text(0.04, 0.012, "Gaussian shown for clarity; the optimizer uses a Student-t likelihood "
             "(dof = 4), which has the same 1/V scaling for small residuals.", fontsize=8.5, color=MUTED)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
