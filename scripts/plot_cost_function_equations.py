"""
The optimization cost as a PNG: total cost, the per-feature Student-t term,
the normalization, the weights and settings of the n=85 run
(configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml), and a legend of
every symbol. The formula follows autotune_gp.cost.zrg_cost_function_student_t
and the cost_fn in scripts/run_two_stage.py.

    python scripts/plot_cost_function_equations.py --out docs/cost_function_n85.png
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, ACCENT, RULE = "#1a1a1a", "#6b6f76", "#1b4f9c", "#d0d4da"

NU = 4.0
C_NU = -math.lgamma((NU + 1) / 2) + math.lgamma(NU / 2) + 0.5 * math.log(NU * math.pi)

EQS = [
    ("Total cost, minimized over the 19 parameters",
     r"$J(\theta) \;=\; \sum_{v} w_v \left[\; w_Z\,\frac{1}{N_Z}\sum_{s\in Z} \ell_{v,s}(\theta)"
     r"\;+\; w_R\,\frac{1}{N_R}\sum_{s\in R} \ell_{v,s}(\theta)"
     r"\;+\; w_G\,\ell_{v,G}(\theta) \;\right]$"),
    ("Cost of one feature: Student-t negative log-likelihood",
     r"$\ell_{v,s}(\theta) \;=\; \frac{1}{2}\log V_{v,s} \;+\; \frac{\nu+1}{2}\,"
     r"\log\!\left(1 + \frac{r_{v,s}^2}{\nu\, V_{v,s}}\right) \;+\; c_\nu$"),
    ("Residual and variance, both in normalized units",
     r"$r_{v,s} \;=\; z_{v,s} - \mu_{v,s}(\theta), \qquad "
     r"V_{v,s} \;=\; \max\left(\sigma^2_{v,s}(\theta),\; 10^{-6}\right), \qquad "
     r"z_{v,s} \;=\; \frac{y^{\mathrm{obs}}_{v,s} - m_{v,s}}{d_{v,s}}$"),
]

SYMBOLS = [
    (r"$\theta$", "the 19 tuning parameters, scaled to 0-1 over the range the PPE sampled"),
    (r"$v$", "an observable: PCP, OSR, OLR, TLWP"),
    (r"$s$", "a feature: one of $N_Z$ = 15 zonal bands ($Z$), $N_R$ = 6 regions ($R$), or the global mean ($G$)"),
    (r"$\mu_{v,s}(\theta)$", "the 3-level surrogate's prediction of ne256 SCREAM at $\\theta$ (normalized)"),
    (r"$\sigma^2_{v,s}(\theta)$", "the surrogate's own predictive variance at $\\theta$ (no model-error term in this run)"),
    (r"$y^{\mathrm{obs}}_{v,s}$", "the observation (IMERG precip, CERES OSR/OLR, MAC-LWP), Aug 2019 - Jul 2020"),
    (r"$m_{v,s},\ d_{v,s}$", "mean and standard deviation of the ne128 PPE at that feature; 1 unit = 1 PPE sd"),
    (r"$\nu$", f"Student-t degrees of freedom = {NU:g}; Gaussian as $\\nu\\to\\infty$"),
    (r"$c_\nu$", f"normalizing constant $-\\log\\Gamma(\\frac{{\\nu+1}}{{2}}) + \\log\\Gamma(\\frac{{\\nu}}{{2}})"
     f" + \\frac{{1}}{{2}}\\log(\\nu\\pi)$ = {C_NU:.3f}; does not move the optimum"),
    (r"$w_v$", "variable weights: PCP 0.3, OSR 0.1, OLR 0.3, TLWP 0.3"),
    (r"$w_Z, w_R, w_G$", "block weights, 1/3 each; zonal and regional are averaged, so the global feature"
     " weighs as much as 15 bands"),
]

NOTES = [
    "Optimizer: L-BFGS-B from 40 starts, each parameter bounded to the range the PPE sampled.",
    r"If $\lambda_{\mathrm{low}} > \lambda_{\mathrm{high}}$ the cost is replaced by $100 + 100\,(\lambda_{\mathrm{low}}"
    r" - \lambda_{\mathrm{high}})$ to keep the pair ordered.",
    r"Off in this run: variance penalty $\kappa\,V$ ($\kappa$ = 0), training-distance penalty ($\lambda$ = 0),"
    r" model-error variance $\sigma^2_\delta$.",
    r"Why Student-t: for $|r| \gg \sqrt{\nu V}$ the fit term grows like $\log r$, not $r^2$, so one feature that"
    " cannot be fitted cannot dominate.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    H = 14.0
    f = 11.5 / H                              # spacing below was laid out for an 11.5-in page
    fig = plt.figure(figsize=(13, H))
    fig.text(0.04, 0.972, "The optimization cost", fontsize=17, color=INK, weight="bold", va="top")
    fig.text(0.04, 1 - 0.062 * f, "3-level multi-fidelity surrogate (ne32 → ne128 → ne256, n = 85), annual means",
             fontsize=11, color=MUTED, va="top")
    y = 1 - 0.115 * f
    for k, (title, eq) in enumerate(EQS, 1):
        fig.text(0.04, y, f"{k}  {title}", fontsize=12, color=ACCENT, weight="bold", va="top")
        fig.text(0.07, y - 0.058 * f, eq, fontsize=19 if k < 3 else 16.5, color=INK, va="center")
        y -= 0.135 * f

    y += 0.02 * f
    fig.add_artist(plt.Line2D([0.04, 0.96], [y, y], color=RULE, lw=1))
    y -= 0.03 * f
    fig.text(0.04, y, "Symbols", fontsize=12, color=ACCENT, weight="bold", va="top")
    y -= 0.045 * f
    for sym, desc in SYMBOLS:
        fig.text(0.07, y, sym, fontsize=12.5, color=INK, va="center")
        fig.text(0.21, y, desc, fontsize=10.5, color=INK, va="center")
        y -= 0.034 * f

    y -= 0.008 * f
    fig.add_artist(plt.Line2D([0.04, 0.96], [y, y], color=RULE, lw=1))
    y -= 0.03 * f
    fig.text(0.04, y, "Settings", fontsize=12, color=ACCENT, weight="bold", va="top")
    y -= 0.045 * f
    for n in NOTES:
        fig.text(0.07, y, n, fontsize=10.5, color=INK, va="center")
        y -= 0.034 * f

    fig.text(0.04, 0.012, "Source: autotune_gp.cost.zrg_cost_function_student_t; "
             "configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml.", fontsize=8.5, color=MUTED)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
