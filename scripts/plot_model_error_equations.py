"""
The model-error calibration equations on one page (no plots): observation
model, per-feature variance, the Student-t cost, the per-block sigma_delta
estimate, and why a larger variance means a smaller weight.

    python scripts/plot_model_error_equations.py --out docs/model_error_equations.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, ACCENT = "#1a1a1a", "#6b6f76", "#1b4f9c"

STEPS = [
    ("1  Observations vs. SCREAM at the true parameters (Kennedy & O'Hagan)",
     r"$z_s \;=\; \eta(s,\theta^\ast) \;+\; \delta_s \;+\; \varepsilon_s$",
     r"$z_s$: observed feature $s$ (zonal band, region or global);  $\eta$: SCREAM, emulated by the GP;"
     r"  $\delta_s$: model error;  $\varepsilon_s$: observation error"),
    ("2  Total variance at each feature",
     r"$V_s(\theta) \;=\; \sigma^2_{\mathrm{GP},s}(\theta) \;+\; \sigma^2_{\mathrm{obs},s}"
     r" \;+\; \sigma^2_{\delta,\,b(s)}$",
     r"surrogate (per feature, depends on $\theta$)  +  specified obs error  +  model error, one value per"
     r" variable $\times$ block $b$ = zonal / regional / global"),
    ("3  Per-feature cost (Student-t negative log-likelihood, $\\nu = 4$)",
     r"$\ell_s(\theta) \;=\; \frac{1}{2}\log V_s \;+\; \frac{\nu+1}{2}\,"
     r"\log\!\left(1 + \frac{r_s^2}{\nu\,V_s}\right), \qquad r_s = z_s - \mu_{\mathrm{GP}}(s,\theta)$",
     r"$\mu_{\mathrm{GP}}$: surrogate mean;  all quantities in normalized units (PPE std per feature)"),
    ("4  Total cost: weighted over variables and blocks",
     r"$C(\theta) \;=\; \sum_{v} w_v \sum_{b} w_b \,\frac{1}{n_b}\sum_{s \in b} \ell_{s,v}(\theta)$",
     r"$w_v$ = 0.3 / 0.1 / 0.3 / 0.3 (PCP / OSR / OLR / TLWP);  $w_b$: zonal / regional / global weights"),
    ("5  Calibration: alternate until $\\theta$ and $\\sigma_\\delta$ stop moving",
     r"$\hat\theta \;=\; \arg\min_\theta\, C(\theta)\,,\qquad"
     r"\hat\sigma^2_{\delta,b} \;=\; \arg\min_{x \,\geq\, 0}\; \frac{1}{n_b}\sum_{s \in b}"
     r" \ell_{s}\!\left(r_{s}^2,\; \sigma^2_{\mathrm{GP},s} + \sigma^2_{\mathrm{obs},s} + x\right)$",
     r"residuals $r$ at $\hat\theta$; whatever obs error is not specified in $\sigma^2_{\mathrm{obs}}$"
     r" ends up in $\hat\sigma^2_\delta$ (residuals can't tell the two apart)"),
    ("6  Why a larger variance means a smaller weight",
     r"$\frac{\partial \ell_s}{\partial \theta} \;=\; -\,\frac{(\nu+1)\, r_s}{\nu V_s + r_s^2}\,"
     r"\frac{\partial \mu_{\mathrm{GP}}}{\partial \theta} \;\propto\; -\,\frac{r_s}{V_s}\,"
     r"\frac{\partial \mu_{\mathrm{GP}}}{\partial \theta} \quad (V_s \mathrm{\ held\ fixed};\ r_s^2 \ll \nu V_s)$",
     r"the pull of a feature on $\theta$ scales as $1/V_s$: larger $V$ $\Rightarrow$ that variable/block counts"
     r" less, the optimum moves toward the others, and the cost valley flattens (wider uncertainty)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig = plt.figure(figsize=(13, 10.5))
    fig.text(0.03, 0.965, "Model-error calibration: the equations", fontsize=16, color=INK,
             weight="bold", va="top")
    y, dy = 0.895, 0.148
    for title, eq, note in STEPS:
        fig.text(0.03, y, title, fontsize=11.5, color=ACCENT, weight="bold", va="top")
        fig.text(0.06, y - 0.045, eq, fontsize=17, color=INK, va="center")
        fig.text(0.06, y - 0.098, note, fontsize=9.5, color=MUTED, va="center", wrap=True)
        y -= dy
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
