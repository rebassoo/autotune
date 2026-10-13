"""
One-page explainer PNG for the three noise models / likelihoods used by the
optimizer: Gaussian (gaussian_gp_variance), Laplace (laplace_gp_variance) and
Student-t (student_t).

Every equation is transcribed from autotune_gp/cost.py
(zrg_cost_function_gaussian_nll / _laplace / _student_t), so the figure shows
what the code actually minimises, including the variance floor, the
Student-t normalising constant, and the weighted block-mean aggregation.

Usage:
    python scripts/make_noise_model_figure.py --out docs/noise_models_likelihoods.png
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Reference-palette tokens (dataviz skill, light mode). Categorical slots are
# assigned in fixed order; text always uses text tokens, never series colour.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
PANEL = "#f3f2ee"
SERIES = {"Gaussian": "#2a78d6", "Laplace": "#eb6834", "Student-t": "#1baf7a"}
STYLE = {"Gaussian": "-", "Laplace": "--", "Student-t": "-."}

NU = 4.0
C_NU = -math.lgamma((NU + 1) / 2) + math.lgamma(NU / 2) + 0.5 * math.log(NU * math.pi)

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

MODELS = [
    {
        "name": "Gaussian",
        "config": "likelihood: gaussian_gp_variance",
        "noise": r"$\varepsilon_i \sim \mathcal{N}(0,\ v_i)$",
        "pdf": r"$p(y_i\mid\theta)=\dfrac{1}{\sqrt{2\pi v_i}}\,"
               r"\exp\!\left(-\dfrac{r_i^{2}}{2v_i}\right)$",
        "nll": r"$\ell_i=\dfrac{r_i^{2}}{2v_i}+\dfrac{1}{2}\ln(2\pi v_i)$",
        "props": [
            r"Var$(\varepsilon_i) = v_i$",
            r"penalty grows as $r^2$  (quadratic)",
            "light tails: one large residual can\ndominate the total cost",
        ],
    },
    {
        "name": "Laplace",
        "config": "likelihood: laplace_gp_variance",
        "noise": r"$\varepsilon_i \sim \mathrm{Laplace}(0,\ b_i),\quad b_i=\sqrt{v_i/2}$",
        "pdf": r"$p(y_i\mid\theta)=\dfrac{1}{2b_i}\,"
               r"\exp\!\left(-\dfrac{|r_i|}{b_i}\right)$",
        "nll": r"$\ell_i=\dfrac{\sqrt{2}\,|r_i|}{\sqrt{v_i}}+\dfrac{1}{2}\ln(2v_i)$",
        "props": [
            r"Var$(\varepsilon_i) = 2b_i^{2} = v_i$",
            r"penalty grows as $|r|$  (linear)",
            "exponential tails; kink at r = 0 makes\nthe Hessian stderr less reliable",
        ],
    },
    {
        "name": "Student-t",
        "config": "likelihood: student_t,  student_t_dof: 4",
        "noise": r"$\varepsilon_i = \sqrt{v_i}\;t,\quad t \sim t_{\nu},\ \ \nu = 4$",
        "pdf": r"$p(y_i\mid\theta)=\dfrac{\Gamma\!\left(\frac{\nu+1}{2}\right)}"
               r"{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\nu\pi v_i}}"
               r"\left(1+\dfrac{r_i^{2}}{\nu v_i}\right)^{-\frac{\nu+1}{2}}$",
        "nll": r"$\ell_i=\dfrac{1}{2}\ln v_i+\dfrac{\nu+1}{2}"
               r"\ln\!\left(1+\dfrac{r_i^{2}}{\nu v_i}\right)+C(\nu)$",
        "extra": r"$C(\nu)=-\ln\Gamma\!\left(\frac{\nu+1}{2}\right)"
                 r"+\ln\Gamma\!\left(\frac{\nu}{2}\right)+\frac{1}{2}\ln(\nu\pi)$"
                 + f"   (= {C_NU:.3f} at ν = 4)",
        "props": [
            r"$v_i$ is the scale$^2$: Var$(\varepsilon_i)=\frac{\nu}{\nu-2}v_i = 2v_i$",
            r"penalty grows as $\ln r^2$ once $r^2 \gg \nu v_i$",
            "power-law tails: robust to outliers;\n"
            r"$\nu\rightarrow\infty$ recovers the Gaussian exactly",
        ],
    },
]


def nll(name: str, r: np.ndarray, v: float = 1.0) -> np.ndarray:
    if name == "Gaussian":
        return r**2 / (2 * v) + 0.5 * np.log(2 * np.pi * v)
    if name == "Laplace":
        return np.sqrt(2) * np.abs(r) / np.sqrt(v) + 0.5 * np.log(2 * v)
    return 0.5 * np.log(v) + 0.5 * (NU + 1) * np.log1p(r**2 / (NU * v)) + C_NU


def panel(fig, x, y, w, h, face=PANEL, edge=GRID):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                  transform=fig.transFigure, facecolor=face,
                                  edgecolor=edge, lw=1.0, zorder=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/noise_models_likelihoods.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(17, 15.5), facecolor=SURFACE)

    # ---- Title -----------------------------------------------------------
    fig.text(0.5, 0.978, "Noise models and likelihoods used in the parameter optimization",
             ha="center", va="top", fontsize=21, weight="bold")
    fig.text(0.5, 0.953, "as implemented in autotune_gp/cost.py  "
             "(zrg_cost_function_gaussian_nll / _laplace / _student_t)",
             ha="center", va="top", fontsize=11.5, color=TEXT2)

    # ---- Shared setup ------------------------------------------------------
    panel(fig, 0.03, 0.845, 0.94, 0.09)
    fig.text(0.045, 0.924, "Shared setup, for every ZRG location i  "
             "(zonal band, region, or global) and every variable",
             va="top", fontsize=13, weight="bold")
    fig.text(0.045, 0.893,
             r"$y_i = \hat f_i(\theta) + \varepsilon_i \qquad "
             r"r_i = \hat f_i(\theta) - y_i \qquad "
             r"v_i = \max\!\left(\sigma^2_{\mathrm{GP},i}(\theta),\ 10^{-6}\right)$",
             va="center", fontsize=18)
    fig.text(0.045, 0.861,
             r"$y_i$ = observation,  $\hat f_i(\theta)$ = GP predictive mean at parameters "
             r"$\theta$,  $\sigma^2_{\mathrm{GP},i}$ = GP predictive variance.  "
             "Residuals are in standardised units (per-variable StandardScaler). "
             "No separate observational error is added:\nthe surrogate's own "
             "uncertainty sets the noise scale, so locations the GP is unsure "
             "about are down-weighted.",
             va="center", fontsize=10.5, color=TEXT2, linespacing=1.4)

    # ---- The three model columns ------------------------------------------
    col_w, gap, x0 = 0.300, 0.020, 0.030
    top, bot = 0.825, 0.365
    for k, m in enumerate(MODELS):
        x = x0 + k * (col_w + gap)
        c = SERIES[m["name"]]
        panel(fig, x, bot, col_w, top - bot, face="white")
        fig.add_artist(Rectangle((x, top - 0.052), 0.007, 0.052, transform=fig.transFigure,
                                 facecolor=c, edgecolor="none"))
        fig.text(x + 0.018, top - 0.017, m["name"], va="top", fontsize=19, weight="bold")
        fig.text(x + 0.018, top - 0.040, m["config"], va="top", fontsize=10,
                 color=TEXT2, family="monospace")

        cx = x + 0.014
        yy = top - 0.080
        fig.text(cx, yy, "Noise model", va="top", fontsize=11, weight="bold", color=TEXT2)
        fig.text(cx + 0.004, yy - 0.036, m["noise"], va="center", fontsize=15.5)

        yy -= 0.070
        fig.text(cx, yy, "Likelihood of one observation", va="top", fontsize=11,
                 weight="bold", color=TEXT2)
        fig.text(cx + 0.004, yy - 0.043, m["pdf"], va="center",
                 fontsize=14.5 if m["name"] != "Student-t" else 13.2)

        yy -= 0.090
        fig.text(cx, yy, r"Negative log-likelihood  $\ell_i = -\ln p(y_i\mid\theta)$",
                 va="top", fontsize=11, weight="bold", color=TEXT2)
        fig.text(cx + 0.004, yy - 0.040, m["nll"], va="center",
                 fontsize=15 if m["name"] != "Student-t" else 13.6)
        if "extra" in m:
            fig.text(cx + 0.004, yy - 0.078, m["extra"], va="center", fontsize=11.2,
                     color=TEXT2)

        yy -= 0.095
        fig.text(cx, yy, "Properties", va="top", fontsize=11, weight="bold", color=TEXT2)
        py = yy - 0.024
        for p in m["props"]:
            fig.text(cx + 0.004, py, "•", va="top", fontsize=11.5, color=TEXT2)
            fig.text(cx + 0.016, py, p, va="top", fontsize=11.5, linespacing=1.35)
            py -= 0.016 * (1 + p.count("\n")) + 0.010

    # ---- Aggregation ---------------------------------------------------------
    panel(fig, 0.03, 0.255, 0.94, 0.095)
    fig.text(0.045, 0.341, "Cost actually minimised (same for all three; only "
             r"$\ell_i$ changes)", va="top", fontsize=13, weight="bold")
    fig.text(0.045, 0.300,
             r"$J(\theta)=\sum_{\mathrm{var}}w_{\mathrm{var}}\left["
             r"w_z\,\overline{\ell}_{\mathrm{zonal}}+w_r\,\overline{\ell}_{\mathrm{regions}}"
             r"+w_g\,\ell_{\mathrm{global}}\right]$",
             va="center", fontsize=18)
    fig.text(0.515, 0.317,
             r"$\overline{\ell}$ = mean of $\ell_i$ over the 15 zonal bands or 6 regions, "
             "not a sum.",
             va="center", fontsize=11, color=TEXT2)
    fig.text(0.515, 0.294,
             r"Weights in these runs:  $w_{\mathrm{PCP}}=w_{\mathrm{OLR}}=w_{\mathrm{TLWP}}=0.3,\ "
             r"w_{\mathrm{OSR}}=0.1;\ \ w_z=w_r=w_g=1/3$",
             va="center", fontsize=11, color=TEXT2)
    fig.text(0.515, 0.271,
             "Means are used because neighbouring bands/regions are correlated, so J is a\n"
             "composite (pseudo-)likelihood: the argmin is unaffected, but stderrs are relative.",
             va="center", fontsize=10.5, color=TEXT2, linespacing=1.35)

    # ---- Shape plots --------------------------------------------------------
    axL = fig.add_axes([0.065, 0.078, 0.40, 0.140], facecolor=SURFACE)
    axR = fig.add_axes([0.565, 0.078, 0.40, 0.140], facecolor=SURFACE)

    e = np.linspace(-8, 8, 1601)
    r = np.linspace(0, 6, 601)
    for m in MODELS:
        n, c, ls = m["name"], SERIES[m["name"]], STYLE[m["name"]]
        axL.plot(e, np.exp(-nll(n, e)), color=c, ls=ls, lw=2.0, label=n)
        pen = nll(n, r) - nll(n, np.zeros(1))[0]
        axR.plot(r, pen, color=c, ls=ls, lw=2.0, label=n)

    axL.set_yscale("log")
    axL.set_ylim(1e-6, 1.0)
    axL.set_xlim(-8, 8)
    axL.set_xlabel(r"noise  $\varepsilon$   (in units of $\sqrt{v}$,  $v = 1$)", fontsize=11)
    axL.set_ylabel("density  (log scale)", fontsize=11)
    axL.set_title("Tails: how plausible is a large residual?", fontsize=12.5,
                  loc="left", color=TEXT, pad=8)

    axR.set_xlim(0, 6)
    axR.set_ylim(0, 12)
    axR.set_xlabel(r"residual  $|r|$   (in units of $\sqrt{v}$,  $v = 1$)", fontsize=11)
    axR.set_ylabel(r"cost added  $\ell(r)-\ell(0)$", fontsize=11)
    axR.set_title("Penalty growth: what one bad location costs", fontsize=12.5,
                  loc="left", color=TEXT, pad=8)

    # Direct labels (identity is never colour-alone: style + label + legend).
    # Place each label against its own curve so no label sits between two.
    def _pen(n, x):
        return float(nll(n, np.array([x]))[0] - nll(n, np.zeros(1))[0])
    g_x = float(np.sqrt(2 * 9.3))                     # Gaussian penalty = 9.3
    axR.text(g_x + 0.18, 9.3, "Gaussian", fontsize=11, color=TEXT, ha="left", va="center")
    axR.text(4.35, _pen("Laplace", 4.35) + 0.85, "Laplace", fontsize=11, color=TEXT,
             ha="left", va="center")
    axR.text(5.15, _pen("Student-t", 5.15) - 0.85, "Student-t", fontsize=11, color=TEXT,
             ha="left", va="center")

    for ax in (axL, axR):
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=10)
    axL.legend(frameon=False, fontsize=10.5, loc="upper right")

    fig.text(0.03, 0.018,
             "Plots use v = 1. Gaussian and Laplace are variance-matched (Var = v); "
             "the Student-t uses v as its scale², so its variance is 2v at ν = 4. "
             "Not shown: the plain 'mae' cost (single-fidelity ESEm run), "
             r"$\ell_i = |r_i|$ with no $v_i$ and no normalisation;"
             "\nit is not a likelihood, so no parameter-uncertainty report exists for it. "
             r"Optional extras in the code (off in these runs): "
             r"$+\kappa\,v_i$ variance penalty (variance_penalty_kappa = 0) "
             "and a training-distance penalty (training_distance_penalty_lambda = 0).",
             va="bottom", fontsize=9.8, color=TEXT2, linespacing=1.4)

    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
