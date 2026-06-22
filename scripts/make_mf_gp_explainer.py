# -*- coding: utf-8 -*-
"""
Generate a PDF explainer for the multi-fidelity AR1 GP:
matrix structure, numerical example, and training.

Usage:
    python scripts/make_mf_gp_explainer.py
Outputs:
    mf_gp_explainer.pdf  (in the current working directory)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

OUTPUT = "mf_gp_explainer.pdf"

# ── colour palette ────────────────────────────────────────────────────────────
C_LF   = "#2166ac"   # blue  – LF block
C_HF   = "#d6604d"   # red   – HF block
C_CROSS= "#4dac26"   # green – cross block
C_HEAD = "#333333"
C_BG   = "#f7f7f7"


# ── layout helpers ────────────────────────────────────────────────────────────
def new_page(pdf, title=""):
    fig = plt.figure(figsize=(8.5, 11))
    ax  = fig.add_axes([0.08, 0.04, 0.84, 0.90])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0.5, 0.97, title, ha="center", va="top",
                fontsize=15, fontweight="bold", color=C_HEAD,
                transform=ax.transAxes)
    return fig, ax


def txt(ax, x, y, s, **kw):
    """Shorthand for ax.text with sensible defaults."""
    kw.setdefault("va",        "top")
    kw.setdefault("ha",        "left")
    kw.setdefault("fontsize",  10.5)
    kw.setdefault("color",     "#222222")
    kw.setdefault("transform", ax.transAxes)
    kw.setdefault("linespacing", 1.55)
    return ax.text(x, y, s, **kw)


def section(ax, x, y, s, **kw):
    kw.setdefault("fontsize",   12)
    kw.setdefault("fontweight", "bold")
    kw.setdefault("color",      C_HEAD)
    return txt(ax, x, y, s, **kw)


def math(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 11)
    return txt(ax, x, y, f"${s}$", **kw)


def hline(ax, y, lw=0.6):
    ax.plot([0, 1], [y, y], color="#bbbbbb", lw=lw, transform=ax.transAxes)


# ── coloured matrix cell ──────────────────────────────────────────────────────
def cell(ax, x, y, w, h, text, facecolor="#ffffff", textcolor="#222222",
         fontsize=8.5, bold=False):
    rect = FancyBboxPatch((x, y - h), w, h,
                          boxstyle="square,pad=0",
                          linewidth=0.5, edgecolor="#999999",
                          facecolor=facecolor,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(x + w / 2, y - h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=textcolor,
            fontweight="bold" if bold else "normal",
            transform=ax.transAxes)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Title + The AR1 Model
# ═══════════════════════════════════════════════════════════════════════════════
def page_title(pdf):
    fig, ax = new_page(pdf)

    ax.text(0.5, 0.88,
            "Multi-Fidelity Gaussian Process\nSurrogate: Structure and Training",
            ha="center", va="top", fontsize=18, fontweight="bold",
            color=C_HEAD, transform=ax.transAxes, linespacing=1.5)

    ax.text(0.5, 0.76,
            "A walkthrough of the AR1 (Kennedy–O'Hagan) covariance matrix,\n"
            "a concrete numerical example, and how the model is trained.",
            ha="center", va="top", fontsize=11, color="#555555",
            transform=ax.transAxes, linespacing=1.6)

    hline(ax, 0.70)

    y = 0.67
    section(ax, 0.0, y, "1.  The AR1 Model")
    y -= 0.05
    txt(ax, 0.0, y,
        "The AR1 (autoregressive, order 1) multi-fidelity model relates a\n"
        "low-fidelity (LF) simulator to a high-fidelity (HF) one via:")
    y -= 0.10

    math(ax, 0.1, y,
         r"y_{\mathrm{HF}}(\mathbf{x})"
         r"\;=\; \rho \cdot y_{\mathrm{LF}}(\mathbf{x})"
         r"\;+\; \delta(\mathbf{x})",
         fontsize=13)
    y -= 0.09

    txt(ax, 0.0, y,
        r"where  $\mathbf{x} \in \mathbb{R}^{19}$  is the vector of tuning parameters and:")
    y -= 0.07

    entries = [
        (r"$y_{\mathrm{LF}}(\mathbf{x})$",
         "GP fitted primarily to LF (ne32) training data."),
        (r"$\rho$",
         "Learned scalar: how much the LF model scales to match HF."),
        (r"$\delta(\mathbf{x})$",
         "Discrepancy GP: HF-specific structure not captured by LF.\n"
         "        Uncorrelated with $y_{\\mathrm{LF}}$ by construction."),
    ]
    for sym, desc in entries:
        ax.text(0.04, y, sym, ha="left", va="top", fontsize=11,
                transform=ax.transAxes)
        txt(ax, 0.20, y, desc)
        y -= 0.085

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "2.  Fidelity Index — Augmented Input")
    y -= 0.055
    txt(ax, 0.0, y,
        "GPy/emukit internally appends a fidelity column to every training\n"
        "point so the kernel can distinguish LF from HF observations:")
    y -= 0.09

    math(ax, 0.1, y,
         r"\tilde{\mathbf{x}}_{\mathrm{LF}} = [\,x_1,\ldots,x_{19},\;0\,]"
         r"\qquad"
         r"\tilde{\mathbf{x}}_{\mathrm{HF}} = [\,x_1,\ldots,x_{19},\;1\,]",
         fontsize=11)
    y -= 0.08

    txt(ax, 0.0, y,
        "The kernel operates on this 20-dimensional augmented space and\n"
        "uses the last column only to decide which sub-kernel applies.")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Kernel / Matrix Block Structure
# ═══════════════════════════════════════════════════════════════════════════════
def page_kernel(pdf):
    fig, ax = new_page(pdf, "3.  The Joint Covariance Matrix")

    y = 0.88
    txt(ax, 0.0, y,
        "For $n_{\\mathrm{LF}}$ low-fidelity and $n_{\\mathrm{HF}}$ high-fidelity "
        "training points the full kernel\nmatrix $\\mathbf{K}$ has size "
        "$(n_{\\mathrm{LF}}+n_{\\mathrm{HF}})\\times(n_{\\mathrm{LF}}+n_{\\mathrm{HF}})$ "
        "and a $2\\times 2$ block structure:")
    y -= 0.10

    txt(ax, 0.12, y, r"$\mathbf{K}$ = (block diagram below)", fontsize=11, color="#555555")
    y -= 0.06

    # Draw coloured block diagram
    bx, by, bw, bh = 0.10, y - 0.01, 0.18, 0.09
    cell(ax, bx,        by,       bw, bh, "K_LL",  facecolor="#d1e5f0", bold=True)
    cell(ax, bx + bw,   by,       bw, bh, "K_LH",  facecolor="#d9f0d3", bold=True)
    cell(ax, bx,        by - bh,  bw, bh, "K_HL",  facecolor="#d9f0d3", bold=True)
    cell(ax, bx + bw,   by - bh,  bw, bh, "K_HH",  facecolor="#fddbc7", bold=True)

    ax.text(bx - 0.02, by - bh / 2,       "LF", ha="right", va="center",
            fontsize=9, color=C_LF, fontweight="bold", transform=ax.transAxes)
    ax.text(bx - 0.02, by - bh - bh / 2,  "HF", ha="right", va="center",
            fontsize=9, color=C_HF, fontweight="bold", transform=ax.transAxes)
    ax.text(bx + bw / 2,       by + 0.01, "LF", ha="center", va="bottom",
            fontsize=9, color=C_LF, fontweight="bold", transform=ax.transAxes)
    ax.text(bx + bw + bw / 2,  by + 0.01, "HF", ha="center", va="bottom",
            fontsize=9, color=C_HF, fontweight="bold", transform=ax.transAxes)

    y -= 0.24
    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Block Formulas")
    y -= 0.055

    blocks = [
        ("K_LL", C_LF,
         r"[\mathbf{K}_{\mathrm{LL}}]_{ij}"
         r"= k_0(\mathbf{x}^L_i,\,\mathbf{x}^L_j)",
         "LF–LF: only the base kernel $k_0$."),
        ("K_HH", C_HF,
         r"[\mathbf{K}_{\mathrm{HH}}]_{ij}"
         r"= \rho^2\,k_0(\mathbf{x}^H_i,\,\mathbf{x}^H_j)"
         r"+ k_1(\mathbf{x}^H_i,\,\mathbf{x}^H_j)",
         "HF–HF: scaled base kernel plus discrepancy $k_1$."),
        ("K_LH", C_CROSS,
         r"[\mathbf{K}_{\mathrm{LH}}]_{ij}"
         r"= \rho\,k_0(\mathbf{x}^L_i,\,\mathbf{x}^H_j)",
         "Cross block: only $\\rho\\cdot k_0$.  No $k_1$ — discrepancy\n"
         "        is uncorrelated with LF.  $\\mathbf{K}_{\\mathrm{HL}}"
         "=\\mathbf{K}_{\\mathrm{LH}}^\\top$."),
    ]
    for label, color, formula, desc in blocks:
        ax.text(0.0, y, label + ":", ha="left", va="top", fontsize=10,
                color=color, fontweight="bold", transform=ax.transAxes)
        math(ax, 0.08, y, formula, fontsize=10)
        y -= 0.055
        txt(ax, 0.08, y, desc)
        y -= 0.07

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "The RBF Sub-kernels  (ARD)")
    y -= 0.055
    txt(ax, 0.0, y,
        "Both $k_0$ and $k_1$ are Radial Basis Function kernels with "
        "Automatic Relevance\nDetermination (ARD) — one lengthscale per "
        "input parameter:")
    y -= 0.085
    math(ax, 0.1, y,
         r"k(\mathbf{x},\mathbf{x}')"
         r"= \sigma^2 \exp"
         r"\!\left(\!-\frac{1}{2}\sum_{i=1}^{19}"
         r"\frac{(x_i - x_i')^2}{\ell_i^2}\right)",
         fontsize=12)
    y -= 0.09
    txt(ax, 0.0, y,
        "A large $\\ell_i$ means the output is insensitive to parameter $i$;\n"
        "a small $\\ell_i$ means the output changes rapidly with $i$.")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Numerical Example setup
# ═══════════════════════════════════════════════════════════════════════════════
def page_example_setup(pdf):
    fig, ax = new_page(pdf, "4.  Numerical Example — Setup")

    y = 0.88
    txt(ax, 0.0, y,
        "To keep the numbers readable we use 1 input parameter, "
        "2 LF points, and 2 HF points.\n"
        "The structure is identical for 19 parameters and 256 points — "
        "just larger.")
    y -= 0.09

    section(ax, 0.0, y, "Training points")
    y -= 0.055
    rows = [
        ("$x_{L1}=0.2$", "LF", "0"),
        ("$x_{L2}=0.8$", "LF", "0"),
        ("$x_{H1}=0.3$", "HF", "1"),
        ("$x_{H2}=0.7$", "HF", "1"),
    ]
    col_w = [0.25, 0.15, 0.20]
    headers = ["Point", "Fidelity", "Fidelity index"]
    xs = [0.02, 0.27, 0.42]
    for hdr, x0 in zip(headers, xs):
        ax.text(x0, y, hdr, ha="left", va="top", fontsize=10,
                fontweight="bold", color=C_HEAD, transform=ax.transAxes)
    y -= 0.045
    hline(ax, y + 0.005)
    y -= 0.01
    for point, fid, idx in rows:
        color = C_LF if fid == "LF" else C_HF
        for val, x0 in zip([point, fid, idx], xs):
            ax.text(x0, y, val, ha="left", va="top", fontsize=10,
                    color=color, transform=ax.transAxes)
        y -= 0.042

    y -= 0.02
    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Kernel parameters chosen for this example")
    y -= 0.055
    params = [
        (r"$k_0$  (base RBF)",
         r"$\sigma^2_0 = 1.0,\quad \ell_0 = 0.5$"),
        (r"$k_1$  (discrepancy RBF)",
         r"$\sigma^2_1 = 0.5,\quad \ell_1 = 0.3$"),
        (r"$\rho$  (AR1 scale)",
         r"$\rho = 0.9 \;\Rightarrow\; \rho^2 = 0.81$"),
    ]
    for label, val in params:
        ax.text(0.02, y, label, ha="left", va="top", fontsize=10,
                transform=ax.transAxes)
        ax.text(0.38, y, val,   ha="left", va="top", fontsize=10,
                transform=ax.transAxes)
        y -= 0.052

    y -= 0.01
    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Base kernel evaluations  $k_0(x, x')$")
    y -= 0.055
    txt(ax, 0.0, y,
        r"Using  $k_0(x,x') = \exp(-(x-x')^2 / 2 \cdot 0.5^2)"
        r"= \exp(-2(x-x')^2)$:")
    y -= 0.065

    pts  = ["L1=0.2", "L2=0.8", "H1=0.3", "H2=0.7"]
    vals = np.array([
        [1.00, 0.49, 0.98, 0.61],
        [0.49, 1.00, 0.61, 0.98],
        [0.98, 0.61, 1.00, 0.73],
        [0.61, 0.98, 0.73, 1.00],
    ])
    cw, rh = 0.14, 0.042
    x0 = 0.14
    for ci, col in enumerate(pts):
        color = C_LF if col.startswith("L") else C_HF
        ax.text(x0 + ci * cw + cw / 2, y, col, ha="center", va="top",
                fontsize=9, fontweight="bold", color=color,
                transform=ax.transAxes)
    y -= 0.038
    for ri, row in enumerate(pts):
        color = C_LF if row.startswith("L") else C_HF
        ax.text(x0 - 0.02, y, row, ha="right", va="top", fontsize=9,
                fontweight="bold", color=color, transform=ax.transAxes)
        for ci, v in enumerate(vals[ri]):
            bg = "#e8f4fb" if (row.startswith("L") and pts[ci].startswith("L")) else \
                 "#fde8e4" if (row.startswith("H") and pts[ci].startswith("H")) else \
                 "#eaf6e4"
            cell(ax, x0 + ci * cw, y, cw, rh, f"{v:.2f}", facecolor=bg, fontsize=9)
        y -= rh + 0.004

    y -= 0.025
    txt(ax, 0.0, y,
        r"Discrepancy kernel at HF points only:  "
        r"$k_1(H1,H1)=0.50$,  "
        r"$k_1(H1,H2)=0.21$,  "
        r"$k_1(H2,H2)=0.50$")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – The full 4×4 matrix
# ═══════════════════════════════════════════════════════════════════════════════
def page_matrix(pdf):
    fig, ax = new_page(pdf, "5.  The Full 4×4 Joint Covariance Matrix")

    y = 0.88
    txt(ax, 0.0, y,
        "Assembling the blocks using the kernel values from page 3 "
        r"($\rho=0.9,\;\rho^2=0.81$):")
    y -= 0.07

    labels = ["L1", "L2", "H1", "H2"]
    colors = [C_LF, C_LF, C_HF, C_HF]
    matrix = np.array([
        [1.00,  0.49,  0.88,  0.55],
        [0.49,  1.00,  0.55,  0.88],
        [0.88,  0.55,  1.31,  0.80],
        [0.55,  0.88,  0.80,  1.31],
    ])
    cw, rh = 0.115, 0.06
    x0 = 0.22

    # column headers
    for ci, (lbl, col) in enumerate(zip(labels, colors)):
        ax.text(x0 + ci * cw + cw / 2, y + 0.005, f"[{lbl}]",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=col, transform=ax.transAxes)

    for ri, (rlbl, rcol) in enumerate(zip(labels, colors)):
        ax.text(x0 - 0.02, y - ri * (rh + 0.008) - rh / 2, f"[{rlbl}]",
                ha="right", va="center", fontsize=10,
                fontweight="bold", color=rcol, transform=ax.transAxes)
        for ci, v in enumerate(matrix[ri]):
            # Colour by block
            if ri < 2 and ci < 2:
                bg = "#cce5f6"   # LL – blue
            elif ri >= 2 and ci >= 2:
                bg = "#fcd8cf"   # HH – red
            else:
                bg = "#d8f0d0"   # cross – green
            bold = (ri == ci)
            cell(ax, x0 + ci * cw,
                 y - ri * (rh + 0.008),
                 cw, rh, f"{v:.2f}",
                 facecolor=bg, fontsize=11, bold=bold)

    # bracket lines
    mat_h = 4 * (rh + 0.008)
    mat_w = 4 * cw
    for xb, sign in [(x0 - 0.035, -1), (x0 + mat_w + 0.01, 1)]:
        for dy in [0, -mat_h + rh]:
            ax.annotate("", xy=(xb + sign * 0.015, y + dy + rh / 2),
                        xytext=(xb, y + dy + rh / 2),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-", color="#444"))
        ax.plot([xb, xb], [y + rh / 2, y - mat_h + rh + rh / 2],
                color="#444", lw=1.5, transform=ax.transAxes)

    y -= mat_h + 0.06

    # Legend
    for label, color, desc in [
        ("K_LL", "#cce5f6", "LF–LF:  k₀(Lᵢ, Lⱼ)"),
        ("K_HH", "#fcd8cf", "HF–HF:  ρ²·k₀(Hᵢ, Hⱼ) + k₁(Hᵢ, Hⱼ)  [diagonal = 0.81×1.00 + 0.50 = 1.31]"),
        ("K_LH", "#d8f0d0", "Cross:  ρ·k₀(Lᵢ, Hⱼ)                  [e.g. 0.9 × 0.98 = 0.88]"),
    ]:
        rect = FancyBboxPatch((0.0, y - 0.022), 0.04, 0.028,
                              boxstyle="square,pad=0",
                              facecolor=color, edgecolor="#999",
                              linewidth=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.06, y - 0.005, f"{label}:  {desc}", ha="left", va="top",
                fontsize=9.5, transform=ax.transAxes)
        y -= 0.042

    y -= 0.02
    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Noise: MixedNoise likelihood")
    y -= 0.055
    txt(ax, 0.0, y,
        "GPy adds a separate noise variance to each fidelity's diagonal entries:")
    y -= 0.065
    math(ax, 0.1, y,
         r"\mathbf{K}_{\mathrm{noisy}}"
         r"= \mathbf{K}"
         r"+ \mathrm{diag}(\,"
         r"\sigma^2_{LF},\ldots,\sigma^2_{LF},"
         r"\;\sigma^2_{HF},\ldots,\sigma^2_{HF}"
         r"\,)",
         fontsize=11)
    txt(ax, 0.52, y - 0.032,
        r"$\leftarrow n_{LF}$ entries $\rightarrow$        "
        r"$\leftarrow n_{HF}$ entries $\rightarrow$",
        fontsize=8.5, color="#555555")
    y -= 0.08
    txt(ax, 0.0, y,
        r"Both $\sigma^2_{LF}$ and $\sigma^2_{HF}$ are free hyperparameters "
        "learned during training.\n"
        "A larger $\\sigma^2_{HF}$ means the model doesn't try to interpolate "
        "every HF point exactly.")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Training
# ═══════════════════════════════════════════════════════════════════════════════
def page_training(pdf):
    fig, ax = new_page(pdf, "6.  Training: Marginal Likelihood Maximisation")

    y = 0.88
    txt(ax, 0.0, y,
        "All hyperparameters are found by maximising the "
        "log marginal likelihood — the\n"
        "probability of observing all training outputs given "
        "the inputs and the model:")
    y -= 0.09

    math(ax, 0.05, y,
         r"\log p(\mathbf{y}\mid\mathbf{X},\theta)"
         r"\;=\;"
         r"-\frac{1}{2}\,\mathbf{y}^T\mathbf{K}_{\mathrm{noisy}}^{-1}\mathbf{y}"
         r"\;-\;\frac{1}{2}\log|\mathbf{K}_{\mathrm{noisy}}|"
         r"\;-\;\frac{n}{2}\log 2\pi",
         fontsize=11.5)
    y -= 0.09

    txt(ax, 0.0, y,
        r"Here $\mathbf{y}$ stacks all training outputs (LF and HF), "
        r"$\mathbf{X}$ stacks all augmented inputs,\n"
        r"and $\theta$ is the set of hyperparameters listed below.\n"
        "The first term rewards fitting the data; the second penalises "
        "overly complex\nmodels (Bayesian Occam's razor). "
        "GPy maximises this using gradient ascent\n"
        r"(L-BFGS), computing $\partial \log p / \partial \theta$ "
        "analytically via the Cholesky\n"
        r"decomposition of $\mathbf{K}_{\mathrm{noisy}}$.")
    y -= 0.22

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Hyperparameters per AR1 model  (43 total)")
    y -= 0.055

    hp_rows = [
        (r"$\ell_{0,1},\ldots,\ell_{0,19}$",
         "19",
         "Base kernel $k_0$ ARD lengthscales — one per tuning parameter"),
        (r"$\sigma^2_0$",
         "1",
         "Base kernel $k_0$ variance (overall amplitude)"),
        (r"$\ell_{1,1},\ldots,\ell_{1,19}$",
         "19",
         "Discrepancy kernel $k_1$ ARD lengthscales"),
        (r"$\sigma^2_1$",
         "1",
         "Discrepancy kernel $k_1$ variance"),
        (r"$\rho$",
         "1",
         "AR1 scaling between fidelities"),
        (r"$\sigma^2_{LF},\;\sigma^2_{HF}$",
         "2",
         "Noise variances for LF and HF observations"),
    ]
    total_row = ("", "43", "Total")

    col_xs = [0.0, 0.38, 0.46]
    col_hdrs = ["Parameter", "Count", "Role"]
    for hdr, x0 in zip(col_hdrs, col_xs):
        ax.text(x0, y, hdr, ha="left", va="top", fontsize=10,
                fontweight="bold", color=C_HEAD, transform=ax.transAxes)
    y -= 0.042
    hline(ax, y + 0.005)
    y -= 0.012
    for sym, cnt, role in hp_rows:
        ax.text(col_xs[0], y, sym,  ha="left", va="top", fontsize=10,
                transform=ax.transAxes)
        ax.text(col_xs[1], y, cnt,  ha="left", va="top", fontsize=10,
                transform=ax.transAxes)
        ax.text(col_xs[2], y, role, ha="left", va="top", fontsize=10,
                color="#333333", transform=ax.transAxes)
        y -= 0.047
    hline(ax, y + 0.005)
    y -= 0.012
    ax.text(col_xs[1], y, "43", ha="left", va="top", fontsize=10,
            fontweight="bold", transform=ax.transAxes)
    ax.text(col_xs[2], y, "Total per AR1 model", ha="left", va="top",
            fontsize=10, fontweight="bold", transform=ax.transAxes)

    y -= 0.06
    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Objective: Joint, not HF-only")
    y -= 0.055
    txt(ax, 0.0, y,
        r"The marginal likelihood sums over all $n_{LF}+n_{HF}$ observations. "
        "It is not\n"
        "weighted toward HF accuracy. HF surrogate quality emerges because the\n"
        r"model must explain the HF observations, and $k_1$ and $\rho$ are the "
        "only\nparameters that can absorb structure unique to HF once $k_0$ is\n"
        "constrained by the more numerous LF points.")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Real problem scale + SF vs MF comparison
# ═══════════════════════════════════════════════════════════════════════════════
def page_scale(pdf):
    fig, ax = new_page(pdf, "9.  Real Problem Scale and Single vs Multi-Fidelity")

    y = 0.88
    section(ax, 0.0, y, "Scale in this application")
    y -= 0.055
    items = [
        (r"Training data",
         r"$n_{LF}=175$ ne32 runs  +  $n_{HF}=81$ ne256 runs  "
         r"$\Rightarrow$ 256 $\times$ 256 matrix per AR1 model"),
        ("Parameters",
         r"$p=19$ tuning parameters $\Rightarrow$ 19 ARD lengthscales in each of $k_0$, $k_1$"),
        ("Number of AR1 models",
         "66  (one per ZRG feature $\\times$ variable: "
         "22 features $\\times$ 3 variables)"),
        ("Hyperparameters (total)",
         "$43 \\times 66 = 2{,}838$  learned independently across all models"),
    ]
    for label, val in items:
        ax.text(0.0, y, "•", ha="left", va="top", fontsize=12,
                transform=ax.transAxes)
        ax.text(0.04, y, f"{label}:  ", ha="left", va="top", fontsize=10,
                fontweight="bold", transform=ax.transAxes)
        txt(ax, 0.04, y - 0.038, val, fontsize=10)
        y -= 0.088

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Single-fidelity vs Multi-fidelity comparison")
    y -= 0.055

    col_xs  = [0.0,   0.38,  0.68]
    col_hdrs = ["",   "Single-fidelity (SF)", "Multi-fidelity (MF)"]
    for hdr, x0 in zip(col_hdrs, col_xs):
        ax.text(x0, y, hdr, ha="left", va="top", fontsize=10,
                fontweight="bold", color=C_HEAD, transform=ax.transAxes)
    y -= 0.042
    hline(ax, y + 0.005)
    y -= 0.012

    rows = [
        ("Framework",         "GPflow / ESEm",            "GPy / emukit"),
        ("Number of models",  "1 joint GPR",              "66 AR1 models"),
        ("Fidelity",          "HF only (ne256, 81 pts)",  "LF + HF (175 + 81 pts)"),
        ("Hyperparameters",   "21  (shared across all\n   66 outputs)",
                              "43 per model\n   (2,838 total, independent)"),
        ("Kernel per output", "Identical $k$ for all 66", "Own $k_0, k_1, \\rho$ per output"),
        ("Regularisation",    "Strong (21 params fit\n"
                              "   from 81$\\times$66 data)",
                              "Weaker (43 params per\n   model, 256 points)"),
        ("Discrepancy",       "None — single fidelity",
                              "$\\delta(\\mathbf{x})$ captures ne256 vs ne32 difference"),
    ]
    for label, sf, mf in rows:
        ax.text(col_xs[0], y, label, ha="left", va="top", fontsize=9.5,
                fontweight="bold", color=C_HEAD, transform=ax.transAxes)
        ax.text(col_xs[1], y, sf,   ha="left", va="top", fontsize=9.5,
                transform=ax.transAxes, linespacing=1.4)
        ax.text(col_xs[2], y, mf,   ha="left", va="top", fontsize=9.5,
                transform=ax.transAxes, linespacing=1.4)
        y -= 0.072

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Interpreting the learned hyperparameters")
    y -= 0.055
    interp = [
        (r"$\rho \approx 1$",
         "LF and HF models agree; the AR1 correction is small."),
        (r"$\rho \ll 1$ or $\rho \gg 1$",
         "LF is a poor proxy for HF; the scaling is large."),
        (r"$\sigma^2_1 \gg \sigma^2_0$",
         "Discrepancy dominates — HF has structure LF cannot explain.\n"
         "       Reported as discrepancy fraction $= \\sigma^2_1/(\\sigma^2_0+\\sigma^2_1)$."),
        (r"Small $\ell_{0,i}$ or $\ell_{1,i}$",
         "Parameter $i$ strongly affects the output for that ZRG feature."),
    ]
    for sym, desc in interp:
        ax.text(0.02, y, sym,  ha="left", va="top", fontsize=10,
                transform=ax.transAxes)
        txt(ax, 0.28, y, desc, fontsize=10)
        y -= 0.065

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6b – Inference
# ═══════════════════════════════════════════════════════════════════════════════
def page_inference(pdf):
    fig, ax = new_page(pdf, "7.  Inference: Predicting at a New Parameter Set")

    y = 0.88
    txt(ax, 0.0, y,
        r"Once training is done ($\mathbf{K}_{\mathrm{noisy}}$ is factorised), "
        "predicting at a new parameter\n"
        r"set $\mathbf{x}^*$ requires only a matrix–vector multiply.  "
        "The GP posterior gives both\na mean prediction and an uncertainty.")
    y -= 0.10

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Predictive mean and variance")
    y -= 0.055

    math(ax, 0.07, y,
         r"\mu^* = \mathbf{k}_*^T \, \mathbf{K}_{\mathrm{noisy}}^{-1} \, \mathbf{y}",
         fontsize=13)
    y -= 0.075
    math(ax, 0.07, y,
         r"\sigma^{2*} = k_{**} - \mathbf{k}_*^T \, \mathbf{K}_{\mathrm{noisy}}^{-1} \, \mathbf{k}_*",
         fontsize=13)
    y -= 0.08

    txt(ax, 0.0, y,
        r"where $\mathbf{y}$ is the stacked vector of all training outputs (LF then HF),\n"
        r"$\mathbf{k}_*$ is the $(n_{LF}+n_{HF})$-length cross-covariance between "
        r"$\mathbf{x}^*$ (at HF fidelity)\nand every training point, "
        r"and $k_{**}$ is the prior variance at $\mathbf{x}^*$.")
    y -= 0.13

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, r"What is in $\mathbf{k}_*$?")
    y -= 0.055
    txt(ax, 0.0, y,
        r"Because we predict at fidelity 1 (HF), the cross-covariance entries "
        "are read\nfrom the HF row of the block kernel:")
    y -= 0.085

    for label, color, formula, desc in [
        ("LF entries", C_LF,
         r"[\mathbf{k}_*]_i = \rho \, k_0(\mathbf{x}^*, \mathbf{x}^L_i)",
         r"one entry per LF training point ($i = 1,\ldots,n_{LF}$)"),
        ("HF entries", C_HF,
         r"[\mathbf{k}_*]_j = \rho^2 k_0(\mathbf{x}^*, \mathbf{x}^H_j)"
         r"+ k_1(\mathbf{x}^*, \mathbf{x}^H_j)",
         r"one entry per HF training point ($j = 1,\ldots,n_{HF}$)"),
        ("Prior var.", "#888888",
         r"k_{**} = \rho^2 \sigma^2_0 + \sigma^2_1",
         "the HF self-covariance (no noise added here)"),
    ]:
        ax.text(0.0, y, label + ":", ha="left", va="top", fontsize=10,
                color=color, fontweight="bold", transform=ax.transAxes)
        math(ax, 0.15, y, formula, fontsize=10)
        y -= 0.052
        txt(ax, 0.15, y, desc, fontsize=9.5, color="#444444")
        y -= 0.062

    y -= 0.005
    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Key insight: LF data contributes to HF predictions")
    y -= 0.055
    txt(ax, 0.0, y,
        r"Even when predicting at fidelity 1 (HF), the LF training points "
        "appear in $\mathbf{k}_*$\n"
        r"(via $\rho \, k_0$) and in $\mathbf{K}_{\mathrm{noisy}}^{-1}$.  "
        "The 175 LF runs therefore inform every\n"
        "HF prediction — not just as extra noise, but as signal that "
        "partially determines\n"
        r"where $\mu^*$ lands.  "
        r"The more LF data, and the closer $\rho$ is to 1, the more it helps.")
    y -= 0.145

    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Computational cost")
    y -= 0.055
    rows = [
        ("Training (once)",
         r"$O(n^3)$ — Cholesky of $\mathbf{K}_{\mathrm{noisy}}$ (n = 256)."
         "\nDone once; the triangular factor is stored."),
        ("Prediction per point",
         r"$O(n^2)$ — evaluate $\mathbf{k}_*$ (n kernel calls) then"
         "\ntwo triangular solves using the stored Cholesky."),
        ("Prediction per batch",
         r"$O(m \cdot n)$ — for $m$ test points, dominated by "
         r"computing the $m \times n$ matrix $\mathbf{K}_{*}$."),
    ]
    for label, desc in rows:
        ax.text(0.0, y, label + ":", ha="left", va="top", fontsize=10,
                fontweight="bold", color=C_HEAD, transform=ax.transAxes)
        txt(ax, 0.35, y, desc, fontsize=10, linespacing=1.45)
        y -= 0.085

    y -= 0.01
    hline(ax, y + 0.01)
    y -= 0.03

    section(ax, 0.0, y, "Using the uncertainty")
    y -= 0.055
    txt(ax, 0.0, y,
        r"$\sigma^{2*}$ is large where training data is sparse — the model is "
        "honest about what\n"
        "it doesn't know.  The optimizer uses only $\\mu^*$ (the mean) to "
        "evaluate candidate\n"
        "parameter sets.  The uncertainty could additionally be used for "
        "Bayesian optimisation\n"
        "(acquisition functions such as Expected Improvement), "
        "which would trade off\n"
        "exploitation vs exploration of the parameter space.")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7b – Worked inference example
# ═══════════════════════════════════════════════════════════════════════════════
def page_inference_example(pdf):
    """Worked numerical example: predict at x*=0.5 using the 2LF+2HF toy data."""
    # Hyperparameters — identical to page_example_setup
    sig2_0, l0       = 1.0, 0.5
    sig2_1, l1       = 0.5, 0.3
    rho              = 0.9
    sig2_lf, sig2_hf = 0.10, 0.05

    def k0(x, xp): return sig2_0 * np.exp(-(x - xp)**2 / (2 * l0**2))
    def k1(x, xp): return sig2_1 * np.exp(-(x - xp)**2 / (2 * l1**2))

    xL     = [0.2, 0.8]
    xH     = [0.3, 0.7]
    x_star = 0.50

    # Training outputs: y_true = x², LF ≈ rho·x², HF = x²
    y = np.array([rho * x**2 for x in xL] + [x**2 for x in xH])

    # Build K_noisy (4×4)
    K = np.zeros((4, 4))
    for i, xi in enumerate(xL):
        for j, xj in enumerate(xL):
            K[i, j] = k0(xi, xj)
        for j, xj in enumerate(xH):
            K[i, 2+j] = rho * k0(xi, xj)
            K[2+j, i] = K[i, 2+j]
    for i, xi in enumerate(xH):
        for j, xj in enumerate(xH):
            K[2+i, 2+j] = rho**2 * k0(xi, xj) + k1(xi, xj)
    K_noisy = K.copy()
    K_noisy[0, 0] += sig2_lf;  K_noisy[1, 1] += sig2_lf
    K_noisy[2, 2] += sig2_hf;  K_noisy[3, 3] += sig2_hf

    # k_star: cross-covariance between x* at HF fidelity and all training points
    k_star = np.array(
        [rho * k0(x_star, xi) for xi in xL] +
        [rho**2 * k0(x_star, xi) + k1(x_star, xi) for xi in xH]
    )
    k_ss = rho**2 * sig2_0 + sig2_1   # = 1.31

    alpha    = np.linalg.solve(K_noisy, y)
    mu_star  = float(k_star @ alpha)
    var_star = float(k_ss - k_star @ np.linalg.solve(K_noisy, k_star))
    sig_star = float(np.sqrt(max(var_star, 0.0)))
    true_val = x_star**2   # 0.25

    fig, ax = new_page(pdf, r"8.  Worked Inference Example — Predict at $x^* = 0.5$")

    y0 = 0.88
    txt(ax, 0.0, y0,
        r"Same toy setup: 2 LF points (0.2, 0.8) + 2 HF points (0.3, 0.7)."
        r"  $y_{\rm true}(x)=x^2$."
        "\nLF outputs = "
        r"$\rho x^2$, HF outputs = $x^2$.  "
        r"Noise: $\sigma^2_{LF}=0.10,\;\sigma^2_{HF}=0.05$."
        "\n"
        r"Stacked output vector: $\mathbf{y} = ["
        + ",\;".join(f"{v:.3f}" for v in y)
        + r"]^T$")
    y0 -= 0.115

    hline(ax, y0 + 0.005);  y0 -= 0.025

    # ── Step 1: k_star ───────────────────────────────────────────────────────
    section(ax, 0.0, y0, r"Step 1 — Cross-covariance vector $\mathbf{k}_*$"
            r"  (4 entries, one per training point)")
    y0 -= 0.048

    rows_k = [
        ("L1", C_LF,
         r"\rho\,k_0(0.5,\,0.2)=0.9\times e^{-0.18}",
         k_star[0], ""),
        ("L2", C_LF,
         r"\rho\,k_0(0.5,\,0.8)=0.9\times e^{-0.18}",
         k_star[1], "(equal to L1 — equidistant)"),
        ("H1", C_HF,
         r"\rho^2 k_0(0.5,\,0.3)+k_1(0.5,\,0.3)"
         r"=0.81\times 0.923+0.400",
         k_star[2], ""),
        ("H2", C_HF,
         r"\rho^2 k_0(0.5,\,0.7)+k_1(0.5,\,0.7)",
         k_star[3], "(equal to H1 — equidistant)"),
    ]
    for label, color, formula, val, note in rows_k:
        ax.text(0.0, y0, label + ":", ha="left", va="top",
                fontsize=10, fontweight="bold", color=color,
                transform=ax.transAxes)
        math(ax, 0.06, y0, formula, fontsize=9)
        ax.text(0.72, y0, f"= {val:.3f}", ha="left", va="top",
                fontsize=10, fontweight="bold", color=color,
                transform=ax.transAxes)
        if note:
            txt(ax, 0.83, y0, note, fontsize=8.5, color="#666666")
        y0 -= 0.045

    txt(ax, 0.0, y0,
        r"HF entries (1.148) are larger than LF entries (0.752) because"
        r" $k_1$ adds to $\rho^2 k_0$.",
        fontsize=9, color="#555555")
    y0 -= 0.048

    hline(ax, y0 + 0.005);  y0 -= 0.022

    # ── Step 2: prior variance ────────────────────────────────────────────────
    section(ax, 0.0, y0, r"Step 2 — Prior HF variance  $k_{**}$")
    y0 -= 0.045
    math(ax, 0.06, y0,
         rf"k_{{**}} = \rho^2\sigma^2_0 + \sigma^2_1"
         rf"= 0.81\times1.0 + 0.5 = {k_ss:.2f}",
         fontsize=10.5)
    txt(ax, 0.67, y0,
        "(before seeing any data, variance = 1.31)",
        fontsize=9, color="#555555")
    y0 -= 0.055

    hline(ax, y0 + 0.005);  y0 -= 0.022

    # ── Step 3: alpha ─────────────────────────────────────────────────────────
    section(ax, 0.0, y0,
            r"Step 3 — Weights  $\alpha = \mathbf{K}_{\rm noisy}^{-1}\,\mathbf{y}$"
            r"  (solved via Cholesky)")
    y0 -= 0.045

    for i, (v, lbl, col) in enumerate(zip(alpha,
                                          ["L1", "L2", "H1", "H2"],
                                          [C_LF, C_LF, C_HF, C_HF])):
        xoff = 0.06 + i * 0.23
        ax.text(xoff, y0, lbl, ha="center", va="top",
                fontsize=9, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(xoff, y0 - 0.030, f"{v:+.3f}", ha="center", va="top",
                fontsize=10, color=col, transform=ax.transAxes)
    y0 -= 0.062

    txt(ax, 0.0, y0,
        r"$\alpha$ is not symmetric: $y$ increases with $x^2$ so L2 and H2 pull"
        " harder than L1 and H1.",
        fontsize=9, color="#555555")
    y0 -= 0.048

    hline(ax, y0 + 0.005);  y0 -= 0.022

    # ── Step 4: mu* ───────────────────────────────────────────────────────────
    section(ax, 0.0, y0, r"Step 4 — Predictive mean  $\mu^* = \mathbf{k}_*^T \alpha$")
    y0 -= 0.045

    terms     = [k_star[i] * alpha[i] for i in range(4)]
    lbls      = ["L1","L2","H1","H2"]
    cols      = [C_LF,C_LF,C_HF,C_HF]

    # Show each term
    parts_str = "  +  ".join(
        f"({k_star[i]:.3f})({'+' if alpha[i] >= 0 else ''}{alpha[i]:.3f})"
        for i in range(4)
    )
    txt(ax, 0.0, y0, r"$\mu^*$ = " + parts_str, fontsize=9.5)
    y0 -= 0.042

    contribs_str = "  +  ".join(f"{t:.3f}" for t in terms)
    txt(ax, 0.0, y0, "       = " + contribs_str, fontsize=9.5)
    y0 -= 0.042

    ax.text(0.0, y0,
            f"       = {mu_star:.3f}",
            ha="left", va="top", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    txt(ax, 0.25, y0,
        f"(true value $x^{{*2}} = {true_val:.2f}$,"
        f"  GP error = {abs(mu_star - true_val):.3f})",
        fontsize=10, color="#444444")
    y0 -= 0.058

    hline(ax, y0 + 0.005);  y0 -= 0.022

    # ── Step 5: sigma* ────────────────────────────────────────────────────────
    section(ax, 0.0, y0,
            r"Step 5 — Predictive std  "
            r"$\sigma^* = \sqrt{k_{**} - \mathbf{k}_*^T \mathbf{K}_{\rm noisy}^{-1} \mathbf{k}_*}$")
    y0 -= 0.045

    reduction = k_ss - var_star
    math(ax, 0.06, y0,
         rf"\sigma^{{2*}} = {k_ss:.2f} - {reduction:.3f} = {var_star:.3f}"
         rf"\quad\Longrightarrow\quad"
         rf"\sigma^* = {sig_star:.3f}",
         fontsize=11)
    y0 -= 0.055

    txt(ax, 0.0, y0,
        f"The prior variance {k_ss:.2f} shrinks by {reduction:.3f} because x*=0.5"
        f" lies between H1=0.3 and H2=0.7.\n"
        f"sigma* = {sig_star:.3f} is small — the model is confident here.\n"
        "Far from all training points the variance would stay close to the prior 1.31.",
        fontsize=9.5, color="#555555")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    with PdfPages(OUTPUT) as pdf:
        # Set PDF metadata
        info = pdf.infodict()
        info["Title"]   = "Multi-Fidelity GP: Structure and Training"
        info["Subject"] = "AR1 Kennedy-O'Hagan multi-fidelity Gaussian Process"

        page_title(pdf)
        page_kernel(pdf)
        page_example_setup(pdf)
        page_matrix(pdf)
        page_training(pdf)
        page_inference(pdf)
        page_inference_example(pdf)
        page_scale(pdf)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
