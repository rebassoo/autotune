"""
Illustration (toy problem, not the real surrogate): what a larger model-error
variance sigma_delta^2 does to the fit.

One tuning parameter theta, two variables with 10 features each, the same
Student-t cost (dof=4) and per-variable weights as the optimizer:
  PCP  wants theta = 0.30 but carries a structural misfit (RMS 0.6, theta can't remove it)
  OLR  wants theta = 0.70, fits well, but is only weakly sensitive to theta
The variance at each feature is V = sigma_GP^2 + sigma_obs^2 + sigma_delta^2.
Only PCP's sigma_delta changes between the curves.

    python scripts/plot_model_error_effect.py --out docs/model_error_effect.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOF = 4.0
INK, MUTED = "#1a1a1a", "#6b6f76"
SHADES = ["#9ec5f4", "#4a8ee6", "#1b4f9c"]          # one hue, light -> dark = small -> large sigma_delta


def term(r2, V):
    return 0.5 * np.log(V) + 0.5 * (DOF + 1) * np.log1p(r2 / (DOF * V))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(3)
    nf = 10
    a = rng.uniform(1.5, 3.0, nf)                    # PCP sensitivities d y / d theta
    b = rng.uniform(0.3, 0.6, nf)                    # OLR sensitivities (weak)
    d = rng.normal(0, 1, nf)
    d *= 0.6 / np.sqrt(np.mean(d ** 2))              # PCP structural misfit, RMS 0.6
    var_P, var_O = 0.34 ** 2 + 0.1 ** 2, 0.2 ** 2 + 0.05 ** 2   # GP + obs variance
    w_P, w_O = 0.3, 0.3

    th = np.linspace(0, 1, 801)
    rP = a[None] * (th[:, None] - 0.30) + d[None]
    rO = b[None] * (th[:, None] - 0.70)
    costO = w_O * term(rO ** 2, var_O).mean(1)

    sigmas = [0.0, 0.3, 1.05]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for s, col in zip(sigmas, SHADES):
        V = var_P + s ** 2
        c = w_P * term(rP ** 2, V).mean(1) + costO
        dc = c - c.min()
        k = th[np.argmin(c)]
        lab = f"$\\sigma_\\delta$ = {s:g}"
        axs[0].plot(th, dc, color=col, lw=2, label=lab)
        axs[0].plot(k, 0, "o", ms=8, color=col, mec="white", mew=1.5, zorder=3)
        L = np.exp(-20 * dc)                         # illustrative effective sample size
        L /= np.trapz(L, th)
        mu = np.trapz(th * L, th)
        sd = np.sqrt(np.trapz((th - mu) ** 2 * L, th))
        axs[1].plot(th, L, color=col, lw=2, label=f"{lab}: optimum {k:.2f}, sd {sd:.3f}")

    for ax in axs[:2]:
        for x0, name in ((0.30, "PCP alone"), (0.70, "OLR alone")):
            ax.axvline(x0, color=MUTED, lw=0.8, ls=":")
            ax.text(x0, 1.0, name, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=8, color=MUTED)
        ax.set_xlabel("tuning parameter $\\theta$ (normalized)")
    axs[0].set_ylim(0, 0.25)
    axs[0].set_ylabel("cost minus its minimum")
    axs[0].set_title("(a) cost: optimum moves, valley flattens", loc="left", fontsize=10, pad=16)
    axs[0].legend(frameon=False, fontsize=8, title="PCP model error", title_fontsize=8)
    axs[1].set_ylabel("relative likelihood exp(-N cost), normalized")
    axs[1].set_title("(b) parameter uncertainty widens", loc="left", fontsize=10, pad=16)
    axs[1].legend(frameon=False, fontsize=7.5, loc="upper right")

    # (c) how much a PCP feature counts relative to sigma_delta = 0
    s = np.linspace(0, 1.5, 200)
    axs[2].plot(s, var_P / (var_P + s ** 2), color=SHADES[2], lw=2)
    for si, col in zip(sigmas, SHADES):
        axs[2].plot(si, var_P / (var_P + si ** 2), "o", ms=8, color=col, mec="white", mew=1.5)
    axs[2].annotate("E3 PCP zonal (1.05):\n~10x less weight", (1.05, var_P / (var_P + 1.05 ** 2)),
                    xytext=(0.75, 0.45), fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axs[2].set_xlabel("$\\sigma_\\delta$ (PPE std units)")
    axs[2].set_ylabel("weight of a PCP feature, 1/V, relative to $\\sigma_\\delta$=0")
    axs[2].set_ylim(0, 1.05)
    axs[2].set_title("(c) larger variance = smaller weight", loc="left", fontsize=10, pad=16)

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Effect of a larger model-error variance on the fit (toy problem, Student-t dof=4)",
                 fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "V = GP variance + obs variance + $\\sigma_\\delta^2$ per feature; only PCP's "
             "$\\sigma_\\delta$ changes. PCP is strongly sensitive to $\\theta$ but has an unfittable misfit "
             "(RMS 0.6); OLR fits exactly at 0.70 but is weakly sensitive.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
