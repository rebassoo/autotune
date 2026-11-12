"""
Zonal-mean precipitation at the optima of the PCP-weight experiments, against obs,
the default parameters and the ne256 PPE spread: (a) PCP, (b) (PCP - obs) / obs in %.
Values are the 3-level surrogate's predictions at each optimum (projection_data).

    python scripts/plot_pcp_fits.py --out docs/pcp_fits_n85.png
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = Path("/pscratch/sd/r/rebassoo/autotune")
PPE = Path("/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_n85/gp_proj.pkl")
RUNS = [  # (label, results dir, colour, marker)
    ("nominal (PCP 0.3, OSR 0.1, OLR 0.3, TLWP 0.3)", "results_mf3_ne256_n85_studentt_dof4", "#2a6fdb", "o"),
    ("PCP only (other variables 0)", "results_mf3_ne256_n85_pcponly", "#e8833a", "s"),
    ("PCP global weight 0 (zonal, regional 1/2 each)", "results_mf3_ne256_n85_pcpg0", "#2e9e5b", "^"),
]
LAT = np.array([-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75], float)
MMD = 86400e3
INK, GREY = "#1a1a1a", "#9aa0a6"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L = lambda p: pickle.load(open(p, "rb"))
    pd = {d: L(S / d / "diagnostics" / "projection_data_seed50.pkl") for _, d, _, _ in RUNS}
    ref = pd[RUNS[0][1]]
    j = list(ref["var_names"]).index("PCP")
    obs = np.asarray(ref["obs"], float)[:, j] * MMD
    dflt = np.asarray(ref["hf_default"], float)[:, j] * MMD
    Y = np.asarray(L(PPE)["Y_train"], float)[:, :, 0] * MMD
    lo, hi = np.percentile(Y, [5, 95], axis=0)
    rms = lambda e: np.sqrt(np.mean(e[:15] ** 2))
    stats = lambda y: (f"RMS {rms(y - obs):.2f} mm/d, {rms(100 * (y - obs) / obs):.1f}%; "
                       f"global {100 * (y[-1] - obs[-1]) / obs[-1]:+.1f}%")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for k, a in enumerate(ax):
        tf = (lambda v: v) if k == 0 else (lambda v: 100 * (v - obs) / obs)
        a.fill_between(LAT, tf(lo)[:15], tf(hi)[:15], color=GREY, alpha=0.22, lw=0,
                       label=f"ne256 PPE, 5–95% (n={len(Y)})")
        a.plot(LAT, tf(obs)[:15], color=INK, lw=2.4, marker="D", ms=5, label="obs (IMERG)")
        a.plot(LAT, tf(dflt)[:15], color=GREY, lw=1.6, ls="--", marker="x", ms=6,
               label=f"default parameters  ({stats(dflt)})")
        for lab, d, c, m in RUNS:
            y = np.asarray(pd[d]["hf_optimal"], float)[:, j] * MMD
            a.plot(LAT, tf(y)[:15], color=c, lw=2, marker=m, ms=5, label=f"{lab}  ({stats(y)})")
        a.axvline(0, color=GREY, lw=0.5)
        a.set_xlabel("latitude band centre (°)")
        a.set_xticks(LAT[::2])
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("precipitation (mm/d)")
    ax[0].set_title("(a) Zonal-mean precipitation", loc="left", fontsize=11)
    ax[1].axhline(0, color=INK, lw=0.8)
    ax[1].set_ylabel("(model − obs) / obs (%)")
    ax[1].set_title("(b) Error against obs, as a percentage of obs", loc="left", fontsize=11)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.8, frameon=False)
    fig.suptitle("Precipitation at the optimum of each PCP weighting (3-level surrogate, n = 85)",
                 fontsize=12.5, x=0.01, ha="left")
    fig.text(0.99, 0.965, "RMS over the 15 bands; global-mean error",
             fontsize=8.5, color=GREY, ha="right")
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.savefig(args.out, dpi=150, facecolor="white")
    for lab, d, _, _ in [("default", None, None, None)] + RUNS:
        y = dflt if d is None else np.asarray(pd[d]["hf_optimal"], float)[:, j] * MMD
        e = y - obs
        print(f"{lab[:30]:32} bands RMS {rms(e):.3f}  regions RMS {np.sqrt(np.mean(e[15:21] ** 2)):.3f}  "
              f"bands+regions RMS {np.sqrt(np.mean(e[:21] ** 2)):.3f}  bands % RMS {rms(100 * e / obs):.1f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
