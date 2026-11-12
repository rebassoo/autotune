"""
Precipitation error against obs (model - obs) by zonal band for actual ne256 PPE
members: the members whose PC1 score (EOF1 of the PPE) is closest to -2, -1, +1 and
+2 sd, with the PPE mean and, faintly, all 85 members coloured by PC1. The real-member
counterpart of panel (b) of plot_pcp_eof1_shift.py, where the curves are the pure EOF1
shift. EOFs as in plot_pcp_eofs.py (15 bands + 6 regions, mm/d, unweighted).

    python scripts/plot_pcp_members_error.py --out docs/pcp_members_error_n85.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.io import load_obs, split_zrg_obs   # noqa: E402

P = Path("/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_n85")
LAT = np.array([-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75], float)
MMD = 86400e3
NZ, F = 15, 21
INK, MUTED, GREY = "#1a1a1a", "#6b6f76", "#c9ccd1"
DIV = LinearSegmentedColormap.from_list("pc", ["#2a6fdb", "#c9ccd1", "#d9542b"])


def short(name):
    """'...F2010-SCREAMv1.20260910_nearbal.m202' -> '20260910_nearbal.m202' (batch.member)."""
    parts = name.split(".")
    return ".".join(parts[-2:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=float, nargs="+", default=[-2, -1, 1, 2],
                    help="PC1 scores (sd); the member closest to each is highlighted")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gp = pickle.load(open(P / "gp_proj.pkl", "rb"))
    names = list(gp["X_train"].index)
    Y = np.asarray(gp["Y_train"], float)[:, :, 0] * MMD
    obs = np.asarray(split_zrg_obs(load_obs(str(P / "obs.pkl"))["zrg_obs"], n_vars=4)[0],
                     float).ravel() * MMD
    n = len(Y)
    mu = Y[:, :F].mean(0)
    U, S, Vt = np.linalg.svd(Y[:, :F] - mu, full_matrices=False)
    if Vt[0, :NZ].mean() < 0:
        U[:, 0] = -U[:, 0]
    pc1 = U[:, 0] * np.sqrt(n)
    rms = lambda e: float(np.sqrt(np.mean(e[:NZ] ** 2)))
    err = Y[:, :NZ] - obs[:NZ]

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    norm = Normalize(-2.5, 2.5)
    for i in np.argsort(np.abs(pc1)):
        ax.plot(LAT, err[i], color=DIV(norm(pc1[i])), lw=0.7, alpha=0.3, zorder=1)
    picked = []
    for t in args.targets:
        i = int(np.argmin(np.where(np.isin(np.arange(n), picked), np.inf, np.abs(pc1 - t))))
        picked.append(i)
    for i in sorted(picked, key=lambda i: pc1[i]):
        ax.plot(LAT, err[i], color=DIV(norm(pc1[i])), lw=2.2, marker="o", ms=4, zorder=3,
                label=f"{short(names[i])}: PC1 {pc1[i]:+.2f} sd, RMSE {rms(err[i]):.2f}")
    ax.plot(LAT, mu[:NZ] - obs[:NZ], color=INK, lw=2.2, ls="--", zorder=4,
            label=f"PPE mean: RMSE {rms(mu - obs[:F]):.2f}")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(LAT[::2])
    ax.set_xlabel("latitude band centre (°)")
    ax.set_ylabel("model − obs (mm/d)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left", title="highlighted members (RMSE over 15 bands, mm/d)",
              title_fontsize=9)
    top = ax.get_ylim()[1]
    ax.set_ylim(top=top + 0.3 * (top - ax.get_ylim()[0]))
    cb = fig.colorbar(ScalarMappable(norm, DIV), ax=ax, pad=0.015, fraction=0.035)
    cb.set_label("PC1 score (sd); thin lines: all members", fontsize=9)
    ax.set_title("Precipitation error against obs for actual ne256 PPE members, by their EOF1 (PC1) score",
                 loc="left", fontsize=11.5)
    fig.text(0.01, 0.01, f"n = {n}, annual means; obs IMERG. The highlighted members have the PC1 score closest "
             "to " + ", ".join(f"{t:+g}" for t in args.targets) + " sd.\nEach also carries some of EOFs 2–21, "
             "so its curve is not a pure EOF1 shift.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")
    for i in picked:
        print(names[i], round(pc1[i], 2), round(rms(err[i]), 3))


if __name__ == "__main__":
    main()
