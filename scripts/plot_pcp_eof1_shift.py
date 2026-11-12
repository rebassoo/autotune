"""
Why precipitation cannot be fit much better: the PPE's leading mode (EOF1) moves
every zonal band the same way, while the error against obs has opposite signs in
the tropics and the extratropics. From the ne256 PPE only (no surrogate).

(a) PPE mean + s * EOF1 for s = -2..+2 sd of PC1, against obs and the PPE 5-95% range.
(b) The same curves as error against obs (model - obs).
(c) The best zonal RMSE reachable by adding the leading K EOFs in the amounts that
    best match obs (least squares, no limit on the amounts), with the PPE mean,
    the best single member and the nominal optimum for reference.
(d) Those amounts in sd of each PC, against the largest score any PPE member has:
    beyond it the pattern is outside what the ensemble sampled.

EOFs as in plot_pcp_eofs.py: 15 bands + 6 regions in mm/d, unweighted.

    python scripts/plot_pcp_eof1_shift.py --out docs/pcp_eof1_shift_n85.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.io import load_obs, split_zrg_obs   # noqa: E402

P = Path("/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_n85")
NOMINAL = Path("/pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_studentt_dof4/diagnostics/"
               "projection_data_seed50.pkl")
LAT = np.array([-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75], float)
MMD = 86400e3
NZ, F = 15, 21
INK, MUTED, GREY, BLUE = "#1a1a1a", "#6b6f76", "#9aa0a6", "#2a6fdb"
DIV = LinearSegmentedColormap.from_list("pc", ["#2a6fdb", "#c9ccd1", "#d9542b"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L = lambda p: pickle.load(open(p, "rb"))
    Y = np.asarray(L(P / "gp_proj.pkl")["Y_train"], float)[:, :, 0] * MMD
    obs = np.asarray(split_zrg_obs(load_obs(str(P / "obs.pkl"))["zrg_obs"], n_vars=4)[0],
                     float).ravel() * MMD
    n = len(Y)
    mu = Y[:, :F].mean(0)
    U, S, Vt = np.linalg.svd(Y[:, :F] - mu, full_matrices=False)
    for k in range(len(S)):
        if Vt[k, :NZ].mean() < 0:
            Vt[k], U[:, k] = -Vt[k], -U[:, k]
    sdk = S / np.sqrt(n)                       # mm/d-norm of 1 sd of each PC
    e1 = Vt[0] * sdk[0]                        # EOF1 for +1 sd, mm/d
    d = obs[:F] - mu
    coef = Vt @ d
    need = coef / sdk                          # PC score (sd) that best matches obs
    pc_max = np.abs(U * np.sqrt(n)).max(0)
    rms = lambda e: float(np.sqrt(np.mean(np.asarray(e)[:NZ] ** 2)))
    lo, hi = np.percentile(Y, [5, 95], axis=0)
    j = list(L(NOMINAL)["var_names"]).index("PCP")
    nominal = np.asarray(L(NOMINAL)["hf_optimal"], float)[:, j] * MMD

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.2, left=0.06, right=0.98, top=0.9, bottom=0.08)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    norm = Normalize(-2, 2)
    ax_a.fill_between(LAT, lo[:NZ], hi[:NZ], color=GREY, alpha=0.2, lw=0, label="PPE 5–95%")
    for s in [-2, -1, 0, 1, 2]:
        y = mu + s * e1
        kw = dict(color=DIV(norm(s)), lw=2.4 if s == 0 else 1.6)
        lab = "PPE mean" if s == 0 else f"PPE mean {s:+d} sd of EOF1"
        ax_a.plot(LAT, y[:NZ], label=lab, **kw)
        ax_b.plot(LAT, (y - obs[:F])[:NZ], label=("PPE mean" if s == 0 else f"PPE mean {s:+d} sd of EOF1") + " − obs", **kw)
    s_best = need[0]
    yb = mu + s_best * e1
    ax_b.plot(LAT, (yb - obs[:F])[:NZ], color=INK, lw=1.6, ls="--",
              label=f"best EOF1 shift ({s_best:+.1f} sd): RMSE {rms(yb - obs[:F]):.2f}")
    ax_a.plot(LAT, obs[:NZ], color=INK, lw=2.4, marker="D", ms=5, label="obs (IMERG)")
    ax_b.axhline(0, color=INK, lw=0.9)
    ax_b.fill_between(LAT, -0.05, 0.05, color=INK, alpha=0.06, lw=0)
    for a in (ax_a, ax_b):
        a.set_xticks(LAT[::2])
        a.set_xlabel("latitude band centre (°)")
        a.spines[["top", "right"]].set_visible(False)
    ax_a.set_ylabel("precipitation (mm/d)")
    ax_b.set_ylabel("model − obs (mm/d)")
    ax_a.set_title("(a) EOF1 raises or lowers every band at once", loc="left", fontsize=11.5)
    ax_b.set_title("(b) The error moves up and down but keeps its shape", loc="left", fontsize=11.5)
    ax_a.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax_b.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax_b.set_ylim(top=1.9)
    ax_b.text(0.99, 0.03, "too wet in the tropics, too dry in the\nmid-latitudes: no single shift fixes both",
              transform=ax_b.transAxes, ha="right", fontsize=9.5, color=MUTED)

    Ks = [0, 1, 2, 3, 4, 6, 10, 21]
    vals = [rms(mu + coef[:K] @ Vt[:K] - obs[:F]) if K else rms(mu - obs[:F]) for K in Ks]
    xs = np.arange(len(Ks))
    ax_c.bar(xs, vals, color=BLUE, alpha=0.8, width=0.6)
    for x, v in zip(xs, vals):
        ax_c.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    best_member = min(rms(Y[i, :F] - obs[:F]) for i in range(n))
    for v, lab, ls in [(rms(nominal - obs[:len(nominal)]), "nominal optimum (surrogate)", "-"),
                       (best_member, "best single PPE member", "--")]:
        ax_c.axhline(v, color=INK, lw=1, ls=ls, label=f"{lab}: {v:.2f}")
    ax_c.legend(fontsize=9, frameon=True, framealpha=1, edgecolor="white", loc="upper right")
    ax_c.set_ylim(0, 0.42)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(["PPE\nmean"] + [f"EOFs\n1–{K}" if K > 1 else "EOF1" for K in Ks[1:]])
    ax_c.set_ylabel("zonal RMSE vs obs (mm/d)")
    ax_c.set_title("(c) Best zonal RMSE when the leading EOFs are added in any amount",
                   loc="left", fontsize=11.5)
    ax_c.spines[["top", "right"]].set_visible(False)

    k = np.arange(1, 13)
    col = [BLUE if abs(need[i]) <= pc_max[i] else "#d9542b" for i in range(12)]
    ax_d.bar(k, need[:12], color=col, width=0.6)
    ax_d.plot(k, pc_max[:12], color=INK, ls="", marker="_", ms=16, mew=2, label="largest score in the PPE (±)")
    ax_d.plot(k, -pc_max[:12], color=INK, ls="", marker="_", ms=16, mew=2)
    ax_d.axhline(0, color=INK, lw=0.8)
    var = S ** 2 / np.sum(S ** 2)
    ax_d.set_xticks(k)
    ax_d.set_xticklabels([f"{i}\n{100 * var[i - 1]:.0f}%" for i in k], fontsize=9)
    ax_d.set_xlabel("EOF (and its share of PPE variance)")
    ax_d.set_ylim(-6, 6)
    ax_d.set_ylabel("score needed to match obs (sd of the PC)")
    ax_d.set_title("(d) Amount of each EOF needed; red = beyond any PPE member", loc="left", fontsize=11.5)
    ax_d.legend(fontsize=9, frameon=False, loc="lower left")
    ax_d.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Why precipitation can't be fit much better: the parameters mostly move all bands together",
                 x=0.01, ha="left", fontsize=14, weight="bold", color=INK)
    fig.text(0.01, 0.935, f"ne256 PPE (n = {n}), annual means. EOFs over 15 bands + 6 regions; RMSE over "
             "the 15 bands. (c, d) treat the PPE as linear in its EOFs, so they show what is reachable in "
             "principle, not what any parameter set gives.", fontsize=9, color=MUTED)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved {args.out}")
    print("best shift", round(s_best, 2), "needed", np.round(need[:12], 2), "max", np.round(pc_max[:12], 2))


if __name__ == "__main__":
    main()
