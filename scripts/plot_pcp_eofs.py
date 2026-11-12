"""
What the PPE's precipitation EOFs look like, from the ne256 runs only (no surrogate).

(a) Each member's zonal-mean anomaly, coloured by its PC1 score: EOF1 moves the whole
    profile up or down together.
(b) The same members with EOF1 removed, same axis: what is left.
(c-h) EOFs 1-6, +-1 sd of the PC, over 15 bands and 6 regions, on one y axis, with the
    part of the obs anomaly (obs - PPE mean) that lies along each EOF, each EOF's share
    of the PPE variance and of the obs anomaly, and the parameters whose values
    correlate most with its PC across the members.

    python scripts/plot_pcp_eofs.py \\
        --top-dir .../mf_prod_v2/preprocess_output_ne256_n85 \\
        --scalers-dir .../mf_prod_v2/preprocess_output_ne128_prod_v2 \\
        --out docs/pcp_eofs_n85.png
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

LAT = np.array([-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75], float)
REGIONS = ["poles", "xtrop land", "xtrop ocean", "trop land", "asc. trop ocean", "desc. trop ocean"]
NZ, NR = 15, 6
MMD = 86400e3
INK, MUTED, GREY, BLUE = "#1a1a1a", "#6b6f76", "#9aa0a6", "#2a6fdb"
DIV = LinearSegmentedColormap.from_list("pc", ["#2a6fdb", "#c9ccd1", "#d9542b"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-dir", required=True)
    ap.add_argument("--scalers-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = lambda p: pickle.load(open(p, "rb"))
    gp = L(Path(args.top_dir) / "gp_proj.pkl")
    Y = np.asarray(gp["Y_train"], float)[:, :, 0] * MMD
    Xdf = gp["X_train"]
    X = L(Path(args.scalers_dir) / "scalers.pkl")["X_pipeline"].transform(Xdf.to_numpy(float))
    names = list(Xdf.columns)
    obs = np.asarray(split_zrg_obs(load_obs(str(Path(args.top_dir) / "obs.pkl"))["zrg_obs"], n_vars=4)[0],
                     float).ravel() * MMD

    f = NZ + NR
    n = len(Y)
    A = Y[:, :f] - Y[:, :f].mean(0)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    for k in range(len(S)):                                     # zonal mean positive
        if Vt[k, :NZ].mean() < 0:
            Vt[k], U[:, k] = -Vt[k], -U[:, k]
    var = S ** 2 / np.sum(S ** 2)
    d = obs[:f] - Y[:, :f].mean(0)
    coef = Vt @ d
    obs_share = coef ** 2 / np.sum(d ** 2)
    pc = U * np.sqrt(n)                                         # unit variance
    res1 = A - np.outer(U[:, 0] * S[0], Vt[0])

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.42, wspace=0.13,
                          left=0.055, right=0.985, top=0.915, bottom=0.165)
    # ---- members ------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0:1])
    ax_b = fig.add_subplot(gs[0, 1:2], sharey=ax_a)
    norm = Normalize(-2.5, 2.5)
    for k in np.argsort(np.abs(pc[:, 0])):
        c = DIV(norm(pc[k, 0]))
        ax_a.plot(LAT, A[k, :NZ], color=c, lw=1.0, alpha=0.85)
        ax_b.plot(LAT, res1[k, :NZ], color=c, lw=1.0, alpha=0.85)
    ax_a.set_title("(a) PPE members minus the mean, coloured by PC1", loc="left", fontsize=11.5)
    ax_b.set_title("(b) The same members with EOF1 removed", loc="left", fontsize=11.5)
    ax_a.set_ylabel("precip anomaly (mm/d)")
    for a in (ax_a, ax_b):
        a.axhline(0, color=INK, lw=0.8)
        a.set_xticks(LAT[::2])
        a.set_xlabel("latitude band centre (°)")
        a.spines[["top", "right"]].set_visible(False)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    cax = fig.add_axes([0.685, 0.72, 0.012, 0.19])
    cb = fig.colorbar(ScalarMappable(norm, DIV), cax=cax)
    cb.set_label("PC1 score (sd)", fontsize=9.5)
    frac = 1 - np.sum(res1[:, :NZ] ** 2, axis=0) / np.sum(A[:, :NZ] ** 2, axis=0)
    same = int(np.sum(np.sign(Vt[0]) == 1))
    fig.text(0.765, 0.905,
             f"EOF1 carries {100 * var[0]:.0f}% of the PPE variance.\n\n"
             f"Its weights have the same sign in\n{same} of the {f} features, so a member\n"
             "with a high PC1 is wetter almost\neverywhere and a low one drier.\n\n"
             f"Share of each band's variance it\nexplains: median {100 * np.median(frac):.0f}%, "
             f"range {100 * frac.min():.0f}–{100 * frac.max():.0f}%.\n\n"
             "Removing it leaves much smaller\nanomalies with no common sign.",
             fontsize=10.5, color=INK, va="top", linespacing=1.35)

    # ---- EOF patterns ---------------------------------------------------------
    xr = np.arange(NR) * 10.0 + 95
    xall = np.r_[LAT, xr]
    ylim = 1.15 * max(np.max(np.abs(Vt[:6] * S[:6, None] / np.sqrt(n))),
                      np.max(np.abs(coef[:6, None] * Vt[:6])))
    letters = "cdefgh"
    axes = []
    for k in range(6):
        a = fig.add_subplot(gs[1 + k // 3, k % 3], sharey=axes[0] if axes else None)
        axes.append(a)
        e = Vt[k] * S[k] / np.sqrt(n)
        part = coef[k] * Vt[k]
        a.axvspan(85, xr[-1] + 7, color="#f6f7f8", lw=0, zorder=0)
        a.axhline(0, color=INK, lw=0.8)
        a.fill_between(LAT, -e[:NZ], e[:NZ], color=BLUE, alpha=0.2, lw=0)
        a.plot(LAT, e[:NZ], color=BLUE, lw=2, marker="o", ms=3.5, label="EOF, +1 sd of its PC (shaded: ±1 sd)")
        a.vlines(xr, -np.abs(e[NZ:]), np.abs(e[NZ:]), color=BLUE, alpha=0.2, lw=7)
        a.plot(xr, e[NZ:], color=BLUE, ls="", marker="o", ms=5)
        a.plot(xall, part, color=INK, ls="", marker="D", ms=4, zorder=4, label="obs anomaly along this EOF")
        r = [np.corrcoef(pc[:, k], X[:, j])[0, 1] for j in range(X.shape[1])]
        top = np.argsort(np.abs(r))[::-1][:2]
        a.set_title(f"({letters[k]}) EOF{k + 1}: {100 * var[k]:.0f}% of PPE variance, "
                    f"{100 * obs_share[k]:.0f}% of obs anomaly", loc="left", fontsize=10.8)
        a.text(0.015, 0.03, "PC correlates with: " + ", ".join(f"{names[j].replace('_factor', '')} {r[j]:+.2f}" for j in top),
               transform=a.transAxes, fontsize=8.1, color=MUTED)
        a.set_ylim(-ylim, ylim)
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(-70, xr[-1] + 7)
        if k >= 3:
            a.set_xticks(list(LAT[::2]) + list(xr))
            a.set_xticklabels([f"{v:.0f}" for v in LAT[::2]] + REGIONS, fontsize=8)
            for t in a.get_xticklabels()[len(LAT[::2]):]:
                t.set_rotation(55)
                t.set_ha("right")
                t.set_rotation_mode("anchor")
            a.set_xlabel("latitude band centre (°), then regions", fontsize=9.5, labelpad=2)
        else:
            a.set_xticks(list(LAT[::2]) + list(xr))
            a.set_xticklabels([])
        if k % 3 == 0:
            a.set_ylabel("mm/d")
        else:
            plt.setp(a.get_yticklabels(), visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.05, 0.035), ncol=2, frameon=False, fontsize=10)

    fig.suptitle("Precipitation EOFs of the ne256 PPE: one mode raises or lowers nearly everything, the rest are "
                 "small patterns", x=0.012, y=0.975, ha="left", fontsize=14.5, weight="bold", color=INK)
    fig.text(0.058, 0.008, f"n = {n}, annual means, EOFs over 15 bands + 6 regions in mm/d, unweighted. An EOF's sign is "
             "arbitrary; each is drawn with a positive zonal mean.\nCorrelations are across the members, one parameter at a "
             "time, and do not separate parameters that vary together.",
             ha="left", fontsize=8.6, color=MUTED, linespacing=1.4)
    fig.savefig(args.out, dpi=150, facecolor="white")

    for k in range(6):
        r = np.array([np.corrcoef(pc[:, k], X[:, j])[0, 1] for j in range(X.shape[1])])
        o = np.argsort(np.abs(r))[::-1][:4]
        b = np.linalg.lstsq(np.c_[np.ones(n), X], pc[:, k], rcond=None)[0]
        fit = np.c_[np.ones(n), X] @ b
        r2 = 1 - np.sum((pc[:, k] - fit) ** 2) / np.sum((pc[:, k] - pc[:, k].mean()) ** 2)
        print(f"EOF{k + 1} var {100 * var[k]:.1f}% obs {100 * obs_share[k]:.1f}% same-sign "
              f"{int(np.sum(np.sign(Vt[k]) == 1))}/{f} linear-R2(params) {r2:.2f} peak "
              f"{([*map(str, LAT.astype(int))] + REGIONS)[int(np.argmax(np.abs(Vt[k])))]}; "
              + ", ".join(f"{names[j]} {r[j]:+.2f}" for j in o))
        print("    mm/d:", np.round(Vt[k] * S[k] / np.sqrt(n), 2))
    print(f"EOF7+ var {100 * var[6:].sum():.1f}% obs {100 * obs_share[6:].sum():.1f}%; "
          f"band variance explained by EOF1 {np.round(100 * frac, 0)}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
