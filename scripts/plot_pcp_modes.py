"""
Can the parameters change precipitation in some latitudes and not others, or only
raise/lower it everywhere?

(a) PPE members' zonal-mean precip minus the ensemble mean (grey), the leading
    EOF of the PPE (+-1 sd of its PC, blue) and what obs need (obs - PPE mean).
(b) Per EOF k: share of the PPE variance it carries vs share of the obs anomaly
    (obs - PPE mean) that lies along it. A volume knob is one EOF with most of the
    variance; obs needing the low-variance EOFs is a pattern the PPE cannot make.
(c) At the optimum: the change still needed (obs - surrogate), and how much of it
    the parameters can supply: with --pcponly-dir, the change the PCP-only
    optimization actually reaches (surrogate, nonlinear, inside the search box);
    and the first-order estimate (bounded least squares on the surrogate Jacobian,
    every parameter kept in [0, 1]). The unbounded first-order fit is reported too.

EOFs use the zonal bands and regions (global excluded, it is their mean);
everything is in mm/d.

    python scripts/plot_pcp_modes.py \\
        --top-dir .../mf_prod_v2/preprocess_output_ne256_n85 \\
        --scalers-dir .../mf_prod_v2/preprocess_output_ne128_prod_v2 \\
        --gp-pkl .../results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl \\
        --results-dir .../results_mf3_ne256_n85_studentt_dof4 \\
        --pcponly-dir .../results_mf3_ne256_n85_pcponly \\
        --out docs/pcp_modes_n85.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.io import load_obs, split_zrg_obs   # noqa: E402

ZONAL = [-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75]
NZ, NR = 15, 6
MMD = 86400e3                      # m/s of water -> mm/d
INK, GREY, BLUE, ORANGE = "#1a1a1a", "#9aa0a6", "#2a6fdb", "#e8833a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-dir", required=True)
    ap.add_argument("--scalers-dir", required=True)
    ap.add_argument("--gp-pkl", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pcponly-dir", help="Results of the optimization with PCP weight 1, others 0.")
    ap.add_argument("--step", type=float, default=0.02, help="Finite-difference step (normalized x).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = lambda p: pickle.load(open(p, "rb"))
    Y = np.asarray(L(Path(args.top_dir) / "gp_proj.pkl")["Y_train"], float)[:, :, 0] * MMD
    obs = np.asarray(split_zrg_obs(load_obs(str(Path(args.top_dir) / "obs.pkl"))["zrg_obs"],
                                   n_vars=4)[0], float).ravel() * MMD
    f = slice(0, NZ + NR)                                   # zonal + regional
    mean = Y.mean(0)
    A = Y[:, f] - mean[f]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    var_frac = S ** 2 / np.sum(S ** 2)
    eof1 = Vt[0] * S[0] / np.sqrt(len(Y))                    # mm/d per 1 sd of PC1
    if eof1[:NZ].mean() < 0:
        eof1 = -eof1
    need_mean = obs[f] - mean[f]
    obs_frac = (Vt @ need_mean) ** 2 / np.sum(need_mean ** 2)
    corr_g = np.array([np.corrcoef(Y[:, k], Y[:, -1])[0, 1] for k in range(NZ + NR)])

    # surrogate Jacobian at the optimum
    sc = L(Path(args.scalers_dir) / "scalers.pkl")["Y_pipeline_PCP"]
    x0 = np.asarray(L(Path(args.results_dir) / "diagnostics" / "projection_data_seed50.pkl")["best_params_norm"])
    gp = L(args.gp_pkl)
    to_mmd = lambda m: sc.inverse_transform(m[None, :])[0] * MMD
    pred = lambda x: to_mmd(np.asarray(gp.predict(x)[0])[0][:, 0])
    y0 = pred(x0)
    J = np.zeros((NZ + NR, len(x0)))
    for i in range(len(x0)):
        h = args.step if x0[i] + args.step <= 1 else -args.step
        x = x0.copy()
        x[i] += h
        J[:, i] = (pred(x)[f] - y0[f]) / h
    need = obs[f] - y0[f]
    # unbounded: how far the parameters would have to move to first order;
    # bounded: the best change that keeps every parameter inside [0, 1]
    dx_free, *_ = np.linalg.lstsq(J, need, rcond=None)
    free_frac = 1 - np.sum((need - J @ dx_free) ** 2) / np.sum(need ** 2)
    from scipy.optimize import lsq_linear
    dx = lsq_linear(J, need, bounds=(-x0, 1 - x0)).x
    reach = J @ dx
    reach_frac = 1 - np.sum((need - reach) ** 2) / np.sum(need ** 2)
    nl, nl_frac = None, None
    if args.pcponly_dir:
        xp = np.asarray(L(Path(args.pcponly_dir) / "diagnostics" / "projection_data_seed50.pkl")["best_params_norm"])
        nl = pred(xp)[f] - y0[f]
        nl_frac = 1 - np.sum((need - nl) ** 2) / np.sum(need ** 2)
    sJ = np.linalg.svd(J, compute_uv=False)
    jac_frac = sJ ** 2 / np.sum(sJ ** 2)

    lat = np.array(ZONAL, float)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(width_ratios=[1.25, 0.9, 1.25]))
    a = ax[0]
    for k in range(len(Y)):
        a.plot(lat, A[k, :NZ], color=GREY, lw=0.6, alpha=0.45,
               label=f"ne256 PPE members − mean (n={len(Y)})" if k == 0 else None)
    a.fill_between(lat, -eof1[:NZ], eof1[:NZ], color=BLUE, alpha=0.18, lw=0)
    a.plot(lat, eof1[:NZ], color=BLUE, lw=2, label=f"EOF1, ±1 sd ({100 * var_frac[0]:.0f}% of PPE variance)")
    a.plot(lat, -eof1[:NZ], color=BLUE, lw=2)
    a.plot(lat, need_mean[:NZ], color=INK, lw=2.2, marker="D", ms=5, label="obs − PPE mean")
    a.axhline(0, color=GREY, lw=0.8)
    a.set_xlabel("latitude band (°)")
    a.set_ylabel("precip anomaly (mm/d)")
    a.set_title("(a) Members shift up or down together", loc="left", fontsize=11)
    a.legend(fontsize=8.5, frameon=False, loc="upper left")

    a = ax[1]
    k = np.arange(1, 7)
    w = 0.38
    a.bar(k - w / 2 - 0.01, 100 * var_frac[:6], w, color=BLUE, label="share of PPE variance")
    a.bar(k + w / 2 + 0.01, 100 * obs_frac[:6], w, color=INK, label="share of (obs − PPE mean)")
    for i in range(6):
        a.text(k[i] - w / 2, 100 * var_frac[i] + 1, f"{100 * var_frac[i]:.0f}", ha="center", fontsize=8)
        a.text(k[i] + w / 2, 100 * obs_frac[i] + 1, f"{100 * obs_frac[i]:.0f}", ha="center", fontsize=8)
    a.text(6.4, 100 * max(var_frac[0], obs_frac[:6].max()) * 0.9,
           f"EOFs 7–21: {100 * var_frac[6:].sum():.0f}% of PPE variance,\n"
           f"{100 * obs_frac[6:].sum():.0f}% of what obs need",
           ha="right", va="top", fontsize=8.5, color=INK)
    a.set_xticks(k)
    a.set_xticklabels([f"EOF{i}" for i in k])
    a.set_ylabel("%")
    a.set_title("(b) The obs anomaly lies off the main mode", loc="left", fontsize=11)
    a.legend(fontsize=8.5, frameon=False, loc="center right")

    a = ax[2]
    a.plot(lat, need[:NZ], color=INK, lw=2.2, marker="D", ms=5, label="still needed: obs − optimum")
    if nl is not None:
        a.plot(lat, nl[:NZ], color=ORANGE, lw=2.2, marker="o", ms=5,
               label=f"reached by the PCP-only optimum: {100 * nl_frac:.0f}% of it")
    a.plot(lat, reach[:NZ], color=ORANGE, lw=1.4, ls="--", alpha=0.8,
           label=f"first-order estimate, inside the box: {100 * reach_frac:.0f}%")
    a.axhline(0, color=GREY, lw=0.8)
    a.set_xlabel("latitude band (°)")
    a.set_ylabel("precip change (mm/d)")
    a.set_title(f"(c) Fitting PCP alone removes {100 * nl_frac:.0f}% of what is left" if nl is not None
                else "(c) How much of what is left the parameters can supply", loc="left", fontsize=11)
    a.legend(fontsize=8.5, frameon=False, loc="lower left")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Precipitation: the parameters move the whole profile, not individual latitudes",
                 fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, 0.008, "EOFs over 15 zonal bands + 6 regions of the ne256 PPE. (c): 3-level MF surrogate at the "
             "19-parameter optimum; % = share of the squared misfit (bands + regions) removed. First order: finite-"
             "difference Jacobian, least squares\nwith every parameter in [0, 1] (RMS move "
             f"{np.sqrt(np.mean(dx ** 2)):.2f} box widths). Unbounded it would reach {100 * free_frac:.0f}%, but only by "
             f"moving the parameters {np.sqrt(np.mean(dx_free ** 2)):.1f} box widths (RMS).",
             fontsize=8, color=GREY, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(args.out, dpi=150)

    print(f"PPE EOF variance %: {np.round(100 * var_frac[:6], 1)}")
    print(f"obs-PPE mean along EOFs %: {np.round(100 * obs_frac[:6], 1)}; EOF1 same sign in "
          f"{int(np.sum(np.sign(eof1) == np.sign(eof1[:NZ].mean())))}/{NZ + NR} features")
    print(f"corr(feature, global PCP) across members: median {np.median(corr_g):.2f}, min {corr_g.min():.2f}")
    print(f"Jacobian singular-value variance %: {np.round(100 * jac_frac[:6], 1)}")
    print(f"needed RMS {np.sqrt(np.mean(need ** 2)):.3f} mm/d; reachable in box {100 * reach_frac:.1f}% "
          f"(RMS dx {np.sqrt(np.mean(dx ** 2)):.2f}); unbounded {100 * free_frac:.1f}% "
          f"(RMS dx {np.sqrt(np.mean(dx_free ** 2)):.2f})"
          + (f"; PCP-only optimum removes {100 * nl_frac:.1f}%" if nl_frac is not None else ""))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
