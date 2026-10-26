"""
Is the misfit at the optimum within what the parameters can do, or beyond it?

Everything is in normalized units -- (y - PPE mean) / PPE sd per feature, the
units the cost and the model-error sigma use -- so 1 on the y axis is the PPE
spread at that feature, for every variable.

Figure 1 (<out>_features.png), one row per variable, x = zonal bands, regions, global:
  grey   : top-level PPE members (what the ensemble actually produced),
  blue   : surrogate prediction at the optimum, +-1 GP sd,
  black  : observation.
A structural error shows up as obs sitting outside the grey cloud, or the
optimum being unable to reach obs even though each band alone could.

Figure 2 (<out>_summary.png), per variable x {zonal, regional}:
  RMS(obs - optimum), RMS GP sd, and the PPE spread (= 1 by construction).
Residual >> GP sd means the surrogate's uncertainty cannot explain the misfit;
residual ~ 1 or more means the misfit is as large as the parametric spread.

    python scripts/plot_structural_error.py \\
        --calib-log /pscratch/sd/r/rebassoo/autotune/calib_n72iso_E3_zeroglobal/calibration_log.json \\
        --round 0 --gp-pkl .../results_mf3_ne256_n72iso_studentt_dof4/mf_gp_trained.pkl \\
        --top-dir .../mf_prod_v2/preprocess_output_ne256_n85_was72 \\
        --scalers-dir .../mf_prod_v2/preprocess_output_ne128_prod_v2 \\
        --out docs/structural_error_n72iso
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from autotune_gp.io import load_gp_proj, load_obs, split_zrg_obs   # noqa: E402
from autotune_gp.transforms import transform_obs                   # noqa: E402

VARS = ["PCP", "OSR", "OLR", "TLWP"]
REGIONS = ["poles", "extratrop land", "extratrop ocean", "tropical land",
           "ascending trop ocean", "descending trop ocean"]
ZONAL = [-65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75]
GREY, BLUE, INK = "#9aa0a6", "#2a6fdb", "#1a1a1a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp-pkl", required=True)
    ap.add_argument("--top-dir", required=True, help="Preprocess dir of the top level (members + obs.pkl).")
    ap.add_argument("--scalers-dir", required=True)
    ap.add_argument("--calib-log", help="calibration_log.json; theta (and sigma) taken from --round.")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--theta", help="JSON list, instead of --calib-log.")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, help="Output prefix.")
    args = ap.parse_args()

    sigma = None
    if args.calib_log:
        L = json.load(open(args.calib_log))
        rd = next(r for r in L["rounds"] if r["round"] == args.round)
        theta, sigma = np.asarray(rd["theta"]), rd.get("sigma_next")
    else:
        theta = np.asarray(json.loads(args.theta))

    with open(Path(args.scalers_dir) / "scalers.pkl", "rb") as f:
        sc = pickle.load(f)
    Ysc = [sc[f"Y_pipeline_{v}"] for v in VARS]
    obs = transform_obs(split_zrg_obs(load_obs(str(Path(args.top_dir) / "obs.pkl"))["zrg_obs"],
                                      n_vars=len(VARS)), Ysc)[0]          # (n_feat, n_vars)
    Y = np.asarray(load_gp_proj(str(Path(args.top_dir) / "gp_proj.pkl"))["Y_train"], dtype=float)
    Yn = np.stack([Ysc[j].transform(Y[:, :, j]) for j in range(len(VARS))], axis=-1)

    with open(args.gp_pkl, "rb") as f:
        gp = pickle.load(f)
    m, v = gp.predict(theta)
    m, sd = np.asarray(m)[0], np.sqrt(np.asarray(v)[0])

    nz, nr = len(ZONAL), len(REGIONS)
    labels = [f"{z}°" for z in ZONAL] + REGIONS + ["global"]
    x = np.arange(len(labels)).astype(float)
    x[nz:] += 0.7
    x[-1] += 0.7
    blocks = {"zonal": slice(0, nz), "regional": slice(nz, nz + nr)}

    # ---- figure 1: per feature --------------------------------------------
    fig, axes = plt.subplots(len(VARS), 1, figsize=(13, 11), sharex=True)
    rng = np.random.default_rng(0)
    for j, (ax, var) in enumerate(zip(axes, VARS)):
        jit = rng.uniform(-0.18, 0.18, size=Yn.shape[0])
        for k in range(len(labels)):
            ax.scatter(x[k] + jit, Yn[:, k, j], s=6, color=GREY, alpha=0.5, lw=0,
                       label=f"ne256 PPE members (n={Yn.shape[0]})" if k == 0 else None)
        ax.errorbar(x, m[:, j], yerr=sd[:, j], fmt="o", ms=5, color=BLUE, lw=2, capsize=0,
                    label="surrogate at optimum ±1 GP sd")
        ax.scatter(x, obs[:, j], marker="D", s=30, color=INK, zorder=5, label="observation")
        ax.axhline(0, color=GREY, lw=0.8, zorder=0)
        for xb in (x[nz - 1] + 0.85, x[nz + nr - 1] + 0.85):
            ax.axvline(xb, color=GREY, lw=0.6, ls=":")
        txt = []
        for b, s in blocks.items():
            r = obs[s, j] - m[s, j]
            t = f"{b}: RMS misfit {np.sqrt(np.mean(r**2)):.2f}, GP sd {np.sqrt(np.mean(sd[s, j]**2)):.2f}"
            if sigma and var in sigma and b in sigma[var]:
                t += f", σ_δ {sigma[var][b]:.2f}"
            txt.append(t)
        ax.text(0.01, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8.5,
                color=INK, bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_ylabel(f"{var}\n(PPE sd units)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False, ncol=3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    fig.suptitle(args.title or "Misfit at the optimum vs. PPE spread (normalized units)")
    fig.tight_layout()
    fig.savefig(f"{args.out}_features.png", dpi=150)

    # ---- figure 2: summary ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    w = 0.38
    for ax, (b, s) in zip(axes, blocks.items()):
        res = [np.sqrt(np.mean((obs[s, j] - m[s, j]) ** 2)) for j in range(len(VARS))]
        gsd = [np.sqrt(np.mean(sd[s, j] ** 2)) for j in range(len(VARS))]
        xi = np.arange(len(VARS))
        ax.bar(xi - w / 2 - 0.01, res, w, color=INK, label="RMS misfit at optimum")
        ax.bar(xi + w / 2 + 0.01, gsd, w, color=BLUE, label="RMS GP sd")
        ax.axhline(1, color=GREY, lw=1.5, ls="--", label="PPE spread (1 sd)")
        for i, (a, g) in enumerate(zip(res, gsd)):
            ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=8.5)
            ax.text(i + w / 2, g + 0.02, f"{g:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(xi)
        ax.set_xticklabels(VARS)
        ax.set_title(b)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("normalized units (PPE sd)")
    axes[1].legend(fontsize=8.5, frameon=False)
    fig.suptitle(args.title or "Misfit at the optimum vs. surrogate uncertainty and PPE spread")
    fig.tight_layout()
    fig.savefig(f"{args.out}_summary.png", dpi=150)
    print(f"Saved {args.out}_features.png and {args.out}_summary.png")


if __name__ == "__main__":
    main()
