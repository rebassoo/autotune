"""
Total-order Sobol indices for one block of features at a time, from a finished
sobol_sensitivity.py run (sobol.pkl + predictions.npy; no GP):

  zonal:  outputs = variance-weighted mean of the per-feature indices over the 15
          zonal bands (the weighting sobol_sensitivity.py uses over all features);
          misfit  = index of the mean squared normalized error over the 15 bands.
  global: outputs = the index of the global mean itself;
          misfit  = index of the squared normalized error of the global mean.

Rows are in the same order in both figures (the all-feature figure's order).

    python scripts/plot_sobol_blocks.py \\
        --sobol-dir /pscratch/sd/r/rebassoo/autotune/sobol_n85 \\
        --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \\
        --out-prefix docs/sobol_n85
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
from SALib.analyze import sobol as sobol_analyze

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.config import load_config                        # noqa: E402
from autotune_gp.io import load_obs, split_zrg_obs                 # noqa: E402
from autotune_gp.transforms import transform_obs                   # noqa: E402

NZ, NR = 15, 6


def heatmap(H, rows, cols, n_out, title, note, path):
    vmax = max(0.3, float(H.max()))
    fig, ax = plt.subplots(figsize=(1.0 * H.shape[1] + 4.8, 0.36 * len(rows) + 2.2))
    im = ax.imshow(H, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for (a, b), x in np.ndenumerate(H):
        ax.text(b, a, f"{x:.2f}", ha="center", va="center", fontsize=7,
                color="white" if x > 0.6 * vmax else "#1a1a1a")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.axvline(n_out - 0.5, color="#1a1a1a", lw=1)
    fig.colorbar(im, ax=ax, fraction=0.04, label="total-order Sobol index ST")
    ax.set_title(title, fontsize=10, loc="left")
    fig.text(0.01, 0.01, note, fontsize=7.5, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sobol-dir", required=True)
    ap.add_argument("--config", required=True, help="The config sobol_sensitivity.py ran with (for obs).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    sd = Path(args.sobol_dir)
    d = pickle.load(open(sd / "sobol.pkl", "rb"))
    M = np.load(sd / "predictions.npy")                          # (n, n_feat, n_vars), normalized
    var_names, pnames = list(d["variables"]), list(d["params"])
    lo, hi = d["box"]
    free = [i for i, n in enumerate(load_config(args.config).optimize.param_physical_bounds)
            if n in pnames]
    problem = {"num_vars": len(pnames), "names": pnames,
               "bounds": [[float(lo[i]), float(hi[i])] for i in free]}

    cfg = load_config(args.config)
    pre = Path(cfg.paths.preprocess_dir)
    sc_dir = Path(cfg.multi_fidelity.scalers_dir) if cfg.multi_fidelity.scalers_dir else pre
    sc = pickle.load(open(sc_dir / "scalers.pkl", "rb"))
    obs = transform_obs(split_zrg_obs(load_obs(str(pre / "obs.pkl"))["zrg_obs"], n_vars=len(var_names)),
                        [sc[f"Y_pipeline_{v}"] for v in var_names])[0]          # (n_feat, n_vars)

    ST, var = np.asarray(d["ST"]), M.var(axis=0)                  # (n_feat, n_vars, n_p), (n_feat, n_vars)
    g = NZ + NR
    zon = slice(0, NZ)
    w = var[zon] / var[zon].sum(axis=0, keepdims=True)
    ST_zon = np.einsum("fj,fjp->jp", w, ST[zon])                  # (n_vars, n_p)
    ST_glob = ST[g]                                               # (n_vars, n_p)
    sq = (M - obs[None]) ** 2
    an = lambda y: sobol_analyze.analyze(problem, y, calc_second_order=False, seed=args.seed)["ST"]
    mis_zon = np.array([an(sq[:, zon, j].mean(axis=1)) for j in range(len(var_names))])
    mis_glob = np.array([an(sq[:, g, j]) for j in range(len(var_names))])

    allH = np.maximum(np.asarray(d["ST_var"]).max(0), np.asarray(d["ST_misfit"]).max(0))
    order = np.argsort(-allH)
    rows = [pnames[p] for p in order]
    cols = var_names + [f"misfit {v}" for v in var_names]
    fixed = ", ".join(d["fixed"])
    base = (f"Surrogate mean over {len(M)} Saltelli samples; {fixed} held at the best fit. "
            "Rows in the order of the all-feature figure (sobol_total_order.png).")
    heatmap(np.hstack([ST_zon.T, mis_zon.T])[order], rows, cols, len(var_names),
            "Total-order Sobol indices, 15 zonal bands only",
            base + "\nOutputs: variance-weighted mean of the 15 bands' indices; misfit: index of the "
            "mean squared normalized error over the bands.",
            f"{args.out_prefix}_zonal.png")
    heatmap(np.hstack([ST_glob.T, mis_glob.T])[order], rows, cols, len(var_names),
            "Total-order Sobol indices, global mean only",
            base + "\nOutputs: index of the global mean; misfit: index of its squared normalized "
            "error against obs.",
            f"{args.out_prefix}_global.png")
    for lab, A, B in [("zonal", ST_zon, mis_zon), ("global", ST_glob, mis_glob)]:
        print(lab)
        for p in order:
            print(f"  {pnames[p]:38s}" + "".join(f"{x:7.3f}" for x in A[:, p]) + "  |"
                  + "".join(f"{x:7.3f}" for x in B[:, p]))


if __name__ == "__main__":
    main()
