"""
Score an already-trained surrogate on top-level members it was NOT trained on.

When new ne256 runs arrive (e.g. the near-balance set), predicting them with
the existing surrogate before retraining is a free, genuinely out-of-sample
test -- unlike k-fold holdout, the points were chosen independently of the
design the surrogate was fitted to. It reports, per variable:

  R2 (normalized, pooled over members x features), RMSE in physical units,
  mean z^2 = mean(r^2 / v) -- ~1 if the GP's predictive variance is honest,
  >1 overconfident, <1 underconfident --
  and per-member global means, predicted vs simulated.

Usage:
    python scripts/validate_on_new_members.py \\
        --gp-pkl /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_studentt_dof4_COLD/mf_gp_trained.pkl \\
        --top-dir .../mf_prod_v2/preprocess_output_ne256_n85 \\
        --scalers-dir .../mf_prod_v2/preprocess_output_ne128_prod_v2 \\
        --select _nearbal --out results/validate_cold_n72_on_nearbal.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.io import load_gp_proj   # noqa: E402

VARS = ["PCP", "OSR", "OLR", "TLWP"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp-pkl", required=True, help="Trained surrogate (mf_gp_trained.pkl).")
    ap.add_argument("--top-dir", required=True, help="Preprocess dir containing the new members.")
    ap.add_argument("--scalers-dir", required=True, help="Dir of the scalers.pkl the GP was trained with.")
    ap.add_argument("--select", default="_nearbal",
                    help="Substring of member names to score (default: _nearbal).")
    ap.add_argument("--out", default=None, help="Optional pickle of predictions and scores.")
    args = ap.parse_args()

    g = load_gp_proj(str(Path(args.top_dir) / "gp_proj.pkl"))
    X, Y = g["X_train"], np.asarray(g["Y_train"], dtype=float)
    sel = np.array([args.select in str(n) for n in X.index])
    if not sel.any():
        raise SystemExit(f"No members matching {args.select!r} in {args.top_dir}")
    names = [str(n) for n in X.index[sel]]

    with open(Path(args.scalers_dir) / "scalers.pkl", "rb") as f:
        sc = pickle.load(f)
    Xn = sc["X_pipeline"].transform(X[sel])
    Ysc = [sc[f"Y_pipeline_{v}"] for v in VARS]

    with open(args.gp_pkl, "rb") as f:
        gp = pickle.load(f)
    train_top = np.asarray(gp.X_levels[-1]) if hasattr(gp, "X_levels") else np.asarray(gp.X_high)
    dup = [n for n, x in zip(names, Xn) if np.min(np.abs(train_top - x).max(axis=1)) < 1e-9]
    if dup:
        print(f"WARNING: {len(dup)} selected member(s) are in the GP's top-level training set "
              f"(not out-of-sample): {dup}")

    M = np.zeros((len(names),) + Y.shape[1:])
    V = np.zeros_like(M)
    for i, x in enumerate(Xn):
        m, v = gp.predict(x)
        M[i], V[i] = np.asarray(m)[0], np.asarray(v)[0]
    Yn = np.stack([Ysc[j].transform(Y[sel][:, :, j]) for j in range(len(VARS))], axis=-1)
    Mp = np.stack([Ysc[j].inverse_transform(M[:, :, j]) for j in range(len(VARS))], axis=-1)
    Yp = Y[sel]

    print(f"\n{len(names)} new members, GP {Path(args.gp_pkl).parent.name}")
    print(f"  {'var':5s} {'R2(norm)':>9s} {'RMSE(phys)':>11s} {'mean z^2':>9s}")
    scores = {}
    for j, v in enumerate(VARS):
        r = (Yn[:, :, j] - M[:, :, j]).ravel()
        ss_tot = np.sum((Yn[:, :, j] - Yn[:, :, j].mean(axis=0)) ** 2)
        r2 = 1 - np.sum(r ** 2) / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean((Yp[:, :, j] - Mp[:, :, j]) ** 2)))
        z2 = float(np.mean(r ** 2 / np.maximum(V[:, :, j].ravel(), 1e-12)))
        scores[v] = {"r2_norm": float(r2), "rmse_phys": rmse, "mean_z2": z2}
        print(f"  {v:5s} {r2:9.3f} {rmse:11.4g} {z2:9.2f}")

    print("\n  global mean, simulated / predicted (+-1 sd):")
    print("  member      " + "  ".join(f"{v:>24s}" for v in VARS))
    for i, n in enumerate(names):
        cells = []
        for j, v in enumerate(VARS):
            sd = np.sqrt(V[i, -1, j]) * Ysc[j].scale_[-1]
            cells.append(f"{Yp[i, -1, j]:8.4g} / {Mp[i, -1, j]:8.4g}±{sd:<5.2g}")
        print(f"  {n.rsplit('.', 1)[-1]:10s}  " + "  ".join(f"{c:>24s}" for c in cells))
    print("  (R2 here is across the new members only; they cluster near balance, so their\n"
          "   spread is small and R2 is a harsher test than the k-fold R2.)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump({"names": names, "pred_norm": M, "var_norm": V, "obs_norm": Yn,
                         "pred_phys": Mp, "sim_phys": Yp, "scores": scores,
                         "gp_pkl": args.gp_pkl}, f)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
