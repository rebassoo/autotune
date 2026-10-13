"""
ne256-alone baseline on exactly the folds the 3-level holdout uses.

This is the "what do you get from the top resolution by itself" row of the
surrogate-performance table. It has to be scored on the SAME folds, with the
SAME scalers, by the SAME metric as evaluate_nfidelity_holdout.py, or the
comparison is meaningless -- so both the fold split and the scoring come from
that module rather than being re-implemented here.

One single-fidelity ARD-RBF GPy GP per output feature is fitted on the
training members of each fold and used to predict the held-out ones; the
predictions are pooled across folds before scoring, exactly as the 3-level
holdout does.

Usage:
    python scripts/evaluate_ne256_alone_holdout.py \
        --top-dir     .../preprocess_output_ne256_all \
        --scalers-dir .../preprocess_output_ne128_prod_v2 \
        --folds 5 --workers 5 \
        --out-dir /pscratch/sd/r/rebassoo/autotune/results_ne256_alone_holdout
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate_nfidelity_holdout import _load_level, _score  # noqa: E402


def _fold_worker(args):
    """Fit a single-fidelity GPy GP on this fold's training members."""
    ki, tr, te, X, Y = args
    from autotune_gp.gp_gpy import SingleFidelityGPyWrapper

    t0 = time.time()
    gp = SingleFidelityGPyWrapper(X[tr], Y[tr])
    gp.train()
    mean, _ = gp.predict(X[te])
    return ki, te, mean, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-dir", required=True)
    ap.add_argument("--scalers-dir", required=True,
                    help="Dir holding scalers.pkl -- use the same one as the 3-level "
                         "holdout so R^2 is computed on identically scaled targets.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit-features", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    top = _load_level(args.top_dir)
    with open(Path(args.scalers_dir) / "scalers.pkl", "rb") as f:
        saved = pickle.load(f)

    var_names = [k.replace("Y_pipeline_", "") for k in saved if k.startswith("Y_pipeline_")]
    y_scalers = [saved[f"Y_pipeline_{v}"] for v in var_names]
    x_sc = saved["X_pipeline"]

    n_top = top["X"].shape[0]
    print(f"ne256 alone: {n_top} members, {top['Y'].shape[1]} features, "
          f"{len(var_names)} variables")

    X = np.asarray(x_sc.transform(top["X"]), dtype=float)
    Y = np.stack([y_scalers[j].transform(top["Y"][:, :, j]) for j in range(len(var_names))],
                 axis=0).transpose(1, 2, 0)

    if args.limit_features:
        nf = min(args.limit_features, Y.shape[1])
        print(f"  SMOKE TEST: first {nf}/{Y.shape[1]} features only")
        Y = Y[:, :nf, :]

    from sklearn.model_selection import KFold
    k = min(args.folds, n_top)
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    jobs = [(i, tr, te, X, Y) for i, (tr, te) in enumerate(kf.split(X))]
    print(f"  {k} folds ({[len(j[2]) for j in jobs]} held out per fold), seed {args.seed}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    workers = args.workers or min(k, os.cpu_count() or k)
    pred = np.full_like(Y, np.nan)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fold_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            ki, te, mean, secs = fut.result()
            pred[te] = mean
            print(f"  [{time.strftime('%H:%M:%S')}] fold {ki + 1}/{k} done in "
                  f"{secs / 60:.1f} min", flush=True)
    if np.isnan(pred).any():
        raise SystemExit("Some held-out members were never predicted.")

    print("\n=== ne256-alone holdout skill (predictions pooled over folds) ===")
    rows = _score(Y, pred, var_names, y_scalers, label="ne256 alone (single-fidelity GPy)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ne256_alone_holdout.pkl", "wb") as f:
        pickle.dump({"rows": rows, "folds": k, "seed": args.seed, "n_top": n_top,
                     "truth": Y, "pred": pred, "var_names": var_names,
                     "top_dir": args.top_dir}, f)
    print(f"\nSaved {out_dir / 'ne256_alone_holdout.pkl'}")


if __name__ == "__main__":
    main()
