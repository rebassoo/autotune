"""
Train a single-fidelity GPy surrogate (autotune_gp.gp_gpy.SingleFidelityGPyWrapper:
one ARD-RBF GP per output feature) on one preprocess directory, with the features
trained in parallel processes. Outputs can be normalized with another level's
scalers (--y-scalers-dir) so the surrogate predicts in the same units as the
multi-fidelity ones, e.g. an ne32-only surrogate in ne128-normalized units.

    python scripts/train_sf_gpy.py \\
        --preprocess-dir .../mf_prod_v2/preprocess_output_ne32_prod_v2 \\
        --y-scalers-dir .../mf_prod_v2/preprocess_output_ne128_prod_v2 \\
        --out .../results_ne32_prod_v2_sf_gpy/sf_gp_gpy_trained.pkl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.gp_gpy import SingleFidelityGPyWrapper   # noqa: E402

D = {}


def _fit(job):
    import GPy
    vi, fi = job
    kern = GPy.kern.RBF(input_dim=D["X"].shape[1], ARD=True)
    m = GPy.models.GPRegression(D["X"], D["Y"][:, fi, vi:vi + 1], kern)
    m.optimize()
    return vi, fi, m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocess-dir", required=True)
    ap.add_argument("--y-scalers-dir", default=None,
                    help="Normalize outputs (and X) with this directory's scalers.pkl "
                         "(default: the preprocess directory's own)")
    ap.add_argument("--variables", nargs="+", default=["PCP", "OSR", "OLR", "TLWP"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L = lambda p: pickle.load(open(p, "rb"))
    pre = Path(args.preprocess_dir)
    gp_proj = L(pre / "gp_proj.pkl")
    sc = L(Path(args.y_scalers_dir or pre) / "scalers.pkl")
    X = sc["X_pipeline"].transform(gp_proj["X_train"].to_numpy(float))
    Yp = np.asarray(gp_proj["Y_train"], float)
    Y = np.stack([sc[f"Y_pipeline_{v}"].transform(Yp[:, :, j]) for j, v in enumerate(args.variables)], axis=2)
    D.update(X=X, Y=Y)
    gp = SingleFidelityGPyWrapper(X, Y)
    jobs = [(vi, fi) for vi in range(Y.shape[2]) for fi in range(Y.shape[1])]
    print(f"{len(X)} runs, {len(jobs)} GPs, {args.workers} workers", flush=True)
    models = [[None] * Y.shape[1] for _ in range(Y.shape[2])]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for n, (vi, fi, m) in enumerate(pool.imap_unordered(_fit, jobs), 1):
            models[vi][fi] = m
            if n % 8 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} trained, {(time.time() - t0) / 60:.1f} min", flush=True)
    gp._models = models
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(gp, fh)
    m, _ = gp.predict_batch(X[:50])
    r2 = 1 - np.sum((m - Y[:50]) ** 2) / np.sum((Y[:50] - Y[:50].mean(0)) ** 2)
    print(f"Saved {args.out}; in-sample R^2 on 50 training runs {r2:.3f}", flush=True)


if __name__ == "__main__":
    main()
