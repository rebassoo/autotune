"""
Build the basis and error covariance for likelihood: pca_gaussian
(autotune_gp.pca_likelihood) from a surrogate holdout file
(nfidelity_holdout.pkl: Y_true, Y_pred, (n_members, n_feat, n_vars), normalized):
EOFs of Y_true, error covariance of Y_pred - Y_true.

--exclude drops a member from both, e.g. the target of a twin experiment, so the
likelihood is built without it. Prints the leave-one-out chi^2 check: each
member's own holdout error against a covariance built without it; the mean
should be close to K.

    python scripts/build_pca_likelihood.py \\
        --holdout .../results_ne256_n85_3level_holdout_ISO/nfidelity_holdout.pkl \\
        --members .../preprocess_output_ne256_n85/gp_proj.pkl \\
        --exclude PPEensemble_...m202 --k 20 --out .../pca_basis.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from autotune_gp.pca_likelihood import build_basis, pca_cost   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--members", required=True,
                    help="gp_proj.pkl whose X_train index names the holdout rows, in order")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = lambda p: pickle.load(open(p, "rb"))
    h = L(args.holdout)
    Yt, Yp = np.asarray(h["Y_true"], float), np.asarray(h["Y_pred"], float)
    gp = L(args.members)
    names = list(gp["X_train"].index)
    if len(names) != len(Yt):
        raise SystemExit(f"{len(names)} members in {args.members}, {len(Yt)} holdout rows")
    # the holdout rows must be the gp_proj members in order (checked on the features)
    ref = np.asarray(gp["Y_train"], float)
    if not np.allclose(np.corrcoef(ref.reshape(len(ref), -1)[:, 0], Yt.reshape(len(Yt), -1)[:, 0])[0, 1], 1):
        raise SystemExit("holdout rows are not in gp_proj member order")
    keep = np.array([n not in args.exclude for n in names])
    if len(args.exclude) != int((~keep).sum()):
        raise SystemExit(f"--exclude: found {int((~keep).sum())} of {len(args.exclude)}")
    shape = Yt.shape[1:]
    Y = Yt[keep].reshape(int(keep.sum()), -1)
    E = (Yp - Yt)[keep].reshape(int(keep.sum()), -1)

    b = build_basis(Y, E, args.k)
    loo = np.array([2 * pca_cost(E[i], build_basis(Y, np.delete(E, i, 0), args.k)) for i in range(len(E))])
    from scipy.stats import chi2
    print(f"K={args.k}: {100 * b['ppe_var_frac']:.1f}% of PPE variance; leave-one-out chi^2 mean "
          f"{loo.mean():.1f}, median {np.median(loo):.1f}, above the 95th percentile "
          f"{100 * np.mean(loo > chi2.ppf(0.95, args.k)):.0f}% (expect {args.k}, {args.k - 0.7:.0f}, 5%)")
    b.update(feature_shape=shape, var_names=list(h["var_names"]), k=args.k, excluded=args.exclude,
             holdout=args.holdout, loo_chi2=loo)
    with open(args.out, "wb") as fh:
        pickle.dump(b, fh)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
