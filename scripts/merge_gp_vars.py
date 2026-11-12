"""
Merge N-fidelity GPs trained on different variables over the same members into
one GP, so the optimizer sees all the variables through a single surrogate.

NFidelityGPWrapper trains one independent model per (variable, feature), so a
merged GP is exactly what a joint training would give, without retraining.
The inputs must be the same: identical X at every level, the same feature
count, and the same X scaler. The merged GP is saved with the levels' Y
concatenated along the variable axis, in the order of --gp.

The script checks the merged GP against each part at a few random points
(predictions must match to the bit) and, with --scalers-dir, that the target
preprocess dir's Y scalers are the ones each part was trained with.

    python scripts/merge_gp_vars.py \\
        --gp .../results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl=PCP,OSR,OLR,TLWP@.../preprocess_output_ne128_prod_v2 \\
        --gp .../results_mf3_ne256_n85_6var/mf_gp_trained.pkl=NETTOA,TAS_LAND,SLP,U200,U850,Z500@.../preprocess_output_ne128_prod_v2_6var \\
        --scalers-dir .../preprocess_output_ne128_prod_v2_10var \\
        --out .../results_mf3_ne256_n85_10var/mf_gp_trained.pkl
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _same_scaler(a, b):
    attrs = [k for k in ("mean_", "scale_", "var_") if hasattr(a, k)]
    return bool(attrs) and all(np.allclose(getattr(a, k), getattr(b, k)) for k in attrs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp", action="append", required=True,
                    help="GP_PKL=VAR1,VAR2,...@SCALERS_DIR it was trained with")
    ap.add_argument("--scalers-dir", help="Preprocess dir the merged GP will be used with.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = []
    for spec in args.gp:
        pkl, rest = spec.split("=", 1)
        names, sc_dir = rest.split("@", 1)
        gp = pickle.load(open(pkl, "rb"))
        names = names.split(",")
        if gp.n_vars != len(names):
            raise SystemExit(f"{pkl}: has {gp.n_vars} variables, {len(names)} names given")
        sc = pickle.load(open(Path(sc_dir) / "scalers.pkl", "rb"))
        have = [k[len("Y_pipeline_"):] for k in sc if k.startswith("Y_pipeline_")]
        if have != names:
            raise SystemExit(f"{sc_dir}: scalers are for {have}, not {names}")
        parts.append((pkl, names, gp, sc))
        print(f"loaded {pkl}: {type(gp).__name__}, {gp.n_levels} levels, vars {names}", flush=True)

    ref = parts[0][2]
    for pkl, _, gp, sc in parts[1:]:
        if type(gp) is not type(ref) or gp.n_levels != ref.n_levels or gp.n_feat != ref.n_feat:
            raise SystemExit(f"{pkl}: type/levels/features differ from {parts[0][0]}")
        for k, (Xa, Xb) in enumerate(zip(ref.X_levels, gp.X_levels)):
            if Xa.shape != Xb.shape or not np.array_equal(Xa, Xb):
                raise SystemExit(f"{pkl}: level {k} X differs; not the same members")
        probe = np.random.default_rng(1).uniform(0, 1, (8, ref.n_params))
        if not np.allclose(sc["X_pipeline"].inverse_transform(probe),
                           parts[0][3]["X_pipeline"].inverse_transform(probe)):
            raise SystemExit(f"{pkl}: X scaler differs")

    names = [n for _, ns, _, _ in parts for n in ns]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate variables: {names}")
    if args.scalers_dir:
        tgt = pickle.load(open(Path(args.scalers_dir) / "scalers.pkl", "rb"))
        have = [k[len("Y_pipeline_"):] for k in tgt if k.startswith("Y_pipeline_")]
        if have != names:
            raise SystemExit(f"{args.scalers_dir}: variable order {have} != merged {names}")
        for _, ns, _, sc in parts:
            for n in ns:
                if not _same_scaler(tgt[f"Y_pipeline_{n}"], sc[f"Y_pipeline_{n}"]):
                    raise SystemExit(f"{n}: scaler in {args.scalers_dir} differs from training")
        print(f"scalers in {args.scalers_dir} match every part", flush=True)

    merged = copy.copy(ref)                       # shallow: the GPy models are shared, not copied
    merged.warm_start = None                      # not needed for prediction; avoids pickling it
    merged.Y_levels = [np.concatenate([gp.Y_levels[k] for _, _, gp, _ in parts], axis=2)
                       for k in range(ref.n_levels)]
    merged._models = [m for _, _, gp, _ in parts for m in gp._models]
    merged.n_vars = len(names)
    merged.var_names = names

    X = np.random.default_rng(0).uniform(0, 1, (4, ref.n_params))
    m, v = merged.predict_batch(X)
    j = 0
    for pkl, ns, gp, _ in parts:
        mp, vp = gp.predict_batch(X)
        if not (np.array_equal(m[..., j:j + len(ns)], mp) and np.array_equal(v[..., j:j + len(ns)], vp)):
            raise SystemExit(f"merged predictions differ from {pkl}")
        j += len(ns)
    print("merged predictions identical to each part", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(merged, f)
    print(f"Saved {args.out}: {merged.n_vars} variables {names}")


if __name__ == "__main__":
    main()
