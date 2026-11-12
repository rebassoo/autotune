"""
Copy a preprocess_output directory keeping only some of its variables, so a
surrogate can be trained on just those (e.g. the six observables added to the
10-variable preprocessing, without retraining the four already fitted).

Writes the files stage 2 reads -- gp_proj.pkl, scalers.pkl, obs.pkl,
column_mask.pkl, default_run_zrg.pkl (if present) -- with the variable axis
subset in the requested order. Variable names and their order in the source
come from the Y_pipeline_<var> entries of its scalers.pkl.

    python scripts/subset_preprocess_vars.py \\
        --src .../mf_prod_v2/preprocess_output_ne128_prod_v2_10var \\
        --dst .../mf_prod_v2/preprocess_output_ne128_prod_v2_6var \\
        --vars NETTOA TAS_LAND SLP U200 U850 Z500
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--vars", nargs="+", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        raise SystemExit("--dst must differ from --src")

    load = lambda n: pickle.load(open(src / n, "rb"))
    scalers = load("scalers.pkl")
    all_vars = [k[len("Y_pipeline_"):] for k in scalers if k.startswith("Y_pipeline_")]
    missing = [v for v in args.vars if v not in all_vars]
    if missing:
        raise SystemExit(f"{missing} not in {src.name} (has {all_vars})")
    idx = [all_vars.index(v) for v in args.vars]

    gp = load("gp_proj.pkl")
    Y = np.asarray(gp["Y_train"])
    if Y.shape[-1] != len(all_vars):
        raise SystemExit(f"gp_proj Y_train has {Y.shape[-1]} variables, scalers {len(all_vars)}")
    n_feat = Y.shape[1]

    obs = load("obs.pkl")
    z = obs["zrg_obs"]
    if z.shape[1] != n_feat * len(all_vars):
        raise SystemExit(f"obs has {z.shape[1]} columns, expected {n_feat} x {len(all_vars)}")
    cols = [c for i in idx for c in range(i * n_feat, (i + 1) * n_feat)]

    mask = load("column_mask.pkl")
    if mask["n_vars"] != len(all_vars):
        raise SystemExit(f"column_mask n_vars={mask['n_vars']}, expected {len(all_vars)}")

    dst.mkdir(parents=True, exist_ok=True)
    dump = lambda obj, n: pickle.dump(obj, open(dst / n, "wb"))
    dump({**gp, "Y_train": Y[..., idx]}, "gp_proj.pkl")
    dump({"X_pipeline": scalers["X_pipeline"],
          **{f"Y_pipeline_{v}": scalers[f"Y_pipeline_{v}"] for v in args.vars}}, "scalers.pkl")
    dump({**obs, "zrg_obs": z.iloc[:, cols]}, "obs.pkl")
    dump({**mask, "n_vars": len(args.vars)}, "column_mask.pkl")
    if (src / "default_run_zrg.pkl").exists():
        d = load("default_run_zrg.pkl")
        dump({**d, "Y_default_ZRG": np.asarray(d["Y_default_ZRG"])[..., idx]}, "default_run_zrg.pkl")
    (dst / "SOURCE.txt").write_text(f"Subset of {src} to variables {args.vars} "
                                    f"by scripts/subset_preprocess_vars.py\n")
    print(f"{dst}: Y {Y[..., idx].shape}, obs {z.iloc[:, cols].shape}, vars {args.vars}")


if __name__ == "__main__":
    main()
