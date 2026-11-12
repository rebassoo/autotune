"""
Evaluate several optima with several surrogates: each surrogate predicts every
optimum, and the misfit to the observations is reported per variable and block
in one common normalized space, so surrogates trained with different scalers
(e.g. the 3-level MF surrogate, ne128 scalers, and a ne256-alone GP, ne256
scalers) can be compared.

A surrogate is NAME=GP_PICKLE@SCALERS_DIR (the scalers it was trained in); an
optimum is NAME=RESULTS_DIR (best point from diagnostics/projection_data_seed50.pkl).
All optima must be in the same normalized parameter space; the script checks
that the X scalers agree.

    python scripts/cross_evaluate_optima.py \\
        --surrogate "MF=.../results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl@.../preprocess_output_ne128_prod_v2" \\
        --surrogate "ne256=.../results_ne256_n85_sf_gpy_studentt_dof4/sf_gp_gpy_trained.pkl@.../preprocess_output_ne256_n85" \\
        --optimum "MF=.../results_mf3_ne256_n85_studentt_dof4" \\
        --optimum "ne256=.../results_ne256_n85_sf_gpy_studentt_dof4" \\
        --obs-dir .../preprocess_output_ne256_n85 --norm-dir .../preprocess_output_ne128_prod_v2 \\
        --out docs/cross_eval_mf3_vs_ne256alone_n85.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.io import load_obs, split_zrg_obs                 # noqa: E402

VARS = ["PCP", "OSR", "OLR", "TLWP"]


def _scalers(d):
    return pickle.load(open(Path(d) / "scalers.pkl", "rb"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surrogate", action="append", required=True, help="NAME=GP_PKL@SCALERS_DIR")
    ap.add_argument("--optimum", action="append", required=True, help="NAME=RESULTS_DIR")
    ap.add_argument("--obs-dir", required=True, help="Preprocess dir whose obs.pkl is the target.")
    ap.add_argument("--norm-dir", required=True, help="Scalers that define the common normalized space.")
    ap.add_argument("--n-zonal", type=int, default=15)
    ap.add_argument("--n-regions", type=int, default=6)
    ap.add_argument("--out", required=True, help="CSV of RMS misfit per surrogate x optimum x variable x block.")
    args = ap.parse_args()

    norm = _scalers(args.norm_dir)
    obs_phys = np.stack([np.asarray(p, float).ravel() for p in
                         split_zrg_obs(load_obs(str(Path(args.obs_dir) / "obs.pkl"))["zrg_obs"],
                                       n_vars=len(VARS))], 1)          # (n_feat, n_vars)
    to_norm = lambda Y: np.stack([norm[f"Y_pipeline_{v}"].transform(Y[None, :, j])[0]
                                  for j, v in enumerate(VARS)], 1)
    obs = to_norm(obs_phys)

    optima = {}
    for o in args.optimum:
        name, d = o.split("=", 1)
        optima[name] = np.asarray(pickle.load(open(Path(d) / "diagnostics" /
                                                   "projection_data_seed50.pkl", "rb"))["best_params_norm"])

    z, r = args.n_zonal, args.n_regions
    blocks = {"zonal": slice(0, z), "regional": slice(z, z + r), "global": slice(z + r, z + r + 1)}
    rows, preds = [], {}
    probe = np.random.default_rng(0).uniform(0, 1, (8, len(next(iter(optima.values())))))
    for s in args.surrogate:
        sname, spec = s.split("=", 1)
        gp_pkl, sc_dir = spec.split("@", 1)
        sc = _scalers(sc_dir)
        if not np.allclose(sc["X_pipeline"].inverse_transform(probe),
                           norm["X_pipeline"].inverse_transform(probe)):
            raise SystemExit(f"{sname}: X scaler differs from --norm-dir's; optima not comparable")
        gp = pickle.load(open(gp_pkl, "rb"))
        print(f"loaded {sname} surrogate ({type(gp).__name__})", flush=True)
        for oname, x in optima.items():
            m, v = gp.predict(x)
            m, v = np.asarray(m)[0], np.asarray(v)[0]
            # back to physical in the surrogate's own scalers, then into the common space
            phys = np.stack([sc[f"Y_pipeline_{vn}"].inverse_transform(m[None, :, j])[0]
                             for j, vn in enumerate(VARS)], 1)
            sd = np.stack([np.sqrt(v[:, j]) * sc[f"Y_pipeline_{vn}"].scale_ /
                           norm[f"Y_pipeline_{vn}"].scale_ for j, vn in enumerate(VARS)], 1)
            mn = to_norm(phys)
            preds[(sname, oname)] = mn
            res = mn - obs
            for j, vn in enumerate(VARS):
                for b, sl in blocks.items():
                    rows.append(dict(surrogate=sname, optimum=oname, variable=vn, block=b,
                                     rms_misfit=float(np.sqrt(np.mean(res[sl, j] ** 2))),
                                     rms_gp_sd=float(np.sqrt(np.mean(sd[sl, j] ** 2)))))
        del gp

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pickle.dump(dict(preds=preds, obs=obs, optima=optima, vars=VARS),
                open(Path(args.out).with_suffix(".pkl"), "wb"))

    print("\nRMS misfit to obs (common normalized units), zonal / regional / |global|")
    for (sname, oname) in preds:
        sel = [x for x in rows if x["surrogate"] == sname and x["optimum"] == oname]
        txt = "  ".join(f"{vn} " + "/".join(f"{x['rms_misfit']:.2f}" for x in sel if x["variable"] == vn)
                        for vn in VARS)
        print(f"  {sname:>6s} surrogate @ {oname:>6s} optimum: {txt}")
    names = list(optima)
    snames = sorted({k[0] for k in preds})
    print("\nRMS difference between surrogates' predictions at the same optimum (all features)")
    for oname in names:
        for i in range(len(snames)):
            for k in range(i + 1, len(snames)):
                d = preds[(snames[i], oname)] - preds[(snames[k], oname)]
                print(f"  @ {oname}: {snames[i]} - {snames[k]}: "
                      + "  ".join(f"{vn} {np.sqrt(np.mean(d[:, j] ** 2)):.2f}" for j, vn in enumerate(VARS)))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
