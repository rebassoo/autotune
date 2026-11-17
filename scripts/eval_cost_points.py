"""
Evaluate the costs at named parameter points with one surrogate: the nominal
Student-t cost of the config and, with --pca-basis, the PCA likelihood cost
(0.5 chi^2), plus each variable's RMS normalized misfit. Obs come from the
config's preprocess dir (so a twin config gives the twin's target).

--points is an npz with 'names' (n,) and 'X' (n, n_params), normalized; points are
clipped to --box-bounds. Prints a table and writes --out (json).

    python scripts/eval_cost_points.py --gp-pkl .../mf_gp_trained.pkl \\
        --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \\
        --points pts.npz --box-bounds .../sobol_box_bounds_n85.npz --out costs.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.backend import get_backend                         # noqa: E402
from autotune_gp.config import load_config                          # noqa: E402
from autotune_gp.cost import zrg_cost_function_student_t            # noqa: E402
from autotune_gp.io import load_obs, split_zrg_obs                   # noqa: E402
from autotune_gp.pca_likelihood import pca_cost                      # noqa: E402
from autotune_gp.transforms import transform_obs                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp-pkl", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--pca-basis", default=None)
    ap.add_argument("--box-bounds", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    var_names = list(cfg.data.variables)
    pre = Path(cfg.paths.preprocess_dir)
    sc_dir = Path(cfg.multi_fidelity.scalers_dir) if cfg.multi_fidelity and cfg.multi_fidelity.scalers_dir else pre
    sc = pickle.load(open(sc_dir / "scalers.pkl", "rb"))
    obs = np.asarray(transform_obs(split_zrg_obs(load_obs(str(pre / "obs.pkl"))["zrg_obs"], n_vars=len(var_names)),
                                   [sc[f"Y_pipeline_{v}"] for v in var_names]), float)          # (1, F, V)
    n_regions, n_snaps = len(cfg.data.regions_list), len(cfg.weights.dy)
    kw = dict(var_w=cfg.weights.variables, zrg_w=cfg.weights.zrg, dy_w=cfg.weights.dy,
              n_zonal=obs.shape[1] // n_snaps - n_regions - 1, n_regions=n_regions,
              backend=get_backend("numpy", "cpu"), var_names=var_names,
              zonal_weights=cfg.weights.zonal_weights, regional_weights=cfg.weights.regional_weights,
              var_floor=cfg.optimize.var_floor, variance_penalty_kappa=cfg.optimize.variance_penalty_kappa,
              dof=cfg.optimize.student_t_dof)
    pca = pickle.load(open(args.pca_basis, "rb")) if args.pca_basis else None
    pts = np.load(args.points)
    X = np.asarray(pts["X"], float)
    if args.box_bounds:
        bb = np.load(args.box_bounds)
        X = np.clip(X, bb["lo"], bb["hi"])
    gp = pickle.load(open(args.gp_pkl, "rb"))
    m, v = gp.predict_batch(X)
    m, v = np.asarray(m, float), np.asarray(v, float)
    out = []
    print(f"{'point':44s} {'Student-t':>10s} {'PCA 0.5chi2':>12s}  RMS misfit " + " ".join(var_names))
    for i, name in enumerate(pts["names"]):
        st = zrg_cost_function_student_t(m[i:i + 1], v[i:i + 1], obs, **kw)
        pc = pca_cost(m[i:i + 1] - obs, pca) if pca is not None else float("nan")
        mis = np.sqrt(np.mean((m[i] - obs[0]) ** 2, axis=0))
        out.append(dict(name=str(name), student_t=float(st), pca=float(pc), rms_misfit=mis.tolist(),
                        x=X[i].tolist()))
        print(f"{str(name):44s} {st:10.4f} {pc:12.3f}  " + " ".join(f"{x:.3f}" for x in mis))
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
