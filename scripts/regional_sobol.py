"""
Regional Sobol sensitivity: does a parameter that barely matters globally matter in
part of parameter space, e.g. near an optimum?

Each parameter's sampled range is split into --n-bins equal intervals. Each "box"
picks one interval per parameter at random (all parameters free; boxes lying wholly
in lambda_low > lambda_high, which the design never samples, are redrawn). Inside the
box a Saltelli sample of the surrogate's mean gives, for every parameter and
variable (autotune_gp.regional_sobol):

  ST      total-order index of the outputs, variance-weighted over the variable's
          features (as sobol_sensitivity.py does globally)
  S1      the same for first order
  effect  absolute total effect: sqrt(mean over features of VT_i), in PPE sd
          (normalized units), i.e. how much the parameter moves the outputs there
  ST_mis  total-order index of the variable's misfit to obs (mean squared normalized
          error over its features)
  sd_box  sqrt(mean over features of the output variance in the box)

At each box centre it also stores the surrogate's prediction and the costs of the
nominal Student-t fit and, with --pca-basis, the PCA likelihood, so the boxes can be
selected afterwards by any fit's cost or distance to any optimum.

Writes <out>/regional_sobol.npz (checkpointed every --save-every boxes; a rerun
resumes). Needs a compute node (the 3-level surrogate).

    python scripts/regional_sobol.py \\
        --gp-pkl .../results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl \\
        --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \\
        --pca-basis .../pca_basis_k20_n85.pkl --n-boxes 5000 --out .../regional_sobol_n85
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from autotune_gp.backend import get_backend                         # noqa: E402
from autotune_gp.config import load_config                          # noqa: E402
from autotune_gp.cost import zrg_cost_function_student_t            # noqa: E402
from autotune_gp.io import load_obs, split_zrg_obs                   # noqa: E402
from autotune_gp.pca_likelihood import pca_cost                      # noqa: E402
from autotune_gp.regional_sobol import saltelli_points, sobol_from_outputs   # noqa: E402
from autotune_gp.transforms import transform_obs                     # noqa: E402

G = {}   # read by forked workers: gp, obs, cost settings


def _ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def draw_boxes(n_boxes, n_bins, names, rng):
    """(n_boxes, d) interval indices; boxes wholly in lambda_low > lambda_high redrawn."""
    d = len(names)
    lo_i = names.index("lambda_low") if "lambda_low" in names else None
    hi_i = names.index("lambda_high") if "lambda_high" in names else None
    out = []
    while len(out) < n_boxes:
        b = rng.integers(0, n_bins, size=d)
        if lo_i is not None and hi_i is not None and b[lo_i] > b[hi_i]:
            continue
        out.append(b)
    return np.array(out)


def one_box(args):
    k, bins = args
    lo, hi, n_bins, n_base = G["lo"], G["hi"], G["n_bins"], G["n_base"]
    w = (hi - lo) / n_bins
    blo, bhi = lo + bins * w, lo + (bins + 1) * w
    d = len(lo)
    X = saltelli_points(blo, bhi, n_base, seed=k)
    M = np.concatenate([np.asarray(G["gp"].predict_batch(X[i:i + G["chunk"]])[0], float)
                        for i in range(0, len(X), G["chunk"])])          # (n_pts, F, V)
    ST, S1, VT, V = sobol_from_outputs(M, n_base, d)                     # (d, F, V), (F, V)
    wf = V / np.maximum(V.sum(axis=0, keepdims=True), 1e-300)
    obs = G["obs"]                                                       # (F, V)
    mis = np.mean((M - obs[None]) ** 2, axis=1)                          # (n_pts, V)
    STm, _, VTm, Vm = sobol_from_outputs(mis, n_base, d)
    c = 0.5 * (blo + bhi)
    m, v = G["gp"].predict_batch(c[None])
    m, v = np.asarray(m, float), np.asarray(v, float)
    st = zrg_cost_function_student_t(m, v, obs[None], **G["cost_kw"])
    pc = pca_cost(m - obs[None], G["pca"]) if G["pca"] is not None else np.nan
    return k, dict(
        ST=np.einsum("fv,dfv->vd", wf, ST), S1=np.einsum("fv,dfv->vd", wf, S1),
        effect=np.sqrt(VT.mean(axis=1)).T, ST_mis=STm.T, effect_mis=np.sqrt(VTm).T,
        sd_box=np.sqrt(V.mean(axis=0)), centre=c,
        centre_misfit=np.mean((m[0] - obs) ** 2, axis=0),
        centre_gp_sd=np.sqrt(v[0].mean(axis=0)), cost_student_t=st, cost_pca=pc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp-pkl", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pca-basis", default=None)
    ap.add_argument("--n-boxes", type=int, default=5000)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--n-base", type=int, default=256, help="Saltelli base size per box (power of 2)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--box-bounds", default=None,
                    help="npz with lo, hi (normalized) for the sampled box, so surrogates trained "
                         "on different levels are analysed over the same box (default: the "
                         "range of the surrogate's own training inputs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    names = list(cfg.optimize.param_physical_bounds)
    var_names = list(cfg.data.variables)
    gp = pickle.load(open(args.gp_pkl, "rb"))
    X_all = np.vstack([np.asarray(x, float) for x in gp.X_levels])
    lo, hi = np.clip(X_all.min(0), 0, 1), np.clip(X_all.max(0), 0, 1)
    if args.box_bounds:
        _bb = np.load(args.box_bounds)
        lo, hi = np.asarray(_bb["lo"], float), np.asarray(_bb["hi"], float)
    pre = Path(cfg.paths.preprocess_dir)
    sc_dir = Path(cfg.multi_fidelity.scalers_dir) if cfg.multi_fidelity.scalers_dir else pre
    sc = pickle.load(open(sc_dir / "scalers.pkl", "rb"))
    obs = np.asarray(transform_obs(split_zrg_obs(load_obs(str(pre / "obs.pkl"))["zrg_obs"],
                                                 n_vars=len(var_names)),
                                   [sc[f"Y_pipeline_{v}"] for v in var_names]), float)[0]
    n_regions = len(cfg.data.regions_list)
    n_snaps = len(cfg.weights.dy)
    n_zonal = obs.shape[0] // n_snaps - n_regions - 1
    G.update(gp=gp, lo=lo, hi=hi, n_bins=args.n_bins, n_base=args.n_base, chunk=args.chunk, obs=obs,
             pca=pickle.load(open(args.pca_basis, "rb")) if args.pca_basis else None,
             cost_kw=dict(var_w=cfg.weights.variables, zrg_w=cfg.weights.zrg, dy_w=cfg.weights.dy,
                          n_zonal=n_zonal, n_regions=n_regions,
                          backend=get_backend("numpy", "cpu"), var_names=var_names,
                          zonal_weights=cfg.weights.zonal_weights,
                          regional_weights=cfg.weights.regional_weights,
                          var_floor=cfg.optimize.var_floor,
                          variance_penalty_kappa=cfg.optimize.variance_penalty_kappa,
                          dof=cfg.optimize.student_t_dof))
    boxes = draw_boxes(args.n_boxes, args.n_bins, names, np.random.default_rng(args.seed))

    path = out / "regional_sobol.npz"
    res = {}
    if path.exists():
        old = np.load(path, allow_pickle=True)
        if np.array_equal(old["boxes"], boxes):
            for k in np.flatnonzero(old["done"]):
                res[int(k)] = {f: old[f][k] for f in old["fields"]}
            _ts(f"resuming: {len(res)} boxes already done")
    todo = [(k, boxes[k]) for k in range(len(boxes)) if k not in res]
    _ts(f"{len(boxes)} boxes ({args.n_bins} bins per parameter, {len(names)} parameters), "
        f"{args.n_base * (len(names) + 2)} surrogate evaluations each; {len(todo)} to do "
        f"on {args.workers} workers")

    def save():
        fields = sorted(next(iter(res.values())))
        done = np.zeros(len(boxes), bool)
        done[list(res)] = True
        arrs = {}
        for f in fields:
            shape = np.shape(next(iter(res.values()))[f])
            a = np.full((len(boxes),) + shape, np.nan)
            for k, r in res.items():
                a[k] = r[f]
            arrs[f] = a
        np.savez(path.with_suffix(".tmp.npz"), boxes=boxes, done=done, fields=np.array(fields),
                 params=np.array(names), variables=np.array(var_names), lo=lo, hi=hi,
                 n_bins=args.n_bins, n_base=args.n_base, **arrs)
        os.replace(path.with_suffix(".tmp.npz"), path)

    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for n, (k, r) in enumerate(pool.imap_unordered(one_box, todo), 1):
            res[k] = r
            if n % args.save_every == 0 or n == len(todo):
                save()
                rate = n / (time.time() - t0)
                _ts(f"  {n}/{len(todo)} boxes, {rate * 3600:.0f}/h, "
                    f"ETA {(len(todo) - n) / rate / 3600:.1f} h")
    _ts(f"saved {path}")


if __name__ == "__main__":
    main()
