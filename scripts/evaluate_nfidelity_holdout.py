"""
Holdout skill of the N-level multi-fidelity surrogate at the TOP fidelity.

The top level (ne256) has far too few members for the 5-fold scheme
evaluate.run_kfold_evaluation_mf uses, and -- more importantly -- the question
here is different: not "how well does the surrogate predict ne128", but "how
well does it predict ne256, the resolution we actually want to optimise".

So: split only the TOP level into folds. Every lower level stays wholly in
training for every fold (the same convention as the two-level k-fold), the
model is retrained per fold, and the held-out top members are predicted. Fold
predictions are pooled before scoring, because with a handful of top members a
per-fold R^2 would be computed from one or two points.

Two baselines are scored on exactly the same held-out points, so the third
level has to earn its place:

  mf2_gp    the existing two-level (ne32 + ne128) GP, which has never seen any
            ne256 data -- i.e. "predict ne256 with the ne128 surrogate"
  ne128_sim the actual ne128 simulation at the same parameter values, where
            that member exists -- i.e. "just use the cheaper resolution", no
            surrogate at all

Usage:
    python scripts/evaluate_nfidelity_holdout.py \
        --level-dir  .../preprocess_output_ne32_prod_v2 \
        --level-dir  .../preprocess_output_ne128_prod_v2 \
        --top-dir    .../preprocess_output_ne256_frontier \
        --scalers-dir .../preprocess_output_ne128_prod_v2 \
        --warm-start-gp /path/to/results_mf_.../mf_gp_trained.pkl \
        --folds 4 --out-dir /path/to/results_dir
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import r2_score

try:                                   # sklearn >= 1.4
    from sklearn.metrics import root_mean_squared_error
except ImportError:                    # pragma: no cover
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(a, b):
        return mean_squared_error(a, b, squared=False)

from autotune_gp.gp_multifidelity import NFidelityGPWrapper, align_Y_to_hf_layout


class _ReorderedWarmStart:
    """Exposes a trained two-level model's ._models in the TOP level's feature order.

    The warm start is indexed by the feature layout it was trained on (ne128).
    If the top level dropped a different set of all-NaN columns, feature f of
    the top layout is not feature f of the warm start, and copying
    hyperparameters positionally would silently pair up unrelated locations.
    """

    def __init__(self, src, src_mask, top_mask):
        src_pos = {c: i for i, c in enumerate(src_mask["valid_feat_indices"])}
        missing = [c for c in top_mask["valid_feat_indices"] if c not in src_pos]
        if missing:
            raise ValueError(
                f"{len(missing)} top-level feature(s) are absent from the warm-start "
                "model's layout, so its hyperparameters cannot be mapped onto them. "
                "Re-run without --warm-start-gp."
            )
        self._order = [src_pos[c] for c in top_mask["valid_feat_indices"]]
        self._models = [[vm[i] for i in self._order] for vm in src._models]


def _load_level(d: str) -> dict:
    d = Path(d)
    with open(d / "gp_proj.pkl", "rb") as f:
        gp = pickle.load(f)
    out = {"X": np.asarray(gp["X_train"], dtype=float),
           "Y": np.asarray(gp["Y_train"], dtype=float),
           "dir": str(d), "mask": None, "names": None}
    mp = d / "column_mask.pkl"
    if mp.exists():
        with open(mp, "rb") as f:
            out["mask"] = pickle.load(f)
    rl = d / "run_list.pkl"
    if rl.exists():
        with open(rl, "rb") as f:
            out["names"] = list(pickle.load(f)["sim_names"])
    return out


def _member_index(name: str):
    last = name.rsplit(".", 1)[-1]
    return int(last[1:]) if last.startswith("m") and last[1:].isdigit() else None


def _drop_matching(X_level, Y_level, X_held, tol=1e-9):
    """Remove lower-level members sitting at a held-out parameter point.

    The design is nested, so for a held-out ne256 member there is normally an
    ne32 and an ne128 run at exactly the same parameters. Leaving those in
    training measures "predict ne256 given cheaper runs at this point"; taking
    them out measures prediction at a parameter setting never simulated at any
    resolution, which is what the optimizer actually asks for.
    """
    if len(X_held) == 0:
        return X_level, Y_level, 0
    d = np.abs(X_level[:, None, :] - X_held[None, :, :]).max(axis=2)
    keep = ~(d <= tol).any(axis=1)
    return X_level[keep], Y_level[keep], int((~keep).sum())


def _fold_worker(args):
    """Train on this fold's training set and predict the held-out top members."""
    (k, tr, te, X_levels, Y_levels, X_top, Y_top,
     top_kernel, warm_path, warm_order, freeze, strict) = args
    if strict:
        held = X_top[te]
        X_levels, Y_levels = list(X_levels), list(Y_levels)
        dropped = []
        for i in range(len(X_levels)):
            X_levels[i], Y_levels[i], nd = _drop_matching(X_levels[i], Y_levels[i], held)
            dropped.append(nd)
        print(f"  fold {k + 1}: strict holdout removed {dropped} lower-level member(s) "
              f"at the {len(held)} held-out parameter point(s)", flush=True)
    warm = None
    if warm_path:
        with open(warm_path, "rb") as f:
            src = pickle.load(f)
        # Same remap as _ReorderedWarmStart, already validated in main().
        warm = SimpleNamespace(_models=[[vm[i] for i in warm_order] for vm in src._models])
    gp = NFidelityGPWrapper(
        X_levels + [X_top[tr]], Y_levels + [Y_top[tr]],
        top_kernel=top_kernel, warm_start=warm,
        # freezing is only meaningful with a warm start to freeze onto
        freeze_lower=bool(freeze and warm is not None),
    )
    t0 = time.time()
    gp.train()
    mean, _ = gp.predict_batch(X_top[te])
    return k, te, mean, time.time() - t0


def _score(truth, pred, var_names, y_scalers=None, label=""):
    """R^2 / RMSE per variable, in normalised and (if scalers given) physical units."""
    rows = []
    for j, v in enumerate(var_names):
        t, p = truth[:, :, j], pred[:, :, j]
        r2n = r2_score(t, p, multioutput="variance_weighted")
        rmn = root_mean_squared_error(t, p)
        if y_scalers is not None:
            tp = y_scalers[j].inverse_transform(t)
            pp = y_scalers[j].inverse_transform(p)
            r2p = r2_score(tp, pp, multioutput="variance_weighted")
            rmp = root_mean_squared_error(tp, pp)
        else:
            r2p = rmp = float("nan")
        rows.append((v, r2n, rmn, r2p, rmp))
    if label:
        print(f"\n  --- {label} ---")
        print(f"  {'Var':6s} {'R2(norm)':>10s} {'RMSE(norm)':>12s} {'R2(phys)':>10s} {'RMSE(phys)':>12s}")
        for v, a, b, c, d in rows:
            print(f"  {v:6s} {a:10.4f} {b:12.6f} {c:10.4f} {d:12.6f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level-dir", action="append", required=True,
                    help="Preprocess dir for one level BELOW the top, lowest first. Repeat.")
    ap.add_argument("--top-dir", required=True, help="Preprocess dir of the top fidelity.")
    ap.add_argument("--scalers-dir", default=None,
                    help="Whose scalers.pkl to use for X and Y (default: highest --level-dir). "
                         "Not the top level: a handful of members cannot estimate them, and "
                         "they must match the warm-start model's space.")
    ap.add_argument("--warm-start-gp", default=None,
                    help="Trained two-level mf_gp_trained.pkl over the two lowest levels.")
    ap.add_argument("--no-freeze-lower", action="store_true",
                    help="Re-fit lower-level hyperparameters too (default: freeze them "
                         "when warm-starting).")
    ap.add_argument("--top-kernel", default="auto", choices=["auto", "ard", "iso"])
    ap.add_argument("--limit-features", type=int, default=0, metavar="N",
                    help="Smoke test: use only the first N ZRG features, so the whole "
                         "path runs in minutes instead of hours. Not for real results.")
    ap.add_argument("--strict-holdout", action="store_true",
                    help="Also remove the lower-level members that sit at the held-out "
                         "parameter points, so no level has seen that point. Measures "
                         "prediction at genuinely novel parameters. Use without "
                         "--warm-start-gp: a warm start fitted on all lower-level "
                         "members carries those points in its hyperparameters.")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0, help="0 = min(folds, cpu_count)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    lower = [_load_level(d) for d in args.level_dir]
    top = _load_level(args.top_dir)
    sc_dir = Path(args.scalers_dir or args.level_dir[-1])
    with open(sc_dir / "scalers.pkl", "rb") as f:
        saved = pickle.load(f)

    n_top = top["X"].shape[0]
    print(f"Levels: " + " -> ".join(f"{Path(l['dir']).name}({l['X'].shape[0]})" for l in lower)
          + f" -> TOP {Path(top['dir']).name}({n_top})")
    if n_top < 4:
        raise SystemExit(f"Only {n_top} top-level member(s); need at least 4 to hold any out.")

    # ---- align every level's features onto the top level's layout ----------
    var_names = [k.replace("Y_pipeline_", "") for k in saved if k.startswith("Y_pipeline_")]
    y_scalers = [saved[f"Y_pipeline_{v}"] for v in var_names]
    x_sc = saved["X_pipeline"]

    if top["mask"] is not None:
        for l in lower:
            if l["mask"] is not None and l["Y"].shape[1] != top["Y"].shape[1]:
                l["Y"] = align_Y_to_hf_layout(l["Y"], l["mask"], top["mask"])
                print(f"  aligned {Path(l['dir']).name} to the top layout: {l['Y'].shape}")

    for l in lower + [top]:
        if l["Y"].shape[1] != top["Y"].shape[1]:
            raise SystemExit(f"{Path(l['dir']).name} has {l['Y'].shape[1]} features, top has "
                             f"{top['Y'].shape[1]}, and no column mask to align them.")

    # ---- normalise (one shared set of scalers for every level) -------------
    def norm_Y(Y):
        return np.stack([y_scalers[j].transform(Y[:, :, j]) for j in range(len(var_names))],
                        axis=0).transpose(1, 2, 0)

    X_levels = [np.asarray(x_sc.transform(l["X"]), dtype=float) for l in lower]
    Y_levels = [norm_Y(l["Y"]) for l in lower]
    X_top, Y_top = np.asarray(x_sc.transform(top["X"]), dtype=float), norm_Y(top["Y"])

    if args.limit_features:
        nf = min(args.limit_features, Y_top.shape[1])
        print(f"  SMOKE TEST: using only the first {nf}/{Y_top.shape[1]} ZRG features")
        Y_levels = [Y[:, :nf, :] for Y in Y_levels]
        Y_top = Y_top[:, :nf, :]

    # ---- warm start, remapped onto the top level's feature order -----------
    warm_order = None
    if args.warm_start_gp:
        with open(args.warm_start_gp, "rb") as f:
            src = pickle.load(f)
        src_mask = lower[-1]["mask"]
        if src_mask is None or top["mask"] is None:
            raise SystemExit("--warm-start-gp needs column_mask.pkl for the top level and "
                             "the level the warm-start model was trained on.")
        # Constructing this validates the mapping and raises if a top-level
        # feature has no counterpart in the warm-start layout.
        warm_order = _ReorderedWarmStart(src, src_mask, top["mask"])._order
        if args.limit_features:
            warm_order = warm_order[:Y_top.shape[1]]
        del src
        print(f"  warm start: {Path(args.warm_start_gp).name}"
              + ("" if args.no_freeze_lower else ", lower levels frozen"))

    # ---- folds over the TOP level only -------------------------------------
    from sklearn.model_selection import KFold
    k = min(args.folds, n_top)
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    jobs = [(i, tr, te, X_levels, Y_levels, X_top, Y_top, args.top_kernel,
             args.warm_start_gp, warm_order, not args.no_freeze_lower,
             args.strict_holdout)
            for i, (tr, te) in enumerate(kf.split(X_top))]
    if args.strict_holdout:
        print("  STRICT holdout: lower-level members at held-out parameter points "
              "are removed from training too")
        if args.warm_start_gp:
            print("  WARNING: --warm-start-gp with --strict-holdout leaks the held-out "
                  "points through the warm start's hyperparameters.")
    print(f"  {k} folds over {n_top} top members "
          f"({[len(j[2]) for j in jobs]} held out per fold)")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    workers = args.workers or min(k, os.cpu_count() or k)
    pred = np.full_like(Y_top, np.nan)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fold_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            ki, te, mean, secs = fut.result()
            pred[te] = mean
            print(f"  [{time.strftime('%H:%M:%S')}] fold {ki + 1}/{k} done in {secs/60:.1f} min",
                  flush=True)
    if np.isnan(pred).any():
        raise SystemExit("Some held-out members were never predicted.")

    print("\n=== Top-fidelity holdout skill (predictions pooled over folds) ===")
    rows_mf = _score(Y_top, pred, var_names, y_scalers, label=f"{len(X_levels)+1}-level MF surrogate")

    # ---- baseline 1: the two-level GP, which never saw any top-level data ---
    rows_b1 = None
    if args.warm_start_gp:
        with open(args.warm_start_gp, "rb") as f:
            src = pickle.load(f)
        m2, _ = src.predict_batch(X_top)
        pos = {c: i for i, c in enumerate(lower[-1]["mask"]["valid_feat_indices"])}
        order = [pos[c] for c in top["mask"]["valid_feat_indices"]]
        rows_b1 = _score(Y_top, m2[:, order, :], var_names, y_scalers,
                         label="baseline: 2-level GP (ne32+ne128), no top-level data")
        del src

    # ---- baseline 2: the next level down's own simulation at the same params
    rows_b2 = None
    if top["names"] and lower[-1]["names"]:
        low_idx = {_member_index(n): i for i, n in enumerate(lower[-1]["names"])}
        pairs = [(i, low_idx[_member_index(n)]) for i, n in enumerate(top["names"])
                 if _member_index(n) in low_idx]
        if len(pairs) >= 3:
            ti = [p[0] for p in pairs]; li = [p[1] for p in pairs]
            rows_b2 = _score(Y_top[ti], Y_levels[-1][li], var_names, y_scalers,
                             label=f"baseline: the {Path(lower[-1]['dir']).name} simulation "
                                   f"at the same parameters ({len(pairs)} matched members)")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "nfidelity_holdout.pkl", "wb") as f:
        pickle.dump({"var_names": var_names, "Y_true": Y_top, "Y_pred": pred,
                     "rows_mf": rows_mf, "rows_2level": rows_b1, "rows_lowsim": rows_b2,
                     "top_dir": top["dir"], "level_dirs": [l["dir"] for l in lower],
                     "folds": k, "seed": args.seed, "n_top": n_top,
                     "strict_holdout": args.strict_holdout,
                     "top_kernel": args.top_kernel,
                     "freeze_lower": not args.no_freeze_lower}, f)
    print(f"\nSaved {out / 'nfidelity_holdout.pkl'}")


if __name__ == "__main__":
    main()
