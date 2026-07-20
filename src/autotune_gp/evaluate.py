"""
Surrogate skill evaluation: R² and RMSE per variable across k-fold splits.

Core functions imported by:
  scripts/evaluate_surrogate.py  (standalone CLI)
  scripts/run_end_to_end.py      (inline after preprocessing)
  scripts/run_two_stage.py       (stage 2, after loading kfold_data.pkl)
"""
from __future__ import annotations

import time as _time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, root_mean_squared_error

from .transforms import fit_transform_X, fit_transform_Y
from .gp import GPWrapper


def _run_kfold_mf_fold(args):
    """Worker for one MF k-fold — runs in a separate process."""
    from .gp_multifidelity import MultiFidelityGPWrapper

    (fold_k, train_idx, test_idx,
     X_high_arr, Y_high, X_low_arr, Y_low_arr, var_names) = args

    n_feat   = Y_high.shape[1]

    X_hi_tr = X_high_arr[train_idx]
    X_hi_te = X_high_arr[test_idx]
    Y_hi_tr = Y_high[train_idx]
    Y_hi_te = Y_high[test_idx]

    X_sc,      X_hi_tr_norm = fit_transform_X(X_hi_tr)
    Y_scalers, Y_hi_tr_norm = fit_transform_Y(Y_hi_tr)

    X_lo_norm    = X_sc.transform(X_low_arr)
    X_hi_te_norm = X_sc.transform(X_hi_te)

    Y_lo_norm = np.stack(
        [Y_scalers[j].transform(Y_low_arr[:, :, j]) for j in range(len(var_names))],
        axis=0,
    ).transpose(1, 2, 0)

    t0 = _time.time()
    gp = MultiFidelityGPWrapper(X_lo_norm, Y_lo_norm, X_hi_tr_norm, Y_hi_tr_norm)
    gp.train()
    train_secs = _time.time() - t0

    pred_norm, _ = gp.predict_batch(X_hi_te_norm)

    fold_results = {}
    feat_r2 = {}
    for j, var in enumerate(var_names):
        t_norm = Y_scalers[j].transform(Y_hi_te[:, :, j])
        p_norm = pred_norm[:, :, j]

        r2_norm   = r2_score(t_norm, p_norm, multioutput="variance_weighted")
        rmse_norm = root_mean_squared_error(t_norm, p_norm)

        p_phys = Y_scalers[j].inverse_transform(p_norm)
        t_phys = Y_hi_te[:, :, j]

        r2_phys   = r2_score(t_phys, p_phys, multioutput="variance_weighted")
        rmse_phys = root_mean_squared_error(t_phys, p_phys)

        feat_r2[var] = np.array([
            r2_score(t_phys[:, fi:fi+1], p_phys[:, fi:fi+1])
            for fi in range(n_feat)
        ])

        fold_results[var] = dict(r2_norm=r2_norm, rmse_norm=rmse_norm,
                                 r2_phys=r2_phys, rmse_phys=rmse_phys)

    return fold_k, fold_results, feat_r2, train_secs

def evaluate_fold(fold: dict, train_gp: bool = True, gp_backend: str = "esem") -> dict:
    """
    Train the surrogate on one fold's train split and evaluate on its test split.

    fold dict keys (produced by preprocessing/pipeline.py make_folds):
        k, train_run_labels, test_run_labels,
        X_train, X_test,
        Y_train_ZRG, Y_test_ZRG,
        var_names,
        {var}_train / {var}_test for each var in var_names

    gp_backend : 'esem' (default, GPflow joint model) or 'gpy' (independent
                 per-feature GPy models). Must match the backend used to train
                 the deployed surrogate, otherwise the reported R² does not
                 describe the model actually being optimized.

    Returns dict keyed by variable name, each with:
        r2_norm, rmse_norm   -- in normalised (StandardScaler) space
        r2_phys, rmse_phys   -- in physical (inverse-transformed) space
    """
    var_names   = fold["var_names"]
    X_train     = fold["X_train"]
    X_test      = fold["X_test"]
    Y_train_ZRG = fold["Y_train_ZRG"]
    test_labels = fold["test_run_labels"]

    # Normalise (fit on train only)
    X_sc,      X_train_norm   = fit_transform_X(X_train)
    Y_scalers, Y_train_norm   = fit_transform_Y(Y_train_ZRG)
    X_test_norm = X_sc.transform(X_test)

    # Normalise test variables with the train-fitted scalers
    norm_test = {
        var: Y_scalers[j].transform(fold[f"{var}_test"])
        for j, var in enumerate(var_names)
    }

    # Train surrogate
    if gp_backend == "gpy":
        from .gp_gpy import SingleFidelityGPyWrapper
        gp = SingleFidelityGPyWrapper(X_train_norm, Y_train_norm)
    else:
        gp = GPWrapper(X_train_norm, Y_train_norm)
    if train_gp:
        gp.train()

    # Predict on test set
    m_gp, _ = gp.predict(X_test_norm)   # (n_test, n_feat, n_vars)

    results = {}
    for j, var in enumerate(var_names):
        pred_norm = pd.DataFrame(m_gp[:, :, j], index=test_labels)
        true_norm = pd.DataFrame(norm_test[var])

        r2_norm   = r2_score(true_norm, pred_norm, multioutput="variance_weighted")
        rmse_norm = root_mean_squared_error(true_norm, pred_norm)

        pred_phys = pd.DataFrame(Y_scalers[j].inverse_transform(pred_norm))
        true_phys = fold[f"{var}_test"].reset_index(drop=True)

        r2_phys   = r2_score(true_phys, pred_phys, multioutput="variance_weighted")
        rmse_phys = root_mean_squared_error(true_phys, pred_phys)

        results[var] = {
            "r2_norm":   r2_norm,
            "rmse_norm": rmse_norm,
            "r2_phys":   r2_phys,
            "rmse_phys": rmse_phys,
        }
    return results


def print_fold_results(fold_k: int, results: dict):
    print(f"\n--- Fold {fold_k} ---")
    _print_table(results)


def print_summary(all_results: dict):
    """Print mean R²/RMSE across all evaluated folds."""
    print("\n=== Mean across folds ===")
    var_names = list(next(iter(all_results.values())).keys())
    summary = {
        var: {
            metric: np.mean([all_results[k][var][metric] for k in all_results])
            for metric in ("r2_norm", "rmse_norm", "r2_phys", "rmse_phys")
        }
        for var in var_names
    }
    _print_table(summary)


def _print_table(results: dict):
    header = f"{'Var':<6}  {'R²(norm)':>10}  {'RMSE(norm)':>12}  {'R²(phys)':>10}  {'RMSE(phys)':>12}"
    print(header)
    print("-" * len(header))
    for var in results:
        r = results[var]
        print(f"{var:<6}  {r['r2_norm']:>10.4f}  {r['rmse_norm']:>12.6f}  "
              f"{r['r2_phys']:>10.4f}  {r['rmse_phys']:>12.6f}")


def run_kfold_evaluation(folds: list, train_gp: bool = True,
                         gp_backend: str = "esem") -> dict:
    """Evaluate all folds and return {fold_k: results_dict}.

    gp_backend : 'esem' (default) or 'gpy'; forwarded to evaluate_fold so the
                 k-fold R² reflects the surrogate backend actually deployed.
                 Runs serially in the main process — the 'gpy' path is pure
                 GPy/numpy (no TensorFlow), so a later process-executor fork
                 during optimization stays safe.
    """
    all_results = {}
    for fold in folds:
        k = fold["k"]
        print(f"Evaluating fold {k}  "
              f"(train={len(fold['train_run_labels'])}, "
              f"test={len(fold['test_run_labels'])}, backend={gp_backend}) ...")
        results = evaluate_fold(fold, train_gp=train_gp, gp_backend=gp_backend)
        print_fold_results(k, results)
        all_results[k] = results

    if len(all_results) > 1:
        print_summary(all_results)

    return all_results


def run_kfold_evaluation_mf(
    X_high,
    Y_high: np.ndarray,
    X_low,
    Y_low: np.ndarray,
    var_names: list,
    k: int = 5,
    seed: int = 42,
    feat_labels: list | None = None,
    checkpoint_dir: str | None = None,
    n_workers: int = 0,
) -> dict:
    """
    K-fold cross-validation for the AR1 multi-fidelity GP.

    HF data is split into k folds; all LF data is kept in every fold.
    Folds run in parallel (n_workers=0 uses min(k, cpu_count)).

    Inputs are in physical (un-normalised) units.
    Y_low must already be aligned to the HF feature layout.
    """
    import os as _os
    import pickle as _pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_hf      = len(X_high) if hasattr(X_high, '__len__') else X_high.shape[0]
    rng       = np.random.RandomState(seed)
    perm      = rng.permutation(n_hf)
    fold_idx  = np.array_split(perm, k)

    X_high_arr = np.asarray(X_high, dtype=float)
    X_low_arr  = np.asarray(X_low,  dtype=float)
    Y_low_arr  = np.asarray(Y_low,  dtype=float)
    Y_high     = np.asarray(Y_high, dtype=float)

    n_feat    = Y_high.shape[1]
    n_models  = n_feat * len(var_names)
    all_results   = {}
    feat_r2_folds = {var: [] for var in var_names}

    if checkpoint_dir:
        _os.makedirs(checkpoint_dir, exist_ok=True)

    # Separate already-checkpointed folds from folds that need running
    pending_args = []
    for fold_k in range(k):
        if checkpoint_dir:
            ckpt = _os.path.join(checkpoint_dir, f"kfold_mf_fold{fold_k + 1}.pkl")
            if _os.path.exists(ckpt):
                print(f"[{_time.strftime('%H:%M:%S')}] Fold {fold_k + 1}: loading checkpoint {ckpt}", flush=True)
                with open(ckpt, "rb") as f:
                    saved = _pickle.load(f)
                all_results[fold_k + 1] = saved["fold_results"]
                for var in var_names:
                    feat_r2_folds[var].append(saved["feat_r2"][var])
                print_fold_results(fold_k + 1, saved["fold_results"])
                continue
        train_idx = np.concatenate([fold_idx[i] for i in range(k) if i != fold_k])
        test_idx  = fold_idx[fold_k]
        pending_args.append((fold_k, train_idx, test_idx,
                             X_high_arr, Y_high, X_low_arr, Y_low_arr, var_names))

    if pending_args:
        workers = n_workers or min(k, _os.cpu_count() or k)
        print(f"[{_time.strftime('%H:%M:%S')}] Running {len(pending_args)} fold(s) in parallel "
              f"with {workers} workers ({n_models} AR1 models per fold) ...", flush=True)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_kfold_mf_fold, a): a[0] for a in pending_args}
            for fut in as_completed(futures):
                fold_k, fold_results, feat_r2, train_secs = fut.result()
                print(f"[{_time.strftime('%H:%M:%S')}] Fold {fold_k + 1} done in "
                      f"{train_secs/60:.1f} min", flush=True)
                all_results[fold_k + 1] = fold_results
                for var in var_names:
                    feat_r2_folds[var].append(feat_r2[var])
                print_fold_results(fold_k + 1, fold_results)

                if checkpoint_dir:
                    ckpt = _os.path.join(checkpoint_dir, f"kfold_mf_fold{fold_k + 1}.pkl")
                    with open(ckpt, "wb") as f:
                        _pickle.dump({"fold_results": fold_results,
                                      "feat_r2": feat_r2}, f)
                    print(f"[{_time.strftime('%H:%M:%S')}]   checkpoint saved → {ckpt}", flush=True)

    if len(all_results) > 1:
        print_summary(all_results)
        _print_feature_r2_summary(feat_r2_folds, var_names, feat_labels)

    return all_results


def _print_feature_r2_summary(feat_r2_folds: dict, var_names: list,
                               feat_labels: list | None):
    """Print per-feature R² (mean ± std across folds) for each variable."""
    n_feat = len(next(iter(feat_r2_folds.values()))[0])
    labels = feat_labels or [f"feat{i}" for i in range(n_feat)]
    label_w = max(len(lb) for lb in labels)

    print("\n=== Per-feature R² (mean ± std across folds, physical space) ===")
    header = f"  {'Feature':<{label_w}}  " + "  ".join(
        f"{'R²':>6} {'±':>1} {'std':>5}  ({v})" for v in var_names
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for fi, label in enumerate(labels):
        row = f"  {label:<{label_w}}  "
        for var in var_names:
            arr  = np.array([fold[fi] for fold in feat_r2_folds[var]])
            mean = arr.mean()
            std  = arr.std()
            row += f"  {mean:+6.3f} ± {std:5.3f}  "
        print(row)

    # Overall summary line
    print()
    for var in var_names:
        arr  = np.array(feat_r2_folds[var])   # (k, n_feat)
        mean = arr.mean(axis=1).mean()         # mean over folds of fold-mean
        std  = arr.mean(axis=1).std()          # std over folds
        print(f"  {var}  overall R² (mean ± std across folds): "
              f"{mean:+.3f} ± {std:.3f}")


def select_top_params_hf(X_high, Y_high, var_names, k=6, param_names=None):
    """Select the top-k parameters by HF Pearson correlation.

    For each (parameter, variable) pair computes mean |r| across all ZRG
    features.  Each parameter is then scored by its maximum mean |r| across
    variables.  Returns the indices (into X_high columns) of the top-k
    parameters, sorted in their original order.

    Parameters
    ----------
    X_high      : array-like, shape (n_hf, n_params)
    Y_high      : ndarray, shape (n_hf, n_feat, n_vars)
    var_names   : list of str
    k           : number of parameters to keep
    param_names : optional list of str — used only for the printed summary

    Returns
    -------
    top_idx     : ndarray of int, shape (k,), sorted ascending
    scores      : ndarray of float, shape (n_params,) — max mean |r| per param
    """
    X       = np.asarray(X_high, dtype=float)
    n_params = X.shape[1]
    n_feat   = Y_high.shape[1]
    n_vars   = len(var_names)

    mean_abs_r = np.zeros((n_params, n_vars))
    for pi in range(n_params):
        for vi in range(n_vars):
            rs = [abs(pearsonr(X[:, pi], Y_high[:, fi, vi])[0])
                  for fi in range(n_feat)]
            mean_abs_r[pi, vi] = np.mean(rs)

    scores  = mean_abs_r.max(axis=1)
    top_idx = np.sort(np.argsort(scores)[::-1][:k])

    names = param_names or [str(i) for i in range(n_params)]
    print(f"\n  Top {k} parameters by HF mean |Pearson r| (max across variables):")
    for rank, idx in enumerate(np.argsort(scores)[::-1][:k], 1):
        per_var = "  ".join(f"{var}={mean_abs_r[idx, vi]:.3f}"
                            for vi, var in enumerate(var_names))
        print(f"    {rank:2d}. {names[idx]:<30s}  score={scores[idx]:.3f}  [{per_var}]")

    return top_idx, scores
