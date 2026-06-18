"""
Surrogate skill evaluation: R² and RMSE per variable across k-fold splits.

Core functions imported by:
  scripts/evaluate_surrogate.py  (standalone CLI)
  scripts/run_end_to_end.py      (inline after preprocessing)
  scripts/run_two_stage.py       (stage 2, after loading kfold_data.pkl)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from .transforms import fit_transform_X, fit_transform_Y
from .gp import GPWrapper

def evaluate_fold(fold: dict, train_gp: bool = True) -> dict:
    """
    Train the surrogate on one fold's train split and evaluate on its test split.

    fold dict keys (produced by preprocessing/pipeline.py make_folds):
        k, train_run_labels, test_run_labels,
        X_train, X_test,
        Y_train_ZRG, Y_test_ZRG,
        var_names,
        {var}_train / {var}_test for each var in var_names

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


def run_kfold_evaluation(folds: list, train_gp: bool = True) -> dict:
    """Evaluate all folds and return {fold_k: results_dict}."""
    all_results = {}
    for fold in folds:
        k = fold["k"]
        print(f"Evaluating fold {k}  "
              f"(train={len(fold['train_run_labels'])}, "
              f"test={len(fold['test_run_labels'])}) ...")
        results = evaluate_fold(fold, train_gp=train_gp)
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
) -> dict:
    """
    K-fold cross-validation for the AR1 multi-fidelity GP.

    HF data is split into k folds; all LF data is kept in every fold.
    Scalers are fitted on the HF training fold only and applied to LF and
    HF test data — identical in spirit to the single-fidelity evaluate_fold.

    Inputs are in physical (un-normalised) units.
    Y_low must already be aligned to the HF feature layout
    (call align_Y_to_hf_layout first if needed).

    Reports R² / RMSE per variable, variance-weighted across features.
    """
    from .gp_multifidelity import MultiFidelityGPWrapper

    n_hf     = len(X_high) if hasattr(X_high, '__len__') else X_high.shape[0]
    rng      = np.random.RandomState(seed)
    perm     = rng.permutation(n_hf)
    fold_idx = np.array_split(perm, k)

    X_low_arr = np.asarray(X_low,  dtype=float)
    Y_low_arr = np.asarray(Y_low,  dtype=float)   # (n_low, n_feat, n_vars)
    Y_high    = np.asarray(Y_high, dtype=float)   # (n_hf,  n_feat, n_vars)

    n_models = Y_high.shape[1] * len(var_names)   # n_feat * n_vars
    all_results = {}

    for fold_k in range(k):
        test_idx  = fold_idx[fold_k]
        train_idx = np.concatenate([fold_idx[i] for i in range(k) if i != fold_k])

        print(f"\nMF k-fold {fold_k + 1}/{k}  "
              f"(HF train={len(train_idx)}, HF test={len(test_idx)}, "
              f"LF={len(X_low_arr)}) — training {n_models} AR1 models ...")

        X_hi_tr = X_high.iloc[train_idx] if hasattr(X_high, 'iloc') else X_high[train_idx]
        X_hi_te = X_high.iloc[test_idx]  if hasattr(X_high, 'iloc') else X_high[test_idx]
        Y_hi_tr = Y_high[train_idx]
        Y_hi_te = Y_high[test_idx]

        # Fit scalers on HF training fold only (same principle as single-fidelity)
        X_sc,      X_hi_tr_norm = fit_transform_X(X_hi_tr)
        Y_scalers, Y_hi_tr_norm = fit_transform_Y(Y_hi_tr)

        X_lo_norm    = X_sc.transform(X_low_arr)
        X_hi_te_norm = X_sc.transform(np.asarray(X_hi_te, dtype=float))

        Y_lo_norm = np.stack(
            [Y_scalers[j].transform(Y_low_arr[:, :, j]) for j in range(len(var_names))],
            axis=0,
        ).transpose(1, 2, 0)   # (n_low, n_feat, n_vars)

        gp = MultiFidelityGPWrapper(X_lo_norm, Y_lo_norm, X_hi_tr_norm, Y_hi_tr_norm)
        gp.train()

        pred_norm, _ = gp.predict_batch(X_hi_te_norm)   # (n_te, n_feat, n_vars)

        fold_results = {}
        for j, var in enumerate(var_names):
            t_norm = Y_scalers[j].transform(Y_hi_te[:, :, j])   # (n_te, n_feat)
            p_norm = pred_norm[:, :, j]

            r2_norm   = r2_score(t_norm, p_norm, multioutput="variance_weighted")
            rmse_norm = root_mean_squared_error(t_norm, p_norm)

            p_phys = Y_scalers[j].inverse_transform(p_norm)
            t_phys = Y_hi_te[:, :, j]

            r2_phys   = r2_score(t_phys, p_phys, multioutput="variance_weighted")
            rmse_phys = root_mean_squared_error(t_phys, p_phys)

            fold_results[var] = dict(r2_norm=r2_norm, rmse_norm=rmse_norm,
                                     r2_phys=r2_phys, rmse_phys=rmse_phys)

        print_fold_results(fold_k + 1, fold_results)
        all_results[fold_k + 1] = fold_results

    if len(all_results) > 1:
        print_summary(all_results)

    return all_results
