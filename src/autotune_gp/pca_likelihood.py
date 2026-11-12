"""
Gaussian likelihood on the leading principal components of the outputs, with the
error covariance estimated from data instead of hand-set weights.

All features of all variables (normalized units, one flat vector per run) are
projected onto the leading K EOFs of the PPE: c = V^T r, r = prediction - target.
The cost is the Gaussian negative log-likelihood, up to a constant,

    cost = 0.5 * c^T Sigma^-1 c,      c ~ N(0, Sigma) at the true parameters,

so 2*cost is chi^2 with K degrees of freedom when Sigma is right. Sigma is the
second moment (bias included) of the projected errors of out-of-sample predictions
(e.g. the holdout residuals of the surrogate at the top-level members), with
Ledoit-Wolf shrinkage. It therefore holds the surrogate's correlated errors, the
target's internal variability, and, if the predictions come from a lower
resolution, the resolution difference. Sigma does not depend on the parameters, so
there is no log-variance term to reward regions where the surrogate is confident.

Not included yet: observation error and the model's structural error against
observations, both needed before this is used with real observations.
"""
from __future__ import annotations

import numpy as np


def build_basis(Y_ppe, errors, k, shrink=True):
    """Basis and error covariance.

    Y_ppe   (n_runs, n_feat): PPE outputs, flattened features, normalized units.
    errors  (n_err, n_feat):  out-of-sample prediction - truth, same layout.
    k:      number of EOFs kept.
    Returns {'V': (n_feat, k), 'Sigma': (k, k), 'Sigma_inv': (k, k), 'ppe_var_frac': float}.
    """
    Y = np.asarray(Y_ppe, float)
    E = np.asarray(errors, float)
    if Y.ndim != 2 or E.ndim != 2 or Y.shape[1] != E.shape[1]:
        raise ValueError(f"need 2-D arrays with matching features, got {Y.shape} and {E.shape}")
    if not 1 <= k <= min(Y.shape):
        raise ValueError(f"k={k} outside 1..{min(Y.shape)}")
    _, s, Vt = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
    V = Vt[:k].T
    P = E @ V
    if shrink:
        from sklearn.covariance import LedoitWolf
        Sigma = LedoitWolf(assume_centered=True).fit(P).covariance_
    else:
        Sigma = P.T @ P / len(P)
    return {"V": V, "Sigma": Sigma, "Sigma_inv": np.linalg.inv(Sigma),
            "ppe_var_frac": float(np.sum(s[:k] ** 2) / np.sum(s ** 2))}


def pca_cost(resid, basis):
    """0.5 * c^T Sigma^-1 c for one residual vector (any shape, flattened in C order)."""
    c = np.asarray(resid, float).reshape(-1) @ basis["V"]
    return float(0.5 * c @ basis["Sigma_inv"] @ c)
