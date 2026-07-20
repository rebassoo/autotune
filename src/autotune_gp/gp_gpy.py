"""
Single-fidelity GP surrogate built directly on GPy — an alternative to the
default ESEm/GPflow ``GPWrapper``.

Motivation
----------
The ESEm wrapper trains one *joint* GPflow model over all features (via ESEm's
Flatten processor).  It works, but the underlying TensorFlow model cannot be
pickled and does not survive ``fork``, so single-fidelity runs must retrain the
GP every time and can only parallelise optimization with threads.

This wrapper mirrors :class:`MultiFidelityGPWrapper`: it trains one independent
GPy ``GPRegression`` (ARD RBF) per output feature — ``n_feat * n_vars`` models —
on the high-fidelity data only (no low-fidelity term, no fidelity index, no
AR1 kernel).  Consequences:

* it pickles cleanly, so ``sf_gp_*_trained.pkl`` save/reload actually works;
* it survives ``fork``, so optimization can use the process executor (the same
  path the multi-fidelity model uses);
* it is architecturally identical to the AR1 model minus the low-fidelity data,
  which makes single-fidelity-vs-multi-fidelity comparisons clean.

It is *not* a drop-in numerical match to the ESEm wrapper: independent
per-feature GPs (own lengthscales, one per feature) differ from ESEm's single
joint GP (shared lengthscales).  Selectable via ``runtime.sf_gp_backend``; the
default stays ESEm so historical behaviour is unchanged.

Thread-safety note: like GPy generally, a single model object is *not* safe to
predict on from multiple threads at once (kernel slice state lives on the shared
object).  Parallelise optimization across processes, not threads.
"""
from __future__ import annotations

import time as _time

import numpy as np


class SingleFidelityGPyWrapper:
    """One GPy GPRegression (ARD RBF) per output feature.

    Same ``.train()`` / ``.predict()`` interface as ``GPWrapper`` and
    ``MultiFidelityGPWrapper`` so it drops into ``run_stage2``.

    Parameters
    ----------
    X_train_norm : (n, n_params) normalised parameter array
    Y_train_norm : (n, n_feat, n_vars) normalised output array
    """

    def __init__(self, X_train_norm: np.ndarray, Y_train_norm: np.ndarray):
        self.X = np.asarray(X_train_norm, dtype=float)
        self.Y = np.asarray(Y_train_norm, dtype=float)
        self.n_params = self.X.shape[1]
        self.n_feat = self.Y.shape[1]
        self.n_vars = self.Y.shape[2]
        self._models: list = []   # [var_idx][feat_idx]

    # ------------------------------------------------------------------
    def train(self, tf_determinism: bool = True):
        """Train one GPRegression per feature.  ``tf_determinism`` is accepted
        for interface parity with ``GPWrapper`` and ignored (no TensorFlow)."""
        import GPy

        n_total = self.n_vars * self.n_feat
        done = 0
        t_start = _time.time()
        self._models = []

        for var_idx in range(self.n_vars):
            var_models = []
            for feat_idx in range(self.n_feat):
                Yf = self.Y[:, feat_idx, var_idx:var_idx + 1]     # (n, 1)
                kern = GPy.kern.RBF(input_dim=self.n_params, ARD=True)
                m = GPy.models.GPRegression(self.X, Yf, kern)
                m.optimize()
                var_models.append(m)
                done += 1
                if done % 20 == 0 or done == n_total:
                    elapsed = _time.time() - t_start
                    print(f"  [{_time.strftime('%H:%M:%S')}] GPy SF training: "
                          f"{done}/{n_total} models "
                          f"(elapsed {elapsed/60:.1f} min)", flush=True)
            self._models.append(var_models)

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray):
        """Predict at all features for one or a batch of parameter sets.

        x : (n_params,) or (n, n_params) — normalised

        Returns
        -------
        mean, var : (n, n_feat, n_vars)
        """
        if not self._models:
            raise RuntimeError("Call .train() before .predict().")

        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = x.shape[0]
        mean = np.zeros((n, self.n_feat, self.n_vars))
        var = np.zeros((n, self.n_feat, self.n_vars))

        for var_idx in range(self.n_vars):
            for feat_idx in range(self.n_feat):
                mu, va = self._models[var_idx][feat_idx].predict(x)
                mean[:, feat_idx, var_idx] = mu[:, 0]
                var[:, feat_idx, var_idx] = va[:, 0]

        return mean, var
