"""
Multi-fidelity GP wrapper using emukit's AR1 (linear autoregressive) model.

Presents the same .train() / .predict() interface as GPWrapper so it can
be dropped into run_stage2 with minimal changes.

The AR1 model for one output dimension:
    y_high(x) = rho * y_low(x) + delta(x)

where y_low is a GP trained on low-fidelity data, rho is a learned scalar,
and delta is an independent "discrepancy" GP.

One AR1 model is trained per output feature (n_feat * n_vars total), which
mirrors how GPWrapper / esem handles multi-output internally.
"""
from __future__ import annotations

import numpy as np


def align_Y_to_hf_layout(
    Y_low: np.ndarray,
    lf_mask: dict,
    hf_mask: dict,
) -> np.ndarray:
    """
    Reindex Y_low columns to match the high-fidelity feature layout.

    Y_low:   (n_low, n_feat_lf, n_vars)
    lf_mask: column_mask.pkl from low-fidelity preprocessing
    hf_mask: column_mask.pkl from high-fidelity preprocessing

    Returns Y_aligned of shape (n_low, n_feat_hf, n_vars).  Any HF feature
    not present in the LF data is filled with NaN (logged as a warning).
    """
    if hf_mask["n_feat_original"] != lf_mask["n_feat_original"]:
        raise ValueError(
            f"HF and LF column masks have different n_feat_original "
            f"({hf_mask['n_feat_original']} vs {lf_mask['n_feat_original']}). "
            "Both configs must use the same ZRG grid (same n_zonal, n_regions, n_snaps)."
        )

    hf_valid = hf_mask["valid_feat_indices"]
    lf_col_to_pos = {col: pos for pos, col in enumerate(lf_mask["valid_feat_indices"])}

    n_low, _, n_vars = Y_low.shape
    n_feat_hf = len(hf_valid)
    Y_aligned = np.full((n_low, n_feat_hf, n_vars), np.nan)

    missing = []
    for hf_pos, orig_col in enumerate(hf_valid):
        if orig_col in lf_col_to_pos:
            Y_aligned[:, hf_pos, :] = Y_low[:, lf_col_to_pos[orig_col], :]
        else:
            missing.append(orig_col)

    if missing:
        print(f"  Warning: {len(missing)} HF feature(s) absent from LF data — filled with NaN.")

    return Y_aligned


class MultiFidelityGPWrapper:
    """
    AR1 multi-fidelity GP trained with emukit / GPy.

    Parameters
    ----------
    X_low, X_high : (n_low, n_params) and (n_high, n_params) normalised arrays
    Y_low, Y_high : (n_low, n_feat, n_vars) and (n_high, n_feat, n_vars) normalised arrays
                    Both should be normalised with the *high-fidelity* Y scalers.
    """

    def __init__(
        self,
        X_low:  np.ndarray,
        Y_low:  np.ndarray,
        X_high: np.ndarray,
        Y_high: np.ndarray,
    ):
        self.X_low  = np.asarray(X_low,  dtype=float)
        self.Y_low  = np.asarray(Y_low,  dtype=float)
        self.X_high = np.asarray(X_high, dtype=float)
        self.Y_high = np.asarray(Y_high, dtype=float)

        self.n_params = X_high.shape[1]
        self.n_feat   = Y_high.shape[1]
        self.n_vars   = Y_high.shape[2]

        if X_low.shape[1] != self.n_params:
            raise ValueError(
                f"X_low has {X_low.shape[1]} params but X_high has {self.n_params}. "
                "Both fidelities must share the same parameter space."
            )

        self._models: list = []   # list[list[GPyLinearMultiFidelityModel]]
                                  # indexed [var_idx][feat_idx]

    # ------------------------------------------------------------------
    def train(self):
        """Train one AR1 model per output feature.  Logs progress."""
        from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
        from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
        import GPy

        n_total = self.n_vars * self.n_feat
        done    = 0
        self._models = []

        for var_idx in range(self.n_vars):
            var_models = []
            for feat_idx in range(self.n_feat):
                Y_lo = self.Y_low [:, feat_idx, var_idx:var_idx + 1]
                Y_hi = self.Y_high[:, feat_idx, var_idx:var_idx + 1]

                # Drop NaN rows (can occur if a LF feature was absent after alignment)
                lo_ok = ~np.isnan(Y_lo).any(axis=1)
                hi_ok = ~np.isnan(Y_hi).any(axis=1)

                X_arr, Y_arr = convert_xy_lists_to_arrays(
                    [self.X_low[lo_ok], self.X_high[hi_ok]],
                    [Y_lo[lo_ok],        Y_hi[hi_ok]],
                )

                kernel = GPy.kern.RBF(input_dim=self.n_params, ARD=True)
                model  = GPyLinearMultiFidelityModel(X_arr, Y_arr, kernel, n_fidelities=2)
                model.optimize()

                var_models.append(model)
                done += 1
                if done % 10 == 0 or done == n_total:
                    print(f"  AR1 training: {done}/{n_total} models done")

            self._models.append(var_models)

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray):
        """
        Predict at high fidelity for a single parameter set.

        x : (n_params,) or (1, n_params) — normalised

        Returns
        -------
        mean : (1, n_feat, n_vars)
        var  : (1, n_feat, n_vars)
        """
        if not self._models:
            raise RuntimeError("Call .train() before .predict().")

        x = np.atleast_2d(np.asarray(x, dtype=float))
        # Append fidelity index = 1 (high fidelity)
        x_hf = np.hstack([x, np.ones((x.shape[0], 1))])

        mean = np.zeros((1, self.n_feat, self.n_vars))
        var  = np.zeros((1, self.n_feat, self.n_vars))

        for var_idx in range(self.n_vars):
            for feat_idx in range(self.n_feat):
                m, v = self._models[var_idx][feat_idx].predict(x_hf)
                mean[0, feat_idx, var_idx] = m[0, 0]
                var [0, feat_idx, var_idx] = v[0, 0]

        return mean, var
