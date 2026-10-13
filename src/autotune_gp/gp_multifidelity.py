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

import pickle
from pathlib import Path

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
        import time as _time
        from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
        from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
        from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
        import GPy

        n_total  = self.n_vars * self.n_feat
        done     = 0
        t_start  = _time.time()
        t_batch  = _time.time()
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

                # LinearMultiFidelityKernel has input_dim = n_params+1 (emukit appends
                # fidelity index), so GPy sees consistent dimensions with no warning.
                kernels = [GPy.kern.RBF(input_dim=self.n_params, ARD=True)
                           for _ in range(2)]
                mf_kernel = LinearMultiFidelityKernel(kernels)
                model = GPyLinearMultiFidelityModel(X_arr, Y_arr, mf_kernel, n_fidelities=2)

                # Constrain rho to (0, 5]: prevents negative rho (physically
                # implausible for two resolutions of the same model) and the
                # degenerate k0_variance->0 / rho->inf collapse.
                model.kern.scaling_param.constrain_bounded(1e-6, 5.0)

                model.optimize()

                var_models.append(model)
                done += 1
                if done % 10 == 0 or done == n_total:
                    elapsed = _time.time() - t_start
                    batch_t = _time.time() - t_batch
                    remaining = (elapsed / done) * (n_total - done)
                    print(f"  [{_time.strftime('%H:%M:%S')}] AR1 training: {done}/{n_total} models "
                          f"(last 10: {batch_t:.1f}s, elapsed: {elapsed/60:.1f}min, "
                          f"est. remaining: {remaining/60:.1f}min)", flush=True)
                    t_batch = _time.time()

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
        m, v = self.predict_batch(np.atleast_2d(x))
        return m[:1], v[:1]

    def predict_lf(self, x: np.ndarray):
        """Predict at low fidelity. Same signature as predict()."""
        m, v = self.predict_lf_batch(np.atleast_2d(x))
        return m[:1], v[:1]

    def predict_batch(self, X: np.ndarray):
        """
        Predict at high fidelity for a batch of parameter sets.

        X : (n, n_params) — normalised

        Returns
        -------
        mean : (n, n_feat, n_vars)
        var  : (n, n_feat, n_vars)
        """
        return self._predict_batch_at_fidelity(X, fidelity=1)

    def predict_lf_batch(self, X: np.ndarray):
        """Predict at low fidelity for a batch. Same signature as predict_batch()."""
        return self._predict_batch_at_fidelity(X, fidelity=0)

    # ------------------------------------------------------------------
    def save_hyperparameters(
        self,
        path: str,
        var_names: list | None = None,
        feat_labels: list | None = None,
    ) -> dict:
        """
        Extract hyperparameters from all trained AR1 models, save to a pickle,
        and print a summary table.

        Parameters
        ----------
        path        : file path for the pickle (e.g. diagnostics/hyperparams.pkl)
        var_names   : list of variable name strings, length n_vars
        feat_labels : list of ZRG feature label strings, length n_feat

        Saved arrays (all shape (n_vars, n_feat)):
            rho            — AR1 scaling ρ
            k0_variance    — base kernel amplitude
            k1_variance    — discrepancy kernel amplitude
            noise_lf       — LF noise variance
            noise_hf       — HF noise variance

        Shape (n_vars, n_feat, n_params):
            k0_lengthscales
            k1_lengthscales
        """
        if not self._models:
            raise RuntimeError("Call .train() before .save_hyperparameters().")

        var_names   = var_names   or [f"var{j}"  for j in range(self.n_vars)]
        feat_labels = feat_labels or [f"feat{i}" for i in range(self.n_feat)]

        rho             = np.zeros((self.n_vars, self.n_feat))
        k0_variance     = np.zeros((self.n_vars, self.n_feat))
        k1_variance     = np.zeros((self.n_vars, self.n_feat))
        noise_lf        = np.zeros((self.n_vars, self.n_feat))
        noise_hf        = np.zeros((self.n_vars, self.n_feat))
        k0_lengthscales = np.zeros((self.n_vars, self.n_feat, self.n_params))
        k1_lengthscales = np.zeros((self.n_vars, self.n_feat, self.n_params))

        for var_idx in range(self.n_vars):
            for feat_idx in range(self.n_feat):
                m = self._models[var_idx][feat_idx]
                rho[var_idx, feat_idx]               = float(m.kern.scaling_param[0])
                k0_variance[var_idx, feat_idx]       = float(m.kern.kernels[0].variance)
                k1_variance[var_idx, feat_idx]       = float(m.kern.kernels[1].variance)
                noise_lf[var_idx, feat_idx]          = float(m.likelihood.likelihoods_list[0].variance)
                noise_hf[var_idx, feat_idx]          = float(m.likelihood.likelihoods_list[1].variance)
                k0_lengthscales[var_idx, feat_idx]   = m.kern.kernels[0].lengthscale[:]
                k1_lengthscales[var_idx, feat_idx]   = m.kern.kernels[1].lengthscale[:]

        data = dict(
            var_names       = var_names,
            feat_labels     = feat_labels,
            rho             = rho,
            k0_variance     = k0_variance,
            k1_variance     = k1_variance,
            noise_lf        = noise_lf,
            noise_hf        = noise_hf,
            k0_lengthscales = k0_lengthscales,
            k1_lengthscales = k1_lengthscales,
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved hyperparameters → {path}")

        # disc% = k1_var / (rho²·k0_var + k1_var): the true fraction of K_HH
        # variance from the discrepancy kernel, accounting for rho amplification.
        disc_frac  = k1_variance / (rho ** 2 * k0_variance + k1_variance + 1e-30)
        label_w    = max(len(lb) for lb in feat_labels)
        var_w      = max(len(v)  for v in var_names)

        header = (f"  {'Feature':<{label_w}}  "
                  + "  ".join(f"{'rho':>6} {'disc%':>6}  ({v})" for v in var_names))
        print(f"\n{header}")
        print("  " + "-" * (len(header) - 2))

        for feat_idx, label in enumerate(feat_labels):
            row = f"  {label:<{label_w}}  "
            for var_idx in range(len(var_names)):
                r  = rho[var_idx, feat_idx]
                df = disc_frac[var_idx, feat_idx] * 100   # as percent
                row += f"  {r:6.3f} {df:5.1f}%  "
            print(row)

        print()
        for var_idx, var in enumerate(var_names):
            print(f"  {var:{var_w}}  rho (min/mean/max): "
                  f"{rho[var_idx].min():.3f} / {rho[var_idx].mean():.3f} / {rho[var_idx].max():.3f}   "
                  f"disc% (min/mean/max): "
                  f"{disc_frac[var_idx].min()*100:.1f} / {disc_frac[var_idx].mean()*100:.1f} / {disc_frac[var_idx].max()*100:.1f}")

        return data

    # ------------------------------------------------------------------
    def _predict_batch_at_fidelity(self, X: np.ndarray, fidelity: int):
        if not self._models:
            raise RuntimeError("Call .train() before .predict().")

        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        x_with_fid = np.hstack([X, np.full((n, 1), fidelity, dtype=float)])
        # MixedNoise likelihood needs to know the fidelity to apply the right noise term
        y_meta = {"output_index": np.full(n, fidelity, dtype=int)}

        mean = np.zeros((n, self.n_feat, self.n_vars))
        var  = np.zeros((n, self.n_feat, self.n_vars))

        for var_idx in range(self.n_vars):
            for feat_idx in range(self.n_feat):
                m, v = self._models[var_idx][feat_idx].predict(x_with_fid,
                                                               Y_metadata=y_meta)
                mean[:, feat_idx, var_idx] = m[:, 0]
                var [:, feat_idx, var_idx] = v[:, 0]

        return mean, var


# ---------------------------------------------------------------------------
# N-level (>= 2 fidelities) linear autoregressive GP
# ---------------------------------------------------------------------------
class NFidelityGPWrapper:
    """
    Linear autoregressive multi-fidelity GP over N >= 2 fidelity levels.

    Generalises MultiFidelityGPWrapper (which is fixed at two levels) to the
    Kennedy & O'Hagan chain emukit's LinearMultiFidelityKernel implements:

        f_1(x) = g_1(x)
        f_t(x) = rho_{t-1} * f_{t-1}(x) + g_t(x)      t = 2 .. N

    so each level is the one below it, rescaled, plus its own discrepancy GP.
    Levels are ordered low -> high; predictions default to the top level.
    One model is trained per output feature (n_feat * n_vars total), exactly
    as in the two-level wrapper.

    MultiFidelityGPWrapper is deliberately left untouched: existing
    mf_gp_trained.pkl checkpoints unpickle into that class, and adding a level
    dimension to it would break them.

    Parameters
    ----------
    X_levels, Y_levels : lists, low fidelity first, top fidelity last.
        X_levels[k] is (n_k, n_params); Y_levels[k] is (n_k, n_feat, n_vars).
        All Y must be normalised with the *same* scalers (in practice the
        ones fitted on the level with enough members to estimate them, i.e.
        ne128 rather than the handful of ne256 runs).
    top_kernel : 'ard' | 'iso' | 'auto'
        Kernel for the *top* level's discrepancy term. 'auto' (default) picks
        'iso' when the top level has fewer than 3 * n_params members: a 19-D
        ARD lengthscale vector cannot be identified from ~16 points, and
        fitting one anyway produces arbitrary per-parameter lengthscales.
        Lower levels always use ARD.
    warm_start : MultiFidelityGPWrapper or None
        A trained two-level model over this object's two lowest levels. Its
        fitted hyperparameters initialise the corresponding kernels, scaling
        factor and noise terms, which is both faster and steadier than
        starting from GPy's defaults. It must have been trained on the same
        features in the same order.
    freeze_lower : bool
        Only meaningful with warm_start. Fixes every hyperparameter belonging
        to the levels below the top, so the fit only estimates the top level's
        rho, discrepancy kernel and noise. With very few top-level members
        this is the honest choice: those members carry no real information
        about the ne32/ne128 relationship, and letting them nudge it invites
        the optimiser to move ~40 well-determined hyperparameters to chase a
        handful of points.
    """

    def __init__(
        self,
        X_levels: list,
        Y_levels: list,
        top_kernel: str = "auto",
        warm_start=None,
        freeze_lower: bool = False,
    ):
        if len(X_levels) != len(Y_levels):
            raise ValueError(f"{len(X_levels)} X levels but {len(Y_levels)} Y levels.")
        if len(X_levels) < 2:
            raise ValueError("Need at least 2 fidelity levels.")
        if top_kernel not in ("ard", "iso", "auto"):
            raise ValueError(f"top_kernel must be 'ard', 'iso' or 'auto', got {top_kernel!r}")

        self.X_levels = [np.asarray(X, dtype=float) for X in X_levels]
        self.Y_levels = [np.asarray(Y, dtype=float) for Y in Y_levels]
        self.n_levels = len(self.X_levels)

        self.n_params = self.X_levels[-1].shape[1]
        self.n_feat   = self.Y_levels[-1].shape[1]
        self.n_vars   = self.Y_levels[-1].shape[2]

        for k, (X, Y) in enumerate(zip(self.X_levels, self.Y_levels)):
            if X.shape[1] != self.n_params:
                raise ValueError(f"Level {k} has {X.shape[1]} params, top level has "
                                 f"{self.n_params}. All levels share one parameter space.")
            if Y.shape[1:] != (self.n_feat, self.n_vars):
                raise ValueError(f"Level {k} Y is {Y.shape[1:]}, top level is "
                                 f"{(self.n_feat, self.n_vars)}. Align features first "
                                 "(see align_Y_to_hf_layout).")
            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"Level {k}: {X.shape[0]} X rows but {Y.shape[0]} Y rows.")

        n_top = self.X_levels[-1].shape[0]
        if top_kernel == "auto":
            top_kernel = "ard" if n_top >= 3 * self.n_params else "iso"
        self.top_kernel   = top_kernel
        self.warm_start   = warm_start
        self.freeze_lower = bool(freeze_lower) and warm_start is not None
        if freeze_lower and warm_start is None:
            raise ValueError("freeze_lower=True requires warm_start.")

        self._models: list = []      # [var_idx][feat_idx]

    # ------------------------------------------------------------------
    def _build_kernels(self):
        import GPy
        ks = [GPy.kern.RBF(input_dim=self.n_params, ARD=True)
              for _ in range(self.n_levels - 1)]
        ks.append(GPy.kern.RBF(input_dim=self.n_params,
                               ARD=(self.top_kernel == "ard")))
        return ks

    def _apply_warm_start(self, model, var_idx: int, feat_idx: int):
        """Copy the two-level fit into the two lowest levels of `model`."""
        src = self.warm_start._models[var_idx][feat_idx]
        for j in (0, 1):
            k_dst, k_src = model.kern.kernels[j], src.kern.kernels[j]
            k_dst.variance = float(k_src.variance)
            # ARD lengthscale vectors have matching length by construction;
            # guard anyway so a mismatched warm start fails loudly.
            if k_dst.lengthscale.shape != k_src.lengthscale.shape:
                raise ValueError("warm_start kernel lengthscale shape "
                                 f"{k_src.lengthscale.shape} != {k_dst.lengthscale.shape}")
            k_dst.lengthscale[:] = k_src.lengthscale[:]
            model.likelihood.likelihoods_list[j].variance = \
                float(src.likelihood.likelihoods_list[j].variance)
        model.kern.scaling_param[0] = float(src.kern.scaling_param[0])

    def _freeze_lower_levels(self, model):
        for j in range(self.n_levels - 1):
            model.kern.kernels[j].fix()
            model.likelihood.likelihoods_list[j].fix()
        # scaling_param[t] links level t to t+1; freeze all but the topmost.
        for t in range(self.n_levels - 2):
            model.kern.scaling_param[[t]].fix()

    # ------------------------------------------------------------------
    def train(self):
        """Train one N-level AR model per output feature. Logs progress."""
        import time as _time
        from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
        from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
        from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel

        n_total = self.n_vars * self.n_feat
        done, t_start, t_batch = 0, _time.time(), _time.time()
        self._models = []
        print(f"  {self.n_levels}-level AR GP: "
              + " + ".join(f"{X.shape[0]}@L{k}" for k, X in enumerate(self.X_levels))
              + f" points, top discrepancy kernel = {self.top_kernel}"
              + (", lower levels warm-started" if self.warm_start is not None else "")
              + (" and frozen" if self.freeze_lower else ""), flush=True)

        for var_idx in range(self.n_vars):
            var_models = []
            for feat_idx in range(self.n_feat):
                Xs, Ys = [], []
                for k in range(self.n_levels):
                    Yk = self.Y_levels[k][:, feat_idx, var_idx:var_idx + 1]
                    ok = ~np.isnan(Yk).any(axis=1)
                    Xs.append(self.X_levels[k][ok])
                    Ys.append(Yk[ok])
                X_arr, Y_arr = convert_xy_lists_to_arrays(Xs, Ys)

                mf_kernel = LinearMultiFidelityKernel(self._build_kernels())
                model = GPyLinearMultiFidelityModel(X_arr, Y_arr, mf_kernel,
                                                    n_fidelities=self.n_levels)
                # Same rho constraint as the two-level wrapper: keeps rho
                # positive (negative is implausible between two resolutions of
                # one model) and blocks the k_variance->0 / rho->inf collapse.
                model.kern.scaling_param.constrain_bounded(1e-6, 5.0, warning=False)

                if self.warm_start is not None:
                    self._apply_warm_start(model, var_idx, feat_idx)
                if self.freeze_lower:
                    self._freeze_lower_levels(model)

                model.optimize()
                var_models.append(model)

                done += 1
                if done % 10 == 0 or done == n_total:
                    elapsed = _time.time() - t_start
                    batch_t = _time.time() - t_batch
                    print(f"  [{_time.strftime('%H:%M:%S')}] {self.n_levels}-level training: "
                          f"{done}/{n_total} models (last 10: {batch_t:.1f}s, "
                          f"elapsed: {elapsed/60:.1f}min, est. remaining: "
                          f"{(elapsed/done)*(n_total-done)/60:.1f}min)", flush=True)
                    t_batch = _time.time()
            self._models.append(var_models)

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray):
        """Predict at the top fidelity for one parameter set. Returns (1, n_feat, n_vars) x2."""
        m, v = self.predict_batch(np.atleast_2d(x))
        return m[:1], v[:1]

    def predict_batch(self, X: np.ndarray):
        """Predict at the top fidelity for a batch. Returns (n, n_feat, n_vars) x2."""
        return self.predict_level_batch(X, self.n_levels - 1)

    def predict_lf(self, x: np.ndarray):
        """Predict at the level just below the top (the two-level wrapper's 'LF')."""
        m, v = self.predict_level_batch(np.atleast_2d(x), self.n_levels - 2)
        return m[:1], v[:1]

    def predict_lf_batch(self, X: np.ndarray):
        return self.predict_level_batch(X, self.n_levels - 2)

    def predict_level_batch(self, X: np.ndarray, level: int):
        """Predict at an explicit fidelity level (0 = lowest)."""
        if not self._models:
            raise RuntimeError("Call .train() before .predict().")
        if not 0 <= level < self.n_levels:
            raise ValueError(f"level must be in [0, {self.n_levels - 1}], got {level}")

        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        x_with_fid = np.hstack([X, np.full((n, 1), level, dtype=float)])
        y_meta = {"output_index": np.full(n, level, dtype=int)}

        mean = np.zeros((n, self.n_feat, self.n_vars))
        var  = np.zeros((n, self.n_feat, self.n_vars))
        for var_idx in range(self.n_vars):
            for feat_idx in range(self.n_feat):
                m, v = self._models[var_idx][feat_idx].predict(x_with_fid, Y_metadata=y_meta)
                mean[:, feat_idx, var_idx] = m[:, 0]
                var[:, feat_idx, var_idx]  = v[:, 0]
        return mean, var

    # ------------------------------------------------------------------
    def hyperparameters(self) -> dict:
        """Per-model hyperparameters: rho per transition, kernel and noise per level."""
        if not self._models:
            raise RuntimeError("Call .train() before .hyperparameters().")
        nv, nf, nl = self.n_vars, self.n_feat, self.n_levels
        out = {
            "rho":        np.zeros((nv, nf, nl - 1)),
            "k_variance": np.zeros((nv, nf, nl)),
            "noise":      np.zeros((nv, nf, nl)),
            "n_levels":   nl,
            "top_kernel": self.top_kernel,
        }
        # Lengthscales per level. Lower levels are ARD (one per parameter); the
        # top level may be isotropic, in which case its single lengthscale is
        # broadcast across the parameter axis so the array stays rectangular.
        # Saving these matters: without them the only way to inspect what the
        # fit actually did is to unpickle the whole multi-GB model.
        out["lengthscales"] = np.zeros((nv, nf, nl, self.n_params))
        out["top_is_ard"] = (self.top_kernel == "ard")
        for vi in range(nv):
            for fi in range(nf):
                m = self._models[vi][fi]
                out["rho"][vi, fi] = np.asarray(m.kern.scaling_param).ravel()[:nl - 1]
                for k in range(nl):
                    out["k_variance"][vi, fi, k] = float(m.kern.kernels[k].variance)
                    out["noise"][vi, fi, k] = float(m.likelihood.likelihoods_list[k].variance)
                    ls = np.atleast_1d(np.asarray(m.kern.kernels[k].lengthscale).ravel())
                    out["lengthscales"][vi, fi, k] = (ls if ls.size == self.n_params
                                                      else np.repeat(ls[0], self.n_params))
        return out
