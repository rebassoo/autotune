from __future__ import annotations
from typing import Dict, List, Optional
import math
import numpy as np
from sklearn.metrics import mean_absolute_error as _mae_sklearn
from .backend import Backend


def zrg_cost_function_mae_weighted(
    preds,
    obs,
    var_w: Dict[str, float],
    zrg_w: Dict[str, float],
    dy_w: Dict[str, float],
    n_zonal: int,
    n_regions: int,
    backend: Backend,
    var_names: Optional[List[str]] = None,
    zonal_weights: Optional[List[float]] = None,
    regional_weights: Optional[List[float]] = None,
) -> float:
    """
    ZRG weighted MAE cost function — variable-count and snapshot-count agnostic.

    preds, obs: (1, n_feat, n_vars)
    var_names:  column-order list of variable names (default: list(var_w.keys()))
    dy_w:       {snapshot_label: weight} — one entry per snapshot, in feature-block order
                e.g. {'ANN': 1.0} or {'DJF': 0.5, 'JJA': 0.5}

    Feature layout per snapshot block:
        [zonal_0 … zonal_{n_zonal-1}, reg_0 … reg_{n_regions-1}, global]
    Total features = len(dy_w) * (n_zonal + n_regions + 1)
    """
    n_obs, n_feat, n_vars = preds.shape
    if n_obs != 1:
        raise ValueError(
            f"Expected n_obs=1, got {n_obs}. The cost function evaluates one "
            "parameter set against a single observation target."
        )

    n_snaps = len(dy_w)
    expected = n_snaps * (n_zonal + n_regions + 1)
    if n_feat != expected:
        raise ValueError(
            "ZRG layout mismatch: got n_feat=%d expected %d "
            "(n_snaps=%d n_zonal=%d n_regions=%d)"
            % (n_feat, expected, n_snaps, n_zonal, n_regions)
        )

    if var_names is None:
        var_names = list(var_w.keys())
    if len(var_names) != n_vars:
        raise ValueError(
            f"var_names has {len(var_names)} entries but preds has {n_vars} variable columns"
        )

    name_to_col = {name: idx for idx, name in enumerate(var_names)}

    preds = preds.squeeze()   # (n_feat, n_vars)
    obs   = obs.squeeze()

    snap_size = n_zonal + n_regions + 1

    # ------------------------------------------------------------------
    # Backend-specific scalar MAE and abs helpers
    # ------------------------------------------------------------------
    if backend.name == "numpy":
        def _mae(po, pp, weights=None) -> float:
            return float(_mae_sklearn(po, pp, sample_weight=weights))
        def _abs_diff(a, b) -> float:
            return float(abs(a - b))

    elif backend.name == "cupy":
        cp = backend.xp
        def _mae(po, pp, weights=None) -> float:
            po, pp = cp.asarray(po), cp.asarray(pp)
            diff = cp.abs(po - pp)
            if weights is not None:
                w = cp.asarray(weights)
                return float((cp.sum(w * diff) / cp.sum(w)).get())
            return float(cp.mean(diff).get())
        def _abs_diff(a, b) -> float:
            return float(cp.abs(cp.asarray(a) - cp.asarray(b)).get())

    elif backend.name == "torch":
        torch = backend.xp
        dev   = backend.torch_device
        def _mae(po, pp, weights=None) -> float:
            po = torch.as_tensor(po, device=dev)
            pp = torch.as_tensor(pp, device=dev)
            diff = torch.abs(po - pp)
            if weights is not None:
                w = torch.as_tensor(weights, dtype=diff.dtype, device=dev)
                return float((torch.sum(w * diff) / torch.sum(w)).detach().cpu().item())
            return float(diff.mean().detach().cpu().item())
        def _abs_diff(a, b) -> float:
            return float(torch.abs(torch.as_tensor(a) - torch.as_tensor(b))
                         .detach().cpu().item())

    else:
        raise ValueError(f"Unsupported backend: {backend.name!r}")

    # ------------------------------------------------------------------
    # Generic weighted-MAE helpers (backend-agnostic)
    # ------------------------------------------------------------------
    def _w_mae(slice_obs, slice_pred, weights=None) -> float:
        return float(np.sum([
            var_w[vn] * _mae(slice_obs[:, name_to_col[vn]],
                              slice_pred[:, name_to_col[vn]],
                              weights)
            for vn in var_w
        ]))

    def _w_abs(row_obs, row_pred) -> float:
        return float(np.sum([
            var_w[vn] * _abs_diff(row_obs[name_to_col[vn]], row_pred[name_to_col[vn]])
            for vn in var_w
        ]))

    # ------------------------------------------------------------------
    # Accumulate cost over all snapshot blocks
    # ------------------------------------------------------------------
    total_cost = 0.0
    for snap_idx, (_, snap_weight) in enumerate(dy_w.items()):
        off   = snap_idx * snap_size
        z_end = off + n_zonal
        r_end = z_end + n_regions
        g_idx = off + snap_size - 1

        snap_cost = (
            zrg_w["zonal"]    * _w_mae(obs[off:z_end],   preds[off:z_end],   zonal_weights) +
            zrg_w["regional"] * _w_mae(obs[z_end:r_end],  preds[z_end:r_end],  regional_weights) +
            zrg_w["global"]   * _w_abs(obs[g_idx],        preds[g_idx])
        )
        total_cost += snap_weight * snap_cost

    return total_cost


def zrg_cost_function_gaussian_nll(
    preds,
    pred_var,
    obs,
    var_w: Dict[str, float],
    zrg_w: Dict[str, float],
    dy_w: Dict[str, float],
    n_zonal: int,
    n_regions: int,
    backend: Backend,
    var_names: Optional[List[str]] = None,
    zonal_weights: Optional[List[float]] = None,
    regional_weights: Optional[List[float]] = None,
    var_floor: float = 1e-6,
    variance_penalty_kappa: float = 0.0,
) -> float:
    """
    ZRG Gaussian negative-log-likelihood cost function, using the GP's own
    predictive variance as the noise term (no separate observational-uncertainty
    estimate is assumed). Variable-count and snapshot-count agnostic, mirroring
    the layout/aggregation structure of `zrg_cost_function_mae_weighted`.

    IMPORTANT — squared error, not absolute error: the per-location term is
        0.5*diff**2/var_clipped + 0.5*log(2*pi*var_clipped) + kappa*var_clipped
    (diff = pred - obs), i.e. this is a Gaussian NLL (MSE-flavored), not an
    absolute-error metric — do not confuse with `zrg_cost_function_mae_weighted`.
    `var_clipped = max(pred_var, var_floor)` guards against division blow-up
    where the GP's predictive variance collapses toward zero.

    variance_penalty_kappa (the `kappa*var_clipped` term, default 0.0 = off):
    without it, the optimizer can lower cost not only by shrinking the
    residual but also by moving to a point where pred_var is *larger* --
    inflating the denominator of the first term shrinks the residual penalty
    faster than the log term grows back, in some regimes. Corners of the
    sampled parameter envelope (far from most training points) are exactly
    where a GP's predictive variance tends to be large, so this term can be
    exploited to make the cost look better by finding a point the surrogate
    is simply less certain about, rather than one that is genuinely a better
    fit. Setting variance_penalty_kappa > 0 adds a direct, monotonic penalty
    on pred_var itself, so moving to a higher-variance point can no longer
    net-improve the cost. This is the pessimistic/risk-averse mirror image of
    the optimistic GP-UCB acquisition rule mu(x) + beta**0.5 * sigma(x) from
    Srinivas, Krause, Kakade & Seeger, "Gaussian Process Optimization in the
    Bandit Setting: No Regret and Experimental Design", ICML 2010 (arXiv:0912.3995),
    Eq. (3.6) -- there, +beta**0.5*sigma is added as an exploration *bonus* to
    a reward being *maximized*, deliberately favoring uncertain-but-promising
    points; here the same mean +/- confidence-width device is instead added as
    a penalty to a cost being *minimized*, deliberately disfavoring uncertain
    points, since exploration is not wanted once inside an optimizer scoring a
    fixed, already-trained surrogate. kappa=0.0 recovers the original cost
    exactly.

    Aggregation within each zonal/regional block is a weighted *mean* over
    locations (matching `zrg_cost_function_mae_weighted`'s weighted-mean-MAE
    behavior), not a sum over every individual location. This is a deliberate
    choice, not just compatibility with existing var_w/zrg_w/dy_w weights:
    zonal/regional locations within a block are spatially correlated (adjacent
    latitude bands / nearby regions share the same large-scale bias pattern),
    not independent observations, so summing a per-location NLL term across many
    such locations would overstate the real information content of a
    spatially-smooth field and understate parameter uncertainty. Consequently
    the value returned here is an *average* NLL per block, not a literal
    total-dataset log-likelihood over independent observations — the argmin
    (best-fit parameters) is unaffected by this choice, but any inverse-Hessian
    covariance derived from this cost (see `autotune_gp.uncertainty`) should be
    read as a relative/order-of-magnitude uncertainty indicator, not a
    calibrated absolute confidence interval.

    preds, obs, pred_var: (1, n_feat, n_vars) — pred_var is the GP's predictive
                          variance at the same points/layout as preds.
    var_names:  column-order list of variable names (default: list(var_w.keys()))
    dy_w:       {snapshot_label: weight} — one entry per snapshot, in feature-block order
                e.g. {'ANN': 1.0} or {'DJF': 0.5, 'JJA': 0.5}

    Feature layout per snapshot block:
        [zonal_0 … zonal_{n_zonal-1}, reg_0 … reg_{n_regions-1}, global]
    Total features = len(dy_w) * (n_zonal + n_regions + 1)
    """
    n_obs, n_feat, n_vars = preds.shape
    if n_obs != 1:
        raise ValueError(
            f"Expected n_obs=1, got {n_obs}. The cost function evaluates one "
            "parameter set against a single observation target."
        )
    if pred_var.shape != preds.shape:
        raise ValueError(
            f"pred_var has shape {pred_var.shape} but preds has shape {preds.shape}; "
            "they must match (pred_var is the GP's predictive variance at the "
            "same points/layout as preds)."
        )

    n_snaps = len(dy_w)
    expected = n_snaps * (n_zonal + n_regions + 1)
    if n_feat != expected:
        raise ValueError(
            "ZRG layout mismatch: got n_feat=%d expected %d "
            "(n_snaps=%d n_zonal=%d n_regions=%d)"
            % (n_feat, expected, n_snaps, n_zonal, n_regions)
        )

    if var_names is None:
        var_names = list(var_w.keys())
    if len(var_names) != n_vars:
        raise ValueError(
            f"var_names has {len(var_names)} entries but preds has {n_vars} variable columns"
        )

    name_to_col = {name: idx for idx, name in enumerate(var_names)}

    preds    = preds.squeeze()      # (n_feat, n_vars)
    obs      = obs.squeeze()
    pred_var = pred_var.squeeze()

    snap_size = n_zonal + n_regions + 1

    # ------------------------------------------------------------------
    # Backend-specific squared-error Gaussian-NLL helpers
    # ------------------------------------------------------------------
    if backend.name == "numpy":
        def _sq_nll_mean(po, pp, pv, weights=None) -> float:
            var_c = np.maximum(pv, var_floor)
            term  = (0.5 * (pp - po) ** 2 / var_c + 0.5 * np.log(2 * np.pi * var_c)
                     + variance_penalty_kappa * var_c)
            if weights is not None:
                w = np.asarray(weights, dtype=float)
                return float(np.sum(w * term) / np.sum(w))
            return float(np.mean(term))
        def _sq_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(0.5 * (p - o) ** 2 / var_c + 0.5 * np.log(2 * np.pi * var_c)
                         + variance_penalty_kappa * var_c)

    elif backend.name == "cupy":
        cp = backend.xp
        def _sq_nll_mean(po, pp, pv, weights=None) -> float:
            po, pp, pv = cp.asarray(po), cp.asarray(pp), cp.asarray(pv)
            var_c = cp.maximum(pv, var_floor)
            term  = (0.5 * (pp - po) ** 2 / var_c + 0.5 * cp.log(2 * cp.pi * var_c)
                     + variance_penalty_kappa * var_c)
            if weights is not None:
                w = cp.asarray(weights)
                return float((cp.sum(w * term) / cp.sum(w)).get())
            return float(cp.mean(term).get())
        def _sq_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(0.5 * (float(p) - float(o)) ** 2 / var_c + 0.5 * float(cp.log(2 * cp.pi * var_c))
                         + variance_penalty_kappa * var_c)

    elif backend.name == "torch":
        torch = backend.xp
        dev   = backend.torch_device
        def _sq_nll_mean(po, pp, pv, weights=None) -> float:
            po = torch.as_tensor(po, device=dev)
            pp = torch.as_tensor(pp, device=dev)
            pv = torch.as_tensor(pv, device=dev)
            var_c = torch.clamp(pv, min=var_floor)
            term  = (0.5 * (pp - po) ** 2 / var_c + 0.5 * torch.log(2 * torch.pi * var_c)
                     + variance_penalty_kappa * var_c)
            if weights is not None:
                w = torch.as_tensor(weights, dtype=term.dtype, device=dev)
                return float((torch.sum(w * term) / torch.sum(w)).detach().cpu().item())
            return float(term.mean().detach().cpu().item())
        def _sq_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(0.5 * (float(p) - float(o)) ** 2 / var_c
                         + 0.5 * float(np.log(2 * np.pi * var_c))
                         + variance_penalty_kappa * var_c)

    else:
        raise ValueError(f"Unsupported backend: {backend.name!r}")

    # ------------------------------------------------------------------
    # Generic weighted-mean-NLL helpers (backend-agnostic)
    # ------------------------------------------------------------------
    def _w_sq_nll_mean(slice_obs, slice_pred, slice_var, weights=None) -> float:
        return float(np.sum([
            var_w[vn] * _sq_nll_mean(slice_obs[:, name_to_col[vn]],
                                      slice_pred[:, name_to_col[vn]],
                                      slice_var[:, name_to_col[vn]],
                                      weights)
            for vn in var_w
        ]))

    def _w_sq_nll_scalar(row_obs, row_pred, row_var) -> float:
        return float(np.sum([
            var_w[vn] * _sq_nll_scalar(row_obs[name_to_col[vn]],
                                        row_pred[name_to_col[vn]],
                                        row_var[name_to_col[vn]])
            for vn in var_w
        ]))

    # ------------------------------------------------------------------
    # Accumulate cost over all snapshot blocks
    # ------------------------------------------------------------------
    total_cost = 0.0
    for snap_idx, (_, snap_weight) in enumerate(dy_w.items()):
        off   = snap_idx * snap_size
        z_end = off + n_zonal
        r_end = z_end + n_regions
        g_idx = off + snap_size - 1

        snap_cost = (
            zrg_w["zonal"]    * _w_sq_nll_mean(obs[off:z_end],  preds[off:z_end],
                                                pred_var[off:z_end], zonal_weights) +
            zrg_w["regional"] * _w_sq_nll_mean(obs[z_end:r_end], preds[z_end:r_end],
                                                pred_var[z_end:r_end], regional_weights) +
            zrg_w["global"]   * _w_sq_nll_scalar(obs[g_idx], preds[g_idx], pred_var[g_idx])
        )
        total_cost += snap_weight * snap_cost

    return total_cost


def zrg_cost_function_laplace(
    preds,
    pred_var,
    obs,
    var_w: Dict[str, float],
    zrg_w: Dict[str, float],
    dy_w: Dict[str, float],
    n_zonal: int,
    n_regions: int,
    backend: Backend,
    var_names: Optional[List[str]] = None,
    zonal_weights: Optional[List[float]] = None,
    regional_weights: Optional[List[float]] = None,
    var_floor: float = 1e-6,
    variance_penalty_kappa: float = 0.0,
) -> float:
    """
    ZRG Laplace negative-log-likelihood cost function, using the GP's own
    predictive variance as the scale term -- the likelihood-based counterpart
    to `zrg_cost_function_mae_weighted`'s raw (unscaled, un-normalized) MAE.

    -log Laplace(r; 0, b) = |r|/b + log(2b). Matching b so the Laplace's own
    variance (2*b^2) equals var_clipped (the same convention
    zrg_cost_function_gaussian_nll uses -- var_clipped *is* the modeled
    variance there) gives b = sqrt(var_clipped/2), which simplifies to:

        l(r,v) = sqrt(2)*|r|/sqrt(var_clipped) + 0.5*log(2*var_clipped)
                 + kappa*var_clipped

    `var_clipped = max(pred_var, var_floor)`, same role and same guard as in
    zrg_cost_function_gaussian_nll. `variance_penalty_kappa` is the same
    optional pessimistic variance penalty documented there (0.0 = off,
    recovers the plain Laplace NLL).

    Unlike zrg_cost_function_mae_weighted, this *is* a proper negative
    log-likelihood (includes the log(2*var_clipped) normalizing term, so it
    integrates to 1 as a density) -- it is heteroscedasticity-aware (weights
    each location by the GP's own confidence there), the same property
    zrg_cost_function_student_t has, but with L1 (absolute-error) rather than
    L2 (squared-error) growth in the residual, and lighter tails than
    Student-t's power-law tails.

    Caution: |r| is not differentiable at r=0. This does not meaningfully
    affect the L-BFGS-B optimization itself (subgradients still work, and
    plain zrg_cost_function_mae_weighted already demonstrates this in
    practice) but it does mean the post-hoc parameter-uncertainty report's
    finite-difference Hessian (see autotune_gp.uncertainty) can be unreliable
    for any parameter whose main influence is on a location where the
    residual sits very close to zero at the optimum -- expect NaN or wildly
    unstable stderr for those specific parameters, not necessarily all of
    them.

    Aggregation, argument shapes, and the ZRG feature layout are identical to
    zrg_cost_function_gaussian_nll -- see that function's docstring.
    """
    n_obs, n_feat, n_vars = preds.shape
    if n_obs != 1:
        raise ValueError(
            f"Expected n_obs=1, got {n_obs}. The cost function evaluates one "
            "parameter set against a single observation target."
        )
    if pred_var.shape != preds.shape:
        raise ValueError(
            f"pred_var has shape {pred_var.shape} but preds has shape {preds.shape}; "
            "they must match (pred_var is the GP's predictive variance at the "
            "same points/layout as preds)."
        )

    n_snaps = len(dy_w)
    expected = n_snaps * (n_zonal + n_regions + 1)
    if n_feat != expected:
        raise ValueError(
            "ZRG layout mismatch: got n_feat=%d expected %d "
            "(n_snaps=%d n_zonal=%d n_regions=%d)"
            % (n_feat, expected, n_snaps, n_zonal, n_regions)
        )

    if var_names is None:
        var_names = list(var_w.keys())
    if len(var_names) != n_vars:
        raise ValueError(
            f"var_names has {len(var_names)} entries but preds has {n_vars} variable columns"
        )

    name_to_col = {name: idx for idx, name in enumerate(var_names)}

    preds    = preds.squeeze()      # (n_feat, n_vars)
    obs      = obs.squeeze()
    pred_var = pred_var.squeeze()

    snap_size = n_zonal + n_regions + 1

    # ------------------------------------------------------------------
    # Backend-specific Laplace-NLL helpers
    # ------------------------------------------------------------------
    if backend.name == "numpy":
        def _lap_nll_mean(po, pp, pv, weights=None) -> float:
            var_c = np.maximum(pv, var_floor)
            term  = (np.sqrt(2.0) * np.abs(pp - po) / np.sqrt(var_c)
                     + 0.5 * np.log(2 * var_c) + variance_penalty_kappa * var_c)
            if weights is not None:
                w = np.asarray(weights, dtype=float)
                return float(np.sum(w * term) / np.sum(w))
            return float(np.mean(term))
        def _lap_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(np.sqrt(2.0) * abs(p - o) / np.sqrt(var_c)
                         + 0.5 * np.log(2 * var_c) + variance_penalty_kappa * var_c)

    elif backend.name == "cupy":
        cp = backend.xp
        def _lap_nll_mean(po, pp, pv, weights=None) -> float:
            po, pp, pv = cp.asarray(po), cp.asarray(pp), cp.asarray(pv)
            var_c = cp.maximum(pv, var_floor)
            term  = (cp.sqrt(2.0) * cp.abs(pp - po) / cp.sqrt(var_c)
                     + 0.5 * cp.log(2 * var_c) + variance_penalty_kappa * var_c)
            if weights is not None:
                w = cp.asarray(weights)
                return float((cp.sum(w * term) / cp.sum(w)).get())
            return float(cp.mean(term).get())
        def _lap_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(math.sqrt(2.0) * abs(float(p) - float(o)) / math.sqrt(var_c)
                         + 0.5 * math.log(2 * var_c) + variance_penalty_kappa * var_c)

    elif backend.name == "torch":
        torch = backend.xp
        dev   = backend.torch_device
        def _lap_nll_mean(po, pp, pv, weights=None) -> float:
            po = torch.as_tensor(po, device=dev)
            pp = torch.as_tensor(pp, device=dev)
            pv = torch.as_tensor(pv, device=dev)
            var_c = torch.clamp(pv, min=var_floor)
            term  = (math.sqrt(2.0) * torch.abs(pp - po) / torch.sqrt(var_c)
                     + 0.5 * torch.log(2 * var_c) + variance_penalty_kappa * var_c)
            if weights is not None:
                w = torch.as_tensor(weights, dtype=term.dtype, device=dev)
                return float((torch.sum(w * term) / torch.sum(w)).detach().cpu().item())
            return float(term.mean().detach().cpu().item())
        def _lap_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(math.sqrt(2.0) * abs(float(p) - float(o)) / math.sqrt(var_c)
                         + 0.5 * math.log(2 * var_c) + variance_penalty_kappa * var_c)

    else:
        raise ValueError(f"Unsupported backend: {backend.name!r}")

    # ------------------------------------------------------------------
    # Generic weighted-mean-NLL helpers (backend-agnostic)
    # ------------------------------------------------------------------
    def _w_lap_nll_mean(slice_obs, slice_pred, slice_var, weights=None) -> float:
        return float(np.sum([
            var_w[vn] * _lap_nll_mean(slice_obs[:, name_to_col[vn]],
                                       slice_pred[:, name_to_col[vn]],
                                       slice_var[:, name_to_col[vn]],
                                       weights)
            for vn in var_w
        ]))

    def _w_lap_nll_scalar(row_obs, row_pred, row_var) -> float:
        return float(np.sum([
            var_w[vn] * _lap_nll_scalar(row_obs[name_to_col[vn]],
                                         row_pred[name_to_col[vn]],
                                         row_var[name_to_col[vn]])
            for vn in var_w
        ]))

    # ------------------------------------------------------------------
    # Accumulate cost over all snapshot blocks
    # ------------------------------------------------------------------
    total_cost = 0.0
    for snap_idx, (_, snap_weight) in enumerate(dy_w.items()):
        off   = snap_idx * snap_size
        z_end = off + n_zonal
        r_end = z_end + n_regions
        g_idx = off + snap_size - 1

        snap_cost = (
            zrg_w["zonal"]    * _w_lap_nll_mean(obs[off:z_end],  preds[off:z_end],
                                                 pred_var[off:z_end], zonal_weights) +
            zrg_w["regional"] * _w_lap_nll_mean(obs[z_end:r_end], preds[z_end:r_end],
                                                 pred_var[z_end:r_end], regional_weights) +
            zrg_w["global"]   * _w_lap_nll_scalar(obs[g_idx], preds[g_idx], pred_var[g_idx])
        )
        total_cost += snap_weight * snap_cost

    return total_cost


def zrg_cost_function_student_t(
    preds,
    pred_var,
    obs,
    var_w: Dict[str, float],
    zrg_w: Dict[str, float],
    dy_w: Dict[str, float],
    n_zonal: int,
    n_regions: int,
    backend: Backend,
    var_names: Optional[List[str]] = None,
    zonal_weights: Optional[List[float]] = None,
    regional_weights: Optional[List[float]] = None,
    var_floor: float = 1e-6,
    variance_penalty_kappa: float = 0.0,
    dof: float = 4.0,
) -> float:
    """
    ZRG Student-t negative-log-likelihood cost function, using the GP's own
    predictive variance as the scale term. Identical aggregation structure to
    zrg_cost_function_gaussian_nll (block-mean over zonal/regional/global,
    same var_w/zrg_w/dy_w weighting, same var_floor clipping and optional
    variance_penalty_kappa), but with a Student-t per-location term instead
    of a Gaussian one:

        0.5*log(var_clipped)
        + 0.5*(dof+1)*log(1 + diff**2/(dof*var_clipped))
        + const(dof) + kappa*var_clipped

    (diff = pred - obs), where
        const(dof) = -lgamma((dof+1)/2) + lgamma(dof/2) + 0.5*log(dof*pi)
    is the dof-dependent (but data-independent) Student-t normalizing
    constant -- included so the returned value is a true NLL, though being
    additive and residual-independent it does not affect the argmin or the
    Hessian. As dof -> infinity this term converges exactly to
    zrg_cost_function_gaussian_nll's `0.5*diff**2/var_clipped +
    0.5*log(2*pi*var_clipped)` (the standard Student-t -> Gaussian limit);
    dof is a fixed hyperparameter here (`optimize.student_t_dof`), not fit.

    Motivation: under the Gaussian NLL, the fit term grows quadratically in
    the residual, so a single hard-to-fit variable/location (e.g. OLR, which
    the ne128 k-fold put at R^2~0.44 vs ~0.72 for the other three) can
    dominate the total cost and pull the optimizer toward whatever parameter
    values shrink that one large residual, including the sampled range's
    edges. The Student-t's fit term grows only ~logarithmically once
    diff**2 >> dof*var_clipped, so a single large residual can no longer
    dominate the sum the way it can under Gaussian-NLL -- this is the
    mechanism the empirical at-bound comparison (MAE vs Gaussian-NLL
    boundary-pinning counts, see conversation) motivated trying. var_floor
    and variance_penalty_kappa play the same roles as in
    zrg_cost_function_gaussian_nll: the heavier tails alone do not eliminate
    the variance-inflation exploit (a large diff**2/(dof*var_clipped) can
    still be shrunk by inflating var_clipped), so kappa remains available as
    a direct counter to it.

    preds, obs, pred_var: (1, n_feat, n_vars) — pred_var is the GP's
                          predictive variance at the same points/layout as
                          preds.
    var_names:  column-order list of variable names (default: list(var_w.keys()))
    dy_w:       {snapshot_label: weight} — one entry per snapshot, in feature-block order
                e.g. {'ANN': 1.0} or {'DJF': 0.5, 'JJA': 0.5}

    Feature layout per snapshot block:
        [zonal_0 … zonal_{n_zonal-1}, reg_0 … reg_{n_regions-1}, global]
    Total features = len(dy_w) * (n_zonal + n_regions + 1)
    """
    n_obs, n_feat, n_vars = preds.shape
    if n_obs != 1:
        raise ValueError(
            f"Expected n_obs=1, got {n_obs}. The cost function evaluates one "
            "parameter set against a single observation target."
        )
    if pred_var.shape != preds.shape:
        raise ValueError(
            f"pred_var has shape {pred_var.shape} but preds has shape {preds.shape}; "
            "they must match (pred_var is the GP's predictive variance at the "
            "same points/layout as preds)."
        )
    if dof <= 0:
        raise ValueError(f"dof must be > 0, got {dof!r}")

    n_snaps = len(dy_w)
    expected = n_snaps * (n_zonal + n_regions + 1)
    if n_feat != expected:
        raise ValueError(
            "ZRG layout mismatch: got n_feat=%d expected %d "
            "(n_snaps=%d n_zonal=%d n_regions=%d)"
            % (n_feat, expected, n_snaps, n_zonal, n_regions)
        )

    if var_names is None:
        var_names = list(var_w.keys())
    if len(var_names) != n_vars:
        raise ValueError(
            f"var_names has {len(var_names)} entries but preds has {n_vars} variable columns"
        )

    name_to_col = {name: idx for idx, name in enumerate(var_names)}

    preds    = preds.squeeze()      # (n_feat, n_vars)
    obs      = obs.squeeze()
    pred_var = pred_var.squeeze()

    snap_size = n_zonal + n_regions + 1

    # dof-dependent normalizing constant, computed once with plain Python
    # math (not backend-specific): it's a fixed scalar, independent of any
    # data the backend would otherwise need to touch.
    const_dof = (-math.lgamma((dof + 1.0) / 2.0) + math.lgamma(dof / 2.0)
                 + 0.5 * math.log(dof * math.pi))

    # ------------------------------------------------------------------
    # Backend-specific Student-t NLL helpers
    # ------------------------------------------------------------------
    if backend.name == "numpy":
        def _t_nll_mean(po, pp, pv, weights=None) -> float:
            var_c = np.maximum(pv, var_floor)
            diff2 = (pp - po) ** 2
            term  = (0.5 * np.log(var_c)
                     + 0.5 * (dof + 1.0) * np.log1p(diff2 / (dof * var_c))
                     + const_dof + variance_penalty_kappa * var_c)
            if weights is not None:
                w = np.asarray(weights, dtype=float)
                return float(np.sum(w * term) / np.sum(w))
            return float(np.mean(term))
        def _t_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            diff2 = (float(p) - float(o)) ** 2
            return float(0.5 * np.log(var_c)
                         + 0.5 * (dof + 1.0) * np.log1p(diff2 / (dof * var_c))
                         + const_dof + variance_penalty_kappa * var_c)

    elif backend.name == "cupy":
        cp = backend.xp
        def _t_nll_mean(po, pp, pv, weights=None) -> float:
            po, pp, pv = cp.asarray(po), cp.asarray(pp), cp.asarray(pv)
            var_c = cp.maximum(pv, var_floor)
            diff2 = (pp - po) ** 2
            term  = (0.5 * cp.log(var_c)
                     + 0.5 * (dof + 1.0) * cp.log1p(diff2 / (dof * var_c))
                     + const_dof + variance_penalty_kappa * var_c)
            if weights is not None:
                w = cp.asarray(weights)
                return float((cp.sum(w * term) / cp.sum(w)).get())
            return float(cp.mean(term).get())
        def _t_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            diff2 = (float(p) - float(o)) ** 2
            return float(0.5 * float(cp.log(var_c))
                         + 0.5 * (dof + 1.0) * float(cp.log1p(diff2 / (dof * var_c)))
                         + const_dof + variance_penalty_kappa * var_c)

    elif backend.name == "torch":
        torch = backend.xp
        dev   = backend.torch_device
        def _t_nll_mean(po, pp, pv, weights=None) -> float:
            po = torch.as_tensor(po, device=dev)
            pp = torch.as_tensor(pp, device=dev)
            pv = torch.as_tensor(pv, device=dev)
            var_c = torch.clamp(pv, min=var_floor)
            diff2 = (pp - po) ** 2
            term  = (0.5 * torch.log(var_c)
                     + 0.5 * (dof + 1.0) * torch.log1p(diff2 / (dof * var_c))
                     + const_dof + variance_penalty_kappa * var_c)
            if weights is not None:
                w = torch.as_tensor(weights, dtype=term.dtype, device=dev)
                return float((torch.sum(w * term) / torch.sum(w)).detach().cpu().item())
            return float(term.mean().detach().cpu().item())
        def _t_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            diff2 = (float(p) - float(o)) ** 2
            return float(0.5 * math.log(var_c)
                         + 0.5 * (dof + 1.0) * math.log1p(diff2 / (dof * var_c))
                         + const_dof + variance_penalty_kappa * var_c)

    else:
        raise ValueError(f"Unsupported backend: {backend.name!r}")

    # ------------------------------------------------------------------
    # Generic weighted-mean-NLL helpers (backend-agnostic)
    # ------------------------------------------------------------------
    def _w_t_nll_mean(slice_obs, slice_pred, slice_var, weights=None) -> float:
        return float(np.sum([
            var_w[vn] * _t_nll_mean(slice_obs[:, name_to_col[vn]],
                                     slice_pred[:, name_to_col[vn]],
                                     slice_var[:, name_to_col[vn]],
                                     weights)
            for vn in var_w
        ]))

    def _w_t_nll_scalar(row_obs, row_pred, row_var) -> float:
        return float(np.sum([
            var_w[vn] * _t_nll_scalar(row_obs[name_to_col[vn]],
                                       row_pred[name_to_col[vn]],
                                       row_var[name_to_col[vn]])
            for vn in var_w
        ]))

    # ------------------------------------------------------------------
    # Accumulate cost over all snapshot blocks
    # ------------------------------------------------------------------
    total_cost = 0.0
    for snap_idx, (_, snap_weight) in enumerate(dy_w.items()):
        off   = snap_idx * snap_size
        z_end = off + n_zonal
        r_end = z_end + n_regions
        g_idx = off + snap_size - 1

        snap_cost = (
            zrg_w["zonal"]    * _w_t_nll_mean(obs[off:z_end],  preds[off:z_end],
                                               pred_var[off:z_end], zonal_weights) +
            zrg_w["regional"] * _w_t_nll_mean(obs[z_end:r_end], preds[z_end:r_end],
                                               pred_var[z_end:r_end], regional_weights) +
            zrg_w["global"]   * _w_t_nll_scalar(obs[g_idx], preds[g_idx], pred_var[g_idx])
        )
        total_cost += snap_weight * snap_cost

    return total_cost


def build_training_coverage_tree(X_norm_list, bounds_low, bounds_high):
    """
    Build a KD-tree over the training ensemble (e.g. HF and LF X arrays,
    passed as X_norm_list), rescaled so each dimension spans its own
    [bounds_low, bounds_high] as [0, 1] -- the same per-parameter rescaling
    already used for bounds_from_data and the uncertainty/comparison plots,
    so every dimension contributes comparably to the distance metric instead
    of wide-range dimensions dominating it.

    Returns (tree, threshold). threshold is the training set's own maximum
    internal nearest-neighbor distance -- i.e. the distance between the
    single most isolated pair of training points anywhere in the ensemble.
    This is the natural, data-derived scale training_distance_penalty() uses
    to decide whether a candidate point is farther from the nearest training
    point than the training data ever is from itself.
    """
    from scipy.spatial import cKDTree

    bl = np.asarray(bounds_low, dtype=float)
    bh = np.asarray(bounds_high, dtype=float)
    span = np.where(bh > bl, bh - bl, 1.0)

    X_train = np.vstack([np.asarray(X, dtype=float) for X in X_norm_list])
    Xr = (X_train - bl) / span

    tree = cKDTree(Xr)
    d_self, _ = tree.query(Xr, k=2)          # k=2: nearest OTHER point, not itself
    threshold = float(d_self[:, 1].max())
    return tree, threshold


def training_distance_penalty(x, tree, threshold, lam, bounds_low, bounds_high):
    """
    Penalty discouraging the optimizer from selecting a joint parameter
    combination far from any point in the training ensemble, even when
    every individual dimension stays within its own bounds_from_data range
    -- per-dimension box bounds say nothing about whether that specific
    *combination* was ever jointly sampled.

    penalty(x) = lam * max(0, dist(x, nearest training point) - threshold)

    Zero for any point no farther from the training data than the training
    data already is from itself (a trust-region-style hinge, not a blanket
    penalty on every point outside the training cloud's exact center).
    lam=0.0 (the default) disables this entirely and returns 0.0 without
    even querying the tree.

    This is NOT part of the Gaussian NLL -- it's a separate, additive
    regularization term. Combined with a genuine likelihood, the resulting
    cost is a negative log-POSTERIOR (likelihood x an informal "prior" that
    distrusts unexplored joint combinations) rather than a pure
    log-likelihood; this remains entirely valid to run the Laplace/
    inverse-Hessian approximation (autotune_gp.uncertainty.parameter_uncertainty)
    against -- MAP + inverse-Hessian-of-negative-log-posterior is standard
    Bayesian practice, not a departure from it.

    x, bounds_low, bounds_high must be in the same (full-dimension) space
    the tree was built in -- see build_training_coverage_tree.
    """
    if lam == 0.0:
        return 0.0

    bl = np.asarray(bounds_low, dtype=float)
    bh = np.asarray(bounds_high, dtype=float)
    span = np.where(bh > bl, bh - bl, 1.0)
    xr = (np.asarray(x, dtype=float) - bl) / span

    dist, _ = tree.query(xr.reshape(1, -1), k=1)
    return float(lam * max(0.0, dist[0] - threshold))


# ---------------------------------------------------------------------------
# Per-variable zonal/regional/global weights
# ---------------------------------------------------------------------------
# zrg_w may carry an optional "per_variable" entry, {var: {zonal, regional,
# global}}, overriding the block weights for those variables only (set from
# weights.zrg_per_variable in the config). Every cost here is a sum over
# variables, so the cost with overrides is the sum of plain calls, one per
# group of variables sharing the same block weights. Without the entry the
# wrapped function is called unchanged.
def _with_per_variable_zrg(fn):
    import functools
    import inspect

    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        zrg_w = ba.arguments["zrg_w"]
        per_var = zrg_w.get("per_variable") if isinstance(zrg_w, dict) else None
        if not per_var:
            return fn(*args, **kwargs)
        var_w = ba.arguments["var_w"]
        unknown = set(per_var) - set(var_w)
        if unknown:
            raise ValueError(f"zrg per_variable names {sorted(unknown)} are not in var_w")
        base = {k: zrg_w[k] for k in ("zonal", "regional", "global")}
        if ba.arguments.get("var_names") is None:
            ba.arguments["var_names"] = list(var_w)
        groups: Dict[tuple, Dict[str, float]] = {}
        for vn, w in var_w.items():
            z = {**base, **per_var.get(vn, {})}
            groups.setdefault((z["zonal"], z["regional"], z["global"]), {})[vn] = w
        total = 0.0
        for (zw, rw, gw), sub in groups.items():
            ba.arguments["var_w"] = sub
            ba.arguments["zrg_w"] = {"zonal": zw, "regional": rw, "global": gw}
            total += fn(*ba.args, **ba.kwargs)
        return total

    return wrapper


zrg_cost_function_mae_weighted = _with_per_variable_zrg(zrg_cost_function_mae_weighted)
zrg_cost_function_gaussian_nll = _with_per_variable_zrg(zrg_cost_function_gaussian_nll)
zrg_cost_function_laplace = _with_per_variable_zrg(zrg_cost_function_laplace)
zrg_cost_function_student_t = _with_per_variable_zrg(zrg_cost_function_student_t)
