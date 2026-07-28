from __future__ import annotations
from typing import Dict, List, Optional
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
) -> float:
    """
    ZRG Gaussian negative-log-likelihood cost function, using the GP's own
    predictive variance as the noise term (no separate observational-uncertainty
    estimate is assumed). Variable-count and snapshot-count agnostic, mirroring
    the layout/aggregation structure of `zrg_cost_function_mae_weighted`.

    IMPORTANT — squared error, not absolute error: the per-location term is
        0.5*diff**2/var_clipped + 0.5*log(2*pi*var_clipped)
    (diff = pred - obs), i.e. this is a Gaussian NLL (MSE-flavored), not an
    absolute-error metric — do not confuse with `zrg_cost_function_mae_weighted`.
    `var_clipped = max(pred_var, var_floor)` guards against division blow-up
    where the GP's predictive variance collapses toward zero.

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
            term  = 0.5 * (pp - po) ** 2 / var_c + 0.5 * np.log(2 * np.pi * var_c)
            if weights is not None:
                w = np.asarray(weights, dtype=float)
                return float(np.sum(w * term) / np.sum(w))
            return float(np.mean(term))
        def _sq_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(0.5 * (p - o) ** 2 / var_c + 0.5 * np.log(2 * np.pi * var_c))

    elif backend.name == "cupy":
        cp = backend.xp
        def _sq_nll_mean(po, pp, pv, weights=None) -> float:
            po, pp, pv = cp.asarray(po), cp.asarray(pp), cp.asarray(pv)
            var_c = cp.maximum(pv, var_floor)
            term  = 0.5 * (pp - po) ** 2 / var_c + 0.5 * cp.log(2 * cp.pi * var_c)
            if weights is not None:
                w = cp.asarray(weights)
                return float((cp.sum(w * term) / cp.sum(w)).get())
            return float(cp.mean(term).get())
        def _sq_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(0.5 * (float(p) - float(o)) ** 2 / var_c + 0.5 * float(cp.log(2 * cp.pi * var_c)))

    elif backend.name == "torch":
        torch = backend.xp
        dev   = backend.torch_device
        def _sq_nll_mean(po, pp, pv, weights=None) -> float:
            po = torch.as_tensor(po, device=dev)
            pp = torch.as_tensor(pp, device=dev)
            pv = torch.as_tensor(pv, device=dev)
            var_c = torch.clamp(pv, min=var_floor)
            term  = 0.5 * (pp - po) ** 2 / var_c + 0.5 * torch.log(2 * torch.pi * var_c)
            if weights is not None:
                w = torch.as_tensor(weights, dtype=term.dtype, device=dev)
                return float((torch.sum(w * term) / torch.sum(w)).detach().cpu().item())
            return float(term.mean().detach().cpu().item())
        def _sq_nll_scalar(o, p, v) -> float:
            var_c = max(float(v), var_floor)
            return float(0.5 * (float(p) - float(o)) ** 2 / var_c
                         + 0.5 * float(np.log(2 * np.pi * var_c)))

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
