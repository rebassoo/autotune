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
