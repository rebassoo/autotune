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
    zonal_weights: Optional[List[float]] = None,
    regional_weights: Optional[List[float]] = None,
) -> float:
    """
    Matches Final_Surrogate_Optimizing_Visualizing.py ZRG_cost_function_mae_weighted:

      - Squeezes preds/obs to (n_feat, 4) then works with 1-D per-variable arrays
      - MAE (optionally area-weighted) for zonal and regional slices
      - L1 for global entries
      - np.mean([A, B, C, D]) for variable aggregation
      - DY1/DY2 weighted sum

    Layout per DY: [zonal (n_zonal), regional (n_regions), global (1)]
    Total features: 2*(n_zonal + n_regions + 1)

    preds, obs shape on entry: (n_obs, n_feat, 4)
    where var index order is [PCP, TLWP, OSR, OLR]
    """
    n_obs, n_feat, n_vars = preds.shape
    if n_obs != 1:
        raise ValueError(
            f"Expected n_obs=1, got {n_obs}. The cost function evaluates one "
            "parameter set against a single observation target."
        )
    if n_vars != 4:
        raise ValueError(f"Expected last dim=4 for variables, got {n_vars}")

    expected = 2 * (n_zonal + n_regions + 1)
    if n_feat != expected:
        raise ValueError(
            "ZRG layout mismatch: got n_feat=%d expected %d (n_zonal=%d n_regions=%d)"
            % (n_feat, expected, n_zonal, n_regions)
        )

    # Squeeze out the n_obs=1 dimension → (n_feat, 4)
    preds = preds.squeeze()
    obs   = obs.squeeze()

    all_num = n_zonal + n_regions + 1
    off     = all_num

    # -------------------------
    # NumPy backend (CPU)
    # -------------------------
    if backend.name == "numpy":
        Pp, Tp, Sp, Lp = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        Po, To, So, Lo = obs[:, 0],   obs[:, 1],   obs[:, 2],   obs[:, 3]

        def mae(a, b, weights=None) -> float:
            return float(_mae_sklearn(a, b, sample_weight=weights))

        def w_mae(a0, a1, weights=None) -> float:
            return float(np.mean([
                var_w["PCP"]  * mae(Po[a0:a1], Pp[a0:a1], weights),
                var_w["TLWP"] * mae(To[a0:a1], Tp[a0:a1], weights),
                var_w["OSR"]  * mae(So[a0:a1], Sp[a0:a1], weights),
                var_w["OLR"]  * mae(Lo[a0:a1], Lp[a0:a1], weights),
            ]))

        DY1_z = zrg_w["zonal"]    * w_mae(0,       n_zonal,              zonal_weights)
        DY1_r = zrg_w["regional"] * w_mae(n_zonal,  n_zonal + n_regions,  regional_weights)
        DY1_g = zrg_w["global"]   * float(np.mean([
            var_w["PCP"]  * abs(Po[all_num - 1] - Pp[all_num - 1]),
            var_w["TLWP"] * abs(To[all_num - 1] - Tp[all_num - 1]),
            var_w["OSR"]  * abs(So[all_num - 1] - Sp[all_num - 1]),
            var_w["OLR"]  * abs(Lo[all_num - 1] - Lp[all_num - 1]),
        ]))

        DY2_z = zrg_w["zonal"]    * w_mae(off,          off + n_zonal,             zonal_weights)
        DY2_r = zrg_w["regional"] * w_mae(off + n_zonal, off + n_zonal + n_regions, regional_weights)
        DY2_g = zrg_w["global"]   * float(np.mean([
            var_w["PCP"]  * abs(Po[-1] - Pp[-1]),
            var_w["TLWP"] * abs(To[-1] - Tp[-1]),
            var_w["OSR"]  * abs(So[-1] - Sp[-1]),
            var_w["OLR"]  * abs(Lo[-1] - Lp[-1]),
        ]))

        return dy_w["DY1"] * (DY1_z + DY1_r + DY1_g) + dy_w["DY2"] * (DY2_z + DY2_r + DY2_g)

    # -------------------------
    # CuPy backend (GPU)
    # -------------------------
    if backend.name == "cupy":
        cp = backend.xp
        Pp = cp.asarray(preds[:, 0])
        Tp = cp.asarray(preds[:, 1])
        Sp = cp.asarray(preds[:, 2])
        Lp = cp.asarray(preds[:, 3])
        Po = cp.asarray(obs[:, 0])
        To = cp.asarray(obs[:, 1])
        So = cp.asarray(obs[:, 2])
        Lo = cp.asarray(obs[:, 3])

        def mae(a, b, weights=None) -> float:
            diff = cp.abs(a - b)
            if weights is not None:
                w = cp.asarray(weights)
                return float((cp.sum(w * diff) / cp.sum(w)).get())
            return float(cp.mean(diff).get())

        def w_mae(a0, a1, weights=None) -> float:
            return float(np.mean([
                var_w["PCP"]  * mae(Po[a0:a1], Pp[a0:a1], weights),
                var_w["TLWP"] * mae(To[a0:a1], Tp[a0:a1], weights),
                var_w["OSR"]  * mae(So[a0:a1], Sp[a0:a1], weights),
                var_w["OLR"]  * mae(Lo[a0:a1], Lp[a0:a1], weights),
            ]))

        DY1_z = zrg_w["zonal"]    * w_mae(0,       n_zonal,              zonal_weights)
        DY1_r = zrg_w["regional"] * w_mae(n_zonal,  n_zonal + n_regions,  regional_weights)
        DY1_g = zrg_w["global"]   * float(np.mean([
            var_w["PCP"]  * float(cp.abs(Po[all_num - 1] - Pp[all_num - 1]).get()),
            var_w["TLWP"] * float(cp.abs(To[all_num - 1] - Tp[all_num - 1]).get()),
            var_w["OSR"]  * float(cp.abs(So[all_num - 1] - Sp[all_num - 1]).get()),
            var_w["OLR"]  * float(cp.abs(Lo[all_num - 1] - Lp[all_num - 1]).get()),
        ]))

        DY2_z = zrg_w["zonal"]    * w_mae(off,          off + n_zonal,             zonal_weights)
        DY2_r = zrg_w["regional"] * w_mae(off + n_zonal, off + n_zonal + n_regions, regional_weights)
        DY2_g = zrg_w["global"]   * float(np.mean([
            var_w["PCP"]  * float(cp.abs(Po[-1] - Pp[-1]).get()),
            var_w["TLWP"] * float(cp.abs(To[-1] - Tp[-1]).get()),
            var_w["OSR"]  * float(cp.abs(So[-1] - Sp[-1]).get()),
            var_w["OLR"]  * float(cp.abs(Lo[-1] - Lp[-1]).get()),
        ]))

        return dy_w["DY1"] * (DY1_z + DY1_r + DY1_g) + dy_w["DY2"] * (DY2_z + DY2_r + DY2_g)

    # -------------------------
    # Torch backend (CPU/GPU)
    # -------------------------
    if backend.name == "torch":
        torch = backend.xp
        dev = backend.torch_device

        Pp = torch.as_tensor(preds[:, 0], device=dev)
        Tp = torch.as_tensor(preds[:, 1], device=dev)
        Sp = torch.as_tensor(preds[:, 2], device=dev)
        Lp = torch.as_tensor(preds[:, 3], device=dev)
        Po = torch.as_tensor(obs[:, 0], device=dev)
        To = torch.as_tensor(obs[:, 1], device=dev)
        So = torch.as_tensor(obs[:, 2], device=dev)
        Lo = torch.as_tensor(obs[:, 3], device=dev)

        def mae(a, b, weights=None) -> float:
            diff = torch.abs(a - b)
            if weights is not None:
                w = torch.as_tensor(weights, dtype=diff.dtype, device=diff.device)
                return float((torch.sum(w * diff) / torch.sum(w)).detach().cpu().item())
            return float(diff.mean().detach().cpu().item())

        def w_mae(a0, a1, weights=None) -> float:
            return float(np.mean([
                var_w["PCP"]  * mae(Po[a0:a1], Pp[a0:a1], weights),
                var_w["TLWP"] * mae(To[a0:a1], Tp[a0:a1], weights),
                var_w["OSR"]  * mae(So[a0:a1], Sp[a0:a1], weights),
                var_w["OLR"]  * mae(Lo[a0:a1], Lp[a0:a1], weights),
            ]))

        def _tabs(t) -> float:
            return float(torch.abs(t).detach().cpu().item())

        DY1_z = zrg_w["zonal"]    * w_mae(0,       n_zonal,              zonal_weights)
        DY1_r = zrg_w["regional"] * w_mae(n_zonal,  n_zonal + n_regions,  regional_weights)
        DY1_g = zrg_w["global"]   * float(np.mean([
            var_w["PCP"]  * _tabs(Po[all_num - 1] - Pp[all_num - 1]),
            var_w["TLWP"] * _tabs(To[all_num - 1] - Tp[all_num - 1]),
            var_w["OSR"]  * _tabs(So[all_num - 1] - Sp[all_num - 1]),
            var_w["OLR"]  * _tabs(Lo[all_num - 1] - Lp[all_num - 1]),
        ]))

        DY2_z = zrg_w["zonal"]    * w_mae(off,          off + n_zonal,             zonal_weights)
        DY2_r = zrg_w["regional"] * w_mae(off + n_zonal, off + n_zonal + n_regions, regional_weights)
        DY2_g = zrg_w["global"]   * float(np.mean([
            var_w["PCP"]  * _tabs(Po[-1] - Pp[-1]),
            var_w["TLWP"] * _tabs(To[-1] - Tp[-1]),
            var_w["OSR"]  * _tabs(So[-1] - Sp[-1]),
            var_w["OLR"]  * _tabs(Lo[-1] - Lp[-1]),
        ]))

        return dy_w["DY1"] * (DY1_z + DY1_r + DY1_g) + dy_w["DY2"] * (DY2_z + DY2_r + DY2_g)

    raise ValueError(f"Unsupported backend: {backend.name!r}")
