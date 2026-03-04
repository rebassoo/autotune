from __future__ import annotations
from typing import Dict
import numpy as np
from .backend import Backend


def _rmse_sklearn_multioutput_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """
    Match sklearn.metrics.root_mean_squared_error for common 1D/2D cases:

    - If 1D: sqrt(mean((a-b)^2))
    - If 2D: treat columns as outputs -> RMSE per column over axis=0,
             then average across columns (multioutput="uniform_average")

    With n_obs=1 and shape (1, n), each per-column RMSE = |a-b|, so the
    average equals the MAE across the n zones/regions.
    """
    d = a - b
    if d.ndim == 1:
        return float(np.sqrt(np.mean(d * d)))
    if d.ndim == 2:
        # per-column RMSE, then average over columns
        return float(np.mean(np.sqrt(np.mean(d * d, axis=0))))
    # Fallback: flatten higher dims (shouldn't occur in our cost usage)
    return float(np.sqrt(np.mean(d * d)))


def zrg_cost_function_rmse_like_reference(
    preds,
    obs,
    var_w: Dict[str, float],
    zrg_w: Dict[str, float],
    dy_w: Dict[str, float],
    n_zonal: int,
    n_regions: int,
    backend: Backend,
) -> float:
    """
    Matches run_GPsurrogate_fromsave.py / Final_Surrogate_ToShare_Short.py:

      - 2D slices [:, a0:a1] passed to RMSE (shape (1, n) -> MAE-like per zone)
      - np.mean([A, B, C, D]) for variable aggregation (true mean of 4 terms)
      - L1 for global entries
      - DY1/DY2 weighted sum

    Layout per DY: [zonal (n_zonal), regional (n_regions), global (1)]
    Total features: 2*(n_zonal + n_regions + 1)

    preds, obs shape: (n_obs, n_feat, 4)
    where var index order is [PCP, TLWP, OSR, OLR]
    """
    n_obs, n_feat, n_vars = preds.shape
    if n_obs != 1:
        raise ValueError(
            f"Expected n_obs=1, got {n_obs}. The cost function evaluates one "
            "parameter set against a single observation target; batch evaluation "
            "(n_obs>1) would require returning a vector of costs, which is not "
            "supported."
        )
    if n_vars != 4:
        raise ValueError(f"Expected last dim=4 for variables, got {n_vars}")

    expected = 2 * (n_zonal + n_regions + 1)
    if n_feat != expected:
        raise ValueError(
            "ZRG layout mismatch: got n_feat=%d expected %d (n_zonal=%d n_regions=%d)"
            % (n_feat, expected, n_zonal, n_regions)
        )

    all_num = n_zonal + n_regions + 1
    off = all_num

    # -------------------------
    # NumPy backend (CPU)
    # -------------------------
    if backend.name == "numpy":
        Pp, Tp, Sp, Lp = preds[:, :, 0], preds[:, :, 1], preds[:, :, 2], preds[:, :, 3]
        Po, To, So, Lo = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2], obs[:, :, 3]

        def rmse(a, b) -> float:
            return _rmse_sklearn_multioutput_numpy(a, b)

        def w_rmse(a0, a1) -> float:
            # 2D slices [:, a0:a1] match fromsave reference; np.mean([...]) averages
            # over the 4 variables rather than summing their weighted contributions.
            return float(np.mean([
                var_w["PCP"]  * rmse(Po[:, a0:a1], Pp[:, a0:a1]),
                var_w["TLWP"] * rmse(To[:, a0:a1], Tp[:, a0:a1]),
                var_w["OSR"]  * rmse(So[:, a0:a1], Sp[:, a0:a1]),
                var_w["OLR"]  * rmse(Lo[:, a0:a1], Lp[:, a0:a1]),
            ]))

        DY1_z = zrg_w["zonal"] * w_rmse(0, n_zonal)
        DY1_r = zrg_w["regional"] * w_rmse(n_zonal, n_zonal + n_regions)
        DY1_g = zrg_w["global"] * float(np.mean([
            var_w["PCP"]  * np.abs(Po[:, all_num - 1] - Pp[:, all_num - 1]),
            var_w["TLWP"] * np.abs(To[:, all_num - 1] - Tp[:, all_num - 1]),
            var_w["OSR"]  * np.abs(So[:, all_num - 1] - Sp[:, all_num - 1]),
            var_w["OLR"]  * np.abs(Lo[:, all_num - 1] - Lp[:, all_num - 1]),
        ]))

        DY2_z = zrg_w["zonal"] * w_rmse(off, off + n_zonal)
        DY2_r = zrg_w["regional"] * w_rmse(off + n_zonal, off + n_zonal + n_regions)
        DY2_g = zrg_w["global"] * float(np.mean([
            var_w["PCP"]  * np.abs(Po[:, -1] - Pp[:, -1]),
            var_w["TLWP"] * np.abs(To[:, -1] - Tp[:, -1]),
            var_w["OSR"]  * np.abs(So[:, -1] - Sp[:, -1]),
            var_w["OLR"]  * np.abs(Lo[:, -1] - Lp[:, -1]),
        ]))

        return dy_w["DY1"] * (DY1_z + DY1_r + DY1_g) + dy_w["DY2"] * (DY2_z + DY2_r + DY2_g)

    # -------------------------
    # CuPy backend (GPU)
    # -------------------------
    if backend.name == "cupy":
        cp = backend.xp
        Pp = cp.asarray(preds[:, :, 0])
        Tp = cp.asarray(preds[:, :, 1])
        Sp = cp.asarray(preds[:, :, 2])
        Lp = cp.asarray(preds[:, :, 3])
        Po = cp.asarray(obs[:, :, 0])
        To = cp.asarray(obs[:, :, 1])
        So = cp.asarray(obs[:, :, 2])
        Lo = cp.asarray(obs[:, :, 3])

        def rmse(a, b) -> float:
            d = a - b
            if d.ndim == 1:
                v = cp.sqrt(cp.mean(d * d))
            else:
                v = cp.mean(cp.sqrt(cp.mean(d * d, axis=0)))
            return float(v.get())

        def w_rmse(a0, a1) -> float:
            return float(np.mean([
                var_w["PCP"]  * rmse(Po[:, a0:a1], Pp[:, a0:a1]),
                var_w["TLWP"] * rmse(To[:, a0:a1], Tp[:, a0:a1]),
                var_w["OSR"]  * rmse(So[:, a0:a1], Sp[:, a0:a1]),
                var_w["OLR"]  * rmse(Lo[:, a0:a1], Lp[:, a0:a1]),
            ]))

        DY1_z = zrg_w["zonal"] * w_rmse(0, n_zonal)
        DY1_r = zrg_w["regional"] * w_rmse(n_zonal, n_zonal + n_regions)
        DY1_g = zrg_w["global"] * float(np.mean([
            var_w["PCP"]  * float(cp.abs(Po[:, all_num - 1] - Pp[:, all_num - 1]).mean().get()),
            var_w["TLWP"] * float(cp.abs(To[:, all_num - 1] - Tp[:, all_num - 1]).mean().get()),
            var_w["OSR"]  * float(cp.abs(So[:, all_num - 1] - Sp[:, all_num - 1]).mean().get()),
            var_w["OLR"]  * float(cp.abs(Lo[:, all_num - 1] - Lp[:, all_num - 1]).mean().get()),
        ]))

        DY2_z = zrg_w["zonal"] * w_rmse(off, off + n_zonal)
        DY2_r = zrg_w["regional"] * w_rmse(off + n_zonal, off + n_zonal + n_regions)
        DY2_g = zrg_w["global"] * float(np.mean([
            var_w["PCP"]  * float(cp.abs(Po[:, -1] - Pp[:, -1]).mean().get()),
            var_w["TLWP"] * float(cp.abs(To[:, -1] - Tp[:, -1]).mean().get()),
            var_w["OSR"]  * float(cp.abs(So[:, -1] - Sp[:, -1]).mean().get()),
            var_w["OLR"]  * float(cp.abs(Lo[:, -1] - Lp[:, -1]).mean().get()),
        ]))

        return dy_w["DY1"] * (DY1_z + DY1_r + DY1_g) + dy_w["DY2"] * (DY2_z + DY2_r + DY2_g)

    # -------------------------
    # Torch backend (CPU/GPU)
    # -------------------------
    if backend.name == "torch":
        torch = backend.xp
        dev = backend.torch_device

        Pp = torch.as_tensor(preds[:, :, 0], device=dev)
        Tp = torch.as_tensor(preds[:, :, 1], device=dev)
        Sp = torch.as_tensor(preds[:, :, 2], device=dev)
        Lp = torch.as_tensor(preds[:, :, 3], device=dev)
        Po = torch.as_tensor(obs[:, :, 0], device=dev)
        To = torch.as_tensor(obs[:, :, 1], device=dev)
        So = torch.as_tensor(obs[:, :, 2], device=dev)
        Lo = torch.as_tensor(obs[:, :, 3], device=dev)

        def rmse(a, b) -> float:
            d = a - b
            if d.ndim == 1:
                v = torch.sqrt(torch.mean(d * d))
            else:
                v = torch.mean(torch.sqrt(torch.mean(d * d, dim=0)))
            return float(v.detach().cpu().item())

        def w_rmse(a0, a1) -> float:
            return float(np.mean([
                var_w["PCP"]  * rmse(Po[:, a0:a1], Pp[:, a0:a1]),
                var_w["TLWP"] * rmse(To[:, a0:a1], Tp[:, a0:a1]),
                var_w["OSR"]  * rmse(So[:, a0:a1], Sp[:, a0:a1]),
                var_w["OLR"]  * rmse(Lo[:, a0:a1], Lp[:, a0:a1]),
            ]))

        def _tabs(t) -> float:
            return float(torch.abs(t).mean().detach().cpu().item())

        DY1_z = zrg_w["zonal"] * w_rmse(0, n_zonal)
        DY1_r = zrg_w["regional"] * w_rmse(n_zonal, n_zonal + n_regions)
        DY1_g = zrg_w["global"] * float(np.mean([
            var_w["PCP"]  * _tabs(Po[:, all_num - 1] - Pp[:, all_num - 1]),
            var_w["TLWP"] * _tabs(To[:, all_num - 1] - Tp[:, all_num - 1]),
            var_w["OSR"]  * _tabs(So[:, all_num - 1] - Sp[:, all_num - 1]),
            var_w["OLR"]  * _tabs(Lo[:, all_num - 1] - Lp[:, all_num - 1]),
        ]))

        DY2_z = zrg_w["zonal"] * w_rmse(off, off + n_zonal)
        DY2_r = zrg_w["regional"] * w_rmse(off + n_zonal, off + n_zonal + n_regions)
        DY2_g = zrg_w["global"] * float(np.mean([
            var_w["PCP"]  * _tabs(Po[:, -1] - Pp[:, -1]),
            var_w["TLWP"] * _tabs(To[:, -1] - Tp[:, -1]),
            var_w["OSR"]  * _tabs(So[:, -1] - Sp[:, -1]),
            var_w["OLR"]  * _tabs(Lo[:, -1] - Lp[:, -1]),
        ]))

        return dy_w["DY1"] * (DY1_z + DY1_r + DY1_g) + dy_w["DY2"] * (DY2_z + DY2_r + DY2_g)

    raise ValueError(f"Unsupported backend: {backend.name!r}")
