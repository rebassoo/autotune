"""Model-discrepancy and observation-error variances for the likelihood costs.

Kennedy & O'Hagan (2001) tie the observations to the top fidelity level as

    z(s) = eta(s, theta*) + delta(s) + eps(s)

so with the emulator plugged in (modular / plug-in likelihood), the variance
at each output feature s is

    V_s(theta) = sigma2_GP(s, theta) + sigma2_delta(s) + sigma2_eps(s).

The cost functions in autotune_gp.cost already take the GP variance as
`pred_var`; the extra terms are simply added to it before the call
(`extra_variance` below builds them in the same (1, n_feat, n_vars) layout).
The surrogate itself is untouched.

sigma_delta is unknown, so scripts/calibrate_model_error.py estimates it from
the residuals at the optimum, one value per variable x {zonal, regional}
block, and re-optimizes until theta and sigma_delta stop moving
(`estimate_block_sigma2` is that per-block step). See
docs/model_error_residual_calibration.md for the full procedure.

Units: the costs work on StandardScaler-normalized targets, one scaler per
variable with a per-feature scale_. A model-error value given in normalized
units therefore means "this fraction of the PPE spread at each feature";
physical values are divided by scale_ per feature.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

BLOCKS = ("zonal", "regional", "global")


def block_indices(n_zonal: int, n_regions: int, n_snaps: int) -> Dict[str, np.ndarray]:
    """Feature indices of each block, pooled over snapshot blocks.

    Layout per snapshot: [zonal_0..zonal_{n_zonal-1}, reg_0..reg_{n_regions-1}, global].
    """
    size = n_zonal + n_regions + 1
    out = {b: [] for b in BLOCKS}
    for k in range(n_snaps):
        off = k * size
        out["zonal"].extend(range(off, off + n_zonal))
        out["regional"].extend(range(off + n_zonal, off + n_zonal + n_regions))
        out["global"].append(off + size - 1)
    return {b: np.asarray(v, dtype=int) for b, v in out.items()}


def block_weights(block: str, zonal_weights, regional_weights, n_snaps: int):
    """Per-location weights the cost uses inside a block, tiled over snapshots."""
    w = {"zonal": zonal_weights, "regional": regional_weights}.get(block)
    return None if w is None else np.tile(np.asarray(w, dtype=float), n_snaps)


def _check_keys(spec: Optional[dict], var_names: List[str], what: str):
    if not spec:
        return
    unknown_vars = [v for v in spec if v not in var_names]
    if unknown_vars:
        raise ValueError(f"optimize.{what} names variables {unknown_vars}; "
                         f"expected a subset of {var_names}")
    allowed = set(BLOCKS) | {f"{b}_phys" for b in BLOCKS}
    for v, blocks in spec.items():
        bad = [k for k in (blocks or {}) if k not in allowed]
        if bad:
            raise ValueError(f"optimize.{what}.{v} has unknown keys {bad}; "
                             f"allowed: {sorted(allowed)}")
        both = [b for b in BLOCKS if b in (blocks or {}) and f"{b}_phys" in blocks]
        if both:
            raise ValueError(f"optimize.{what}.{v} gives both '{both[0]}' and "
                             f"'{both[0]}_phys'; use one.")


def extra_variance(model_error: Optional[dict], obs_error: Optional[dict],
                   var_names: List[str], Y_scalers, n_zonal: int, n_regions: int,
                   n_snaps: int, phys_default: bool = False) -> Optional[np.ndarray]:
    """sigma2_delta + sigma2_eps per feature, normalized units, shape (1, n_feat, n_vars).

    model_error values are normalized-unit standard deviations unless the key
    is '<block>_phys'. obs_error values are physical-unit standard deviations
    (phys_default semantics; '<block>_phys' is accepted as an alias).
    Returns None when neither is set, so callers can skip the addition and
    reproduce the original cost bit-for-bit.
    """
    if not model_error and not obs_error:
        return None
    _check_keys(model_error, var_names, "model_error")
    _check_keys(obs_error, var_names, "obs_error")

    idx = block_indices(n_zonal, n_regions, n_snaps)
    n_feat = n_snaps * (n_zonal + n_regions + 1)
    out = np.zeros((1, n_feat, len(var_names)))
    for j, v in enumerate(var_names):
        scale = np.asarray(Y_scalers[j].scale_, dtype=float)
        if scale.shape[0] != n_feat:
            raise ValueError(f"Y scaler for {v} has {scale.shape[0]} features, "
                             f"layout has {n_feat}.")
        for spec, phys in ((model_error, False), (obs_error, True)):
            blocks = (spec or {}).get(v) or {}
            for b, f in idx.items():
                if b in blocks:
                    s = float(blocks[b])
                    out[0, f, j] += (s / scale[f]) ** 2 if phys else s ** 2
                elif f"{b}_phys" in blocks:
                    out[0, f, j] += (float(blocks[f"{b}_phys"]) / scale[f]) ** 2
    return out


# ---------------------------------------------------------------------------
# Per-location NLL terms, numpy only. Must match autotune_gp.cost exactly
# (tests/test_model_error.py checks this) so the per-block estimate maximizes
# the same likelihood the optimizer uses.
# ---------------------------------------------------------------------------
def _term(likelihood: str, r2: np.ndarray, var: np.ndarray, dof: float, kappa: float):
    if likelihood == "gaussian_gp_variance":
        t = 0.5 * r2 / var + 0.5 * np.log(2 * np.pi * var)
    elif likelihood == "laplace_gp_variance":
        t = math.sqrt(2.0) * np.sqrt(r2) / np.sqrt(var) + 0.5 * np.log(2.0 * var)
    elif likelihood == "student_t":
        t = 0.5 * np.log(var) + 0.5 * (dof + 1.0) * np.log1p(r2 / (dof * var))
    else:
        raise ValueError(f"Model-error calibration needs a likelihood cost, got {likelihood!r} "
                         "('mae' has no variance term to add sigma_delta to).")
    return t + kappa * var


def estimate_block_sigma2(r, gp_var, other_var, weights=None,
                          likelihood: str = "gaussian_gp_variance", dof: float = 4.0,
                          var_floor: float = 1e-6, kappa: float = 0.0) -> Dict[str, float]:
    """sigma2_delta for one variable x block, given residuals at the optimum.

    r, gp_var, other_var : 1-D over the block's features (normalized units).
    other_var is everything else already in V (obs error).

    Returns
      'mle'    : argmin over s >= 0 of the block's weighted-mean NLL with
                 V = max(gp_var + other_var + s, var_floor), using the same
                 per-location term as the optimizer's likelihood. For the
                 Student-t cost the variance argument is the squared scale,
                 so this is on the same footing as gp_var there.
      'moment' : max(0, wmean(r^2 - gp_var - other_var)) -- the Gaussian
                 method-of-moments value, logged for comparison.
    """
    r = np.asarray(r, dtype=float)
    gv = np.asarray(gp_var, dtype=float)
    ov = np.zeros_like(r) if other_var is None else np.asarray(other_var, dtype=float)
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    r2 = r ** 2

    moment = max(0.0, float(np.sum(w * (r2 - gv - ov))))

    def f(s):
        return float(np.sum(w * _term(likelihood, r2, np.maximum(gv + ov + s, var_floor),
                                      dof, kappa)))

    # 1-D and cheap: log-spaced grid, then golden-section refinement around the
    # best grid point. The NLL in s can have a flat region near 0, so s = 0 is
    # always a candidate.
    hi = max(10.0 * float(np.max(r2)), 10.0 * float(np.max(gv + ov)), 1e-3)
    grid = np.concatenate([[0.0], np.geomspace(1e-8, hi, 400)])
    vals = np.array([f(s) for s in grid])
    i = int(np.argmin(vals))
    lo_s, hi_s = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if hi_s > lo_s:
        g = (math.sqrt(5) - 1) / 2
        a, b = lo_s, hi_s
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(80):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        s_best = (a + b) / 2
        mle = s_best if f(s_best) <= vals[i] else float(grid[i])
    else:
        mle = float(grid[i])
    return {"mle": float(mle), "moment": moment}
