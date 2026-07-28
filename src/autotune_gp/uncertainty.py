"""
Post-hoc parameter-uncertainty reporting for the optimizer's best-found point.

Only statistically meaningful when the cost function passed in is a genuine
negative-log-likelihood (optimize.likelihood='gaussian_gp_variance' — see
autotune_gp.cost.zrg_cost_function_gaussian_nll), not the plain weighted-MAE cost.
Even then, because that cost aggregates via a per-block weighted mean rather than
summing over every individual (spatially correlated) location, the resulting
covariance should be read as a relative/order-of-magnitude uncertainty indicator,
not a calibrated absolute confidence interval.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def numerical_hessian(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    eps: float = 1e-4,
    bounds_low: Optional[np.ndarray] = None,
    bounds_high: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Central finite-difference Hessian of a scalar function `f` at `x0`.

    Diagonal terms use the standard 3-point stencil; off-diagonal terms use the
    symmetric 4-point stencil, generalized to per-dimension step sizes. No GP
    dependency — `f` can be any scalar-valued callable of a 1-D array, which
    makes this independently testable against a hand-built quadratic with a
    known analytic Hessian.

    bounds_low, bounds_high — if given, the step size for each dimension is
    shrunk (not just clipped) so that x0[i] +/- step always stays within
    [bounds_low[i], bounds_high[i]], preserving a valid symmetric difference
    instead of ever evaluating `f` at an out-of-domain point. Without this, a
    parameter close enough to its bound that x0[i] - eps < bounds_low[i] (or
    the +eps analog) would query `f` at a meaningless, never-trained point.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.shape[0]

    if bounds_low is None:
        eps_vec = np.full(n, eps)
    else:
        bounds_low  = np.asarray(bounds_low, dtype=float)
        bounds_high = np.asarray(bounds_high, dtype=float)
        room = np.minimum(x0 - bounds_low, bounds_high - x0)
        eps_vec = np.clip(np.minimum(eps, room), 1e-10, None)

    H = np.zeros((n, n))
    f0 = float(f(x0))

    fp = np.empty(n)
    fm = np.empty(n)
    for i in range(n):
        xp = x0.copy(); xp[i] += eps_vec[i]
        xm = x0.copy(); xm[i] -= eps_vec[i]
        fp[i] = float(f(xp))
        fm[i] = float(f(xm))
        H[i, i] = (fp[i] - 2.0 * f0 + fm[i]) / eps_vec[i] ** 2

    for i in range(n):
        for j in range(i + 1, n):
            xpp = x0.copy(); xpp[i] += eps_vec[i]; xpp[j] += eps_vec[j]
            xpm = x0.copy(); xpm[i] += eps_vec[i]; xpm[j] -= eps_vec[j]
            xmp = x0.copy(); xmp[i] -= eps_vec[i]; xmp[j] += eps_vec[j]
            xmm = x0.copy(); xmm[i] -= eps_vec[i]; xmm[j] -= eps_vec[j]
            h_ij = ((float(f(xpp)) - float(f(xpm)) - float(f(xmp)) + float(f(xmm)))
                    / (4.0 * eps_vec[i] * eps_vec[j]))
            H[i, j] = h_ij
            H[j, i] = h_ij

    return H


def parameter_uncertainty(
    cost_fn: Callable[[np.ndarray], float],
    x_best: np.ndarray,
    param_names: Optional[List[str]] = None,
    eps: float = 1e-4,
    likelihood: str = "mae",
    bounds_low: Optional[np.ndarray] = None,
    bounds_high: Optional[np.ndarray] = None,
    boundary_tol: float = 1e-3,
) -> dict:
    """
    Approximate parameter covariance at the optimizer's best-found point, via
    the Laplace approximation: cov ≈ inverse Hessian of `cost_fn` at `x_best`.

    `cost_fn` must take the same dimensionality `x_best` does — in particular,
    for the multi-fidelity subset-optimization path (--vary-params), `cost_fn`
    already operates on the reduced (varied-only) parameter vector internally,
    so `x_best`/`param_names` here should be in that same reduced dimensionality
    (no expansion to the full parameter vector is needed or wanted).

    bounds_low, bounds_high — the same box bounds passed to optimize_parallel.
    When given: (a) the Hessian is computed with bounds-aware step sizes (see
    numerical_hessian), so it never evaluates cost_fn outside the feasible box;
    and (b) any parameter within `boundary_tol` of a bound is excluded from the
    matrix inversion entirely. A parameter pinned at a box constraint is not a
    stationary point of the *unconstrained* cost function — L-BFGS-B stopped it
    there because the wall blocked further improvement, not because the
    gradient vanished — so there is no reason to expect valid unconstrained
    curvature there. Folding it into the same inversion as the free parameters
    would corrupt the entire result (matrix inversion mixes every row/column
    together), not just that parameter's own entry, which is why the previous
    behavior returned NaN for *every* parameter whenever *any* one was pinned.
    Excluded parameters are still reported with stderr=NaN; the rest get a real
    standard error computed from the submatrix of free parameters only.

    Prints a summary table and returns a dict (for pickling) with keys:
    x, hessian, cov, stderr, corr, param_names, at_bound.
    """
    x_best = np.asarray(x_best, dtype=float)
    n = x_best.shape[0]
    if param_names is None:
        param_names = [f"param{i}" for i in range(n)]
    if len(param_names) != n:
        raise ValueError(f"param_names has {len(param_names)} entries but x_best has {n}")

    H = numerical_hessian(cost_fn, x_best, eps=eps, bounds_low=bounds_low, bounds_high=bounds_high)

    if bounds_low is None:
        at_bound = np.zeros(n, dtype=bool)
    else:
        bl = np.asarray(bounds_low, dtype=float)
        bh = np.asarray(bounds_high, dtype=float)
        at_bound = ((x_best - bl) < boundary_tol) | ((bh - x_best) < boundary_tol)

    free = np.where(~at_bound)[0]
    cov    = np.full((n, n), np.nan)
    stderr = np.full(n, np.nan)
    corr   = np.full((n, n), np.nan)

    invertible = free.size > 0
    if invertible:
        H_free = H[np.ix_(free, free)]
        try:
            cov_free = np.linalg.inv(H_free)
            if np.any(np.diag(cov_free) <= 0) or not np.all(np.isfinite(cov_free)):
                invertible = False
        except np.linalg.LinAlgError:
            invertible = False
        if invertible:
            se = np.sqrt(np.diag(cov_free))
            cov[np.ix_(free, free)]    = cov_free
            stderr[free]               = se
            corr[np.ix_(free, free)]   = cov_free / np.outer(se, se)

    if at_bound.any():
        pinned = [param_names[i] for i in range(n) if at_bound[i]]
        print(f"  {len(pinned)} parameter(s) within {boundary_tol} of a bound — excluded from "
              f"the Hessian inversion (no meaningful unconstrained curvature there): "
              f"{', '.join(pinned)}")
    if not invertible:
        print("  Warning: Hessian over the free (non-boundary) parameters is not invertible / "
              "not positive-definite (too coarse an `eps`, a genuinely flat/degenerate "
              "direction, or every parameter is pinned at a bound) — parameter uncertainty is "
              "undefined for them too; returning NaNs.")

    name_w = max(len(nm) for nm in param_names)
    print(f"\n  {'Parameter':<{name_w}}  {'value':>10}  {'stderr':>10}")
    print("  " + "-" * (name_w + 26))
    for i, name in enumerate(param_names):
        print(f"  {name:<{name_w}}  {x_best[i]:10.4f}  {stderr[i]:10.4f}")

    if likelihood == "mae":
        print("\n  Warning: likelihood='mae' — this cost is not a real log-likelihood, "
              "so the covariance above has no statistical meaning. Set "
              "optimize.likelihood='gaussian_gp_variance' for a meaningful report.")
    else:
        print("\n  Note: this cost aggregates via a per-block weighted MEAN rather than "
              "a sum over every individual location (see "
              "zrg_cost_function_gaussian_nll's docstring) — treat the stderr/correlations "
              "above as relative/order-of-magnitude uncertainty indicators, not calibrated "
              "absolute confidence intervals.")

    return {
        "x": x_best,
        "hessian": H,
        "cov": cov,
        "stderr": stderr,
        "corr": corr,
        "param_names": param_names,
        "at_bound": at_bound,
    }
