"""Model-error variance: layout, units, likelihood consistency, estimator."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from autotune_gp.backend import get_backend
from autotune_gp.cost import (zrg_cost_function_gaussian_nll, zrg_cost_function_laplace,
                              zrg_cost_function_student_t)
from autotune_gp.model_error import (_term, block_indices, estimate_block_sigma2,
                                     extra_variance)


class _Sc:
    def __init__(self, scale):
        self.scale_ = np.asarray(scale, dtype=float)


N_Z, N_R, N_S = 3, 2, 1          # 6 features: z z z r r g
VARS = ["A", "B"]
SCALERS = [_Sc([1, 2, 4, 1, 2, 10]), _Sc(np.ones(6))]


def test_unset_returns_none():
    assert extra_variance(None, None, VARS, SCALERS, N_Z, N_R, N_S) is None
    assert extra_variance({}, {}, VARS, SCALERS, N_Z, N_R, N_S) is None


def test_block_layout_two_snapshots():
    idx = block_indices(N_Z, N_R, 2)
    assert idx["zonal"].tolist() == [0, 1, 2, 6, 7, 8]
    assert idx["regional"].tolist() == [3, 4, 9, 10]
    assert idx["global"].tolist() == [5, 11]


def test_units_and_blocks():
    me = {"A": {"zonal": 0.5, "global_phys": 5.0}}
    oe = {"A": {"regional": 2.0}, "B": {"zonal": 1.0}}
    ev = extra_variance(me, oe, VARS, SCALERS, N_Z, N_R, N_S)[0]
    # A: zonal normalized 0.5^2; regional obs 2/scale; global 5/10
    np.testing.assert_allclose(ev[:, 0], [0.25, 0.25, 0.25, 4.0, 1.0, 0.25])
    np.testing.assert_allclose(ev[:, 1], [1, 1, 1, 0, 0, 0])


def test_bad_keys_raise():
    with pytest.raises(ValueError):
        extra_variance({"C": {"zonal": 1}}, None, VARS, SCALERS, N_Z, N_R, N_S)
    with pytest.raises(ValueError):
        extra_variance({"A": {"zonal": 1, "zonal_phys": 1}}, None, VARS, SCALERS, N_Z, N_R, N_S)
    with pytest.raises(ValueError):
        extra_variance({"A": {"zonl": 1}}, None, VARS, SCALERS, N_Z, N_R, N_S)


@pytest.mark.parametrize("lik,fn,kw", [
    ("gaussian_gp_variance", zrg_cost_function_gaussian_nll, {}),
    ("laplace_gp_variance", zrg_cost_function_laplace, {}),
    ("student_t", zrg_cost_function_student_t, {"dof": 4.0}),
])
def test_term_matches_cost(lik, fn, kw):
    """The estimator's per-location term must equal the optimizer's (up to a constant)."""
    rng = np.random.default_rng(0)
    # two variable columns (the cost squeezes, so one would collapse); only A weighted
    preds = rng.normal(size=(1, 6, 2)); obs = rng.normal(size=(1, 6, 2))
    v1 = rng.uniform(0.1, 1, size=(1, 6, 2)); v2 = v1 + 0.3
    args = dict(var_w={"A": 1.0}, zrg_w={"zonal": 1.0, "regional": 1.0, "global": 1.0},
                dy_w={"ANN": 1.0}, n_zonal=N_Z, n_regions=N_R, backend=get_backend("numpy", "cpu"),
                var_names=["A", "B"], **kw)
    c = lambda v: fn(preds, v, obs, **args)

    def manual(v):
        r2 = ((preds - obs) ** 2)[0, :, 0]; vv = v[0, :, 0]
        t = _term(lik, r2, vv, 4.0, 0.0)
        return t[:3].mean() + t[3:5].mean() + t[5]
    assert c(v2) - c(v1) == pytest.approx(manual(v2) - manual(v1), rel=1e-10)


def test_estimator_recovers_sigma():
    rng = np.random.default_rng(1)
    n, gp, true = 4000, 0.2, 0.5
    r = rng.normal(scale=np.sqrt(gp + true), size=n)
    est = estimate_block_sigma2(r, np.full(n, gp), np.zeros(n))
    assert est["mle"] == pytest.approx(true, rel=0.08)
    assert est["moment"] == pytest.approx(true, rel=0.08)


def test_estimator_zero_when_gp_explains_residuals():
    r = np.array([0.1, -0.1, 0.05])
    est = estimate_block_sigma2(r, np.full(3, 1.0), np.zeros(3))
    assert est["mle"] == 0.0 and est["moment"] == 0.0


def test_estimator_student_t_and_weights():
    r = np.array([1.0, 1.0, 1.0, 5.0])
    w = np.array([1.0, 1.0, 1.0, 0.0])            # zero weight hides the outlier
    est = estimate_block_sigma2(r, np.full(4, 0.1), None, weights=w, likelihood="student_t")
    # Student-t with all-equal |r|=1: MLE scale^2 is r^2 (nu+1)/nu... minus gp -> finite, > 0
    assert 0 < est["mle"] < 5
    with pytest.raises(ValueError):
        estimate_block_sigma2(r, np.ones(4), None, likelihood="mae")
