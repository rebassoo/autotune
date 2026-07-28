"""
Tests for autotune_gp.uncertainty (numerical_hessian, parameter_uncertainty).

No GP training, config object, or real data required — mirrors the no-GP,
analytic-cost-function style used by TestOptimizeParallel in test_optimization.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.uncertainty import numerical_hessian, parameter_uncertainty


# ---------------------------------------------------------------------------
# numerical_hessian
# ---------------------------------------------------------------------------

class TestNumericalHessian:

    def test_matches_analytic_hessian_of_quadratic(self):
        """f(x) = x^T A x has Hessian 2A exactly (up to finite-difference error)."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])   # symmetric

        def f(x):
            x = np.asarray(x, dtype=float)
            return float(x @ A @ x)

        H = numerical_hessian(f, np.array([0.7, -0.3]), eps=1e-3)
        assert H == pytest.approx(2 * A, abs=1e-3)

    def test_hessian_is_symmetric(self):
        rng = np.random.RandomState(0)
        M = rng.randn(3, 3)
        A = M @ M.T   # SPD, but not passed in symmetrized form to f

        def f(x):
            x = np.asarray(x, dtype=float)
            return float(x @ A @ x)

        H = numerical_hessian(f, np.array([1.0, 0.5, -1.2]), eps=1e-3)
        assert H == pytest.approx(H.T, abs=1e-8)

    def test_zero_hessian_for_linear_function(self):
        a = np.array([2.0, -1.0, 0.5])

        def f(x):
            return float(np.dot(a, x))

        H = numerical_hessian(f, np.zeros(3), eps=1e-3)
        assert H == pytest.approx(np.zeros((3, 3)), abs=1e-6)

    def test_never_evaluates_outside_bounds(self):
        """A parameter closer to its bound than `eps` must not push f out of
        [bounds_low, bounds_high] — this is the bug being fixed: previously
        x0[i] - eps could go negative for x0[i] << eps."""
        bounds_low  = np.array([0.0, 0.0])
        bounds_high = np.array([1.0, 1.0])

        def f(x):
            x = np.asarray(x, dtype=float)
            assert np.all(x >= bounds_low - 1e-12), f"evaluated below lower bound: {x}"
            assert np.all(x <= bounds_high + 1e-12), f"evaluated above upper bound: {x}"
            return float(np.sum(x ** 2))

        x0 = np.array([1e-6, 0.5])   # first dim far closer to 0 than eps=1e-4
        H = numerical_hessian(f, x0, eps=1e-4, bounds_low=bounds_low, bounds_high=bounds_high)
        assert np.all(np.isfinite(H))

    def test_bounds_none_matches_unbounded_behavior(self):
        """Passing bounds_low=None (the default) reproduces the original
        fixed-eps behavior exactly, for a point far from any bound."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])

        def f(x):
            x = np.asarray(x, dtype=float)
            return float(x @ A @ x)

        x0 = np.array([0.7, -0.3])
        H_unbounded = numerical_hessian(f, x0, eps=1e-3)
        H_bounded   = numerical_hessian(f, x0, eps=1e-3,
                                        bounds_low=np.array([-10.0, -10.0]),
                                        bounds_high=np.array([10.0, 10.0]))
        assert H_bounded == pytest.approx(H_unbounded, abs=1e-8)


# ---------------------------------------------------------------------------
# parameter_uncertainty
# ---------------------------------------------------------------------------

class TestParameterUncertainty:

    def test_recovers_sensible_covariance_for_gaussian_nll_shaped_cost(self):
        """
        cost(x) = 0.5*(x-mu)^2/sigma^2 is a per-parameter Gaussian NLL (up to an
        additive constant) — its inverse Hessian should recover sigma^2 exactly.
        """
        mu, sigma = np.array([1.0, -2.0]), np.array([0.5, 2.0])

        def cost_fn(x):
            x = np.asarray(x, dtype=float)
            return float(np.sum(0.5 * (x - mu) ** 2 / sigma ** 2))

        result = parameter_uncertainty(cost_fn, x_best=mu, param_names=["a", "b"],
                                        eps=1e-3, likelihood="gaussian_gp_variance")
        assert result["stderr"] == pytest.approx(sigma, rel=1e-2)
        assert result["param_names"] == ["a", "b"]

    def test_degenerate_hessian_returns_nan_without_raising(self):
        """A perfectly flat cost has a zero Hessian (non-invertible) — should
        warn and return NaNs, not crash."""
        def cost_fn(x):
            return 0.0

        result = parameter_uncertainty(cost_fn, x_best=np.array([0.0, 0.0]),
                                        eps=1e-3, likelihood="gaussian_gp_variance")
        assert np.all(np.isnan(result["stderr"]))
        assert np.all(np.isnan(result["cov"]))

    def test_default_param_names_when_not_given(self):
        def cost_fn(x):
            return float(np.sum(np.asarray(x, dtype=float) ** 2))

        result = parameter_uncertainty(cost_fn, x_best=np.array([0.1, 0.2, 0.3]), eps=1e-3)
        assert result["param_names"] == ["param0", "param1", "param2"]

    def test_raises_on_param_names_length_mismatch(self):
        def cost_fn(x):
            return float(np.sum(np.asarray(x, dtype=float) ** 2))

        with pytest.raises(ValueError, match="param_names"):
            parameter_uncertainty(cost_fn, x_best=np.array([0.0, 0.0]), param_names=["only_one"])

    def test_boundary_parameter_does_not_poison_free_parameters(self):
        """
        Regression test for the bug this fix addresses: one parameter pinned at
        (or within eps of) a box bound used to make the *entire* covariance
        matrix collapse to NaN, even for perfectly well-behaved free parameters.
        With bounds_low/bounds_high supplied, the pinned parameter should be
        excluded and reported as NaN, but the free ones should get real,
        accurate standard errors.
        """
        mu, sigma = np.array([1.0, -2.0, 0.3]), np.array([0.5, 2.0, 0.1])

        def cost_fn(x):
            x = np.asarray(x, dtype=float)
            return float(np.sum(0.5 * (x - mu) ** 2 / sigma ** 2))

        # param 2's best-fit value is pinned essentially at its lower bound 0.0,
        # and closer to it than the default eps=1e-4 would be.
        x_best = np.array([1.0, -2.0, 1e-6])
        bounds_low  = np.array([-10.0, -10.0, 0.0])
        bounds_high = np.array([10.0, 10.0, 1.0])

        result = parameter_uncertainty(
            cost_fn, x_best=x_best, param_names=["a", "b", "c"],
            eps=1e-4, likelihood="gaussian_gp_variance",
            bounds_low=bounds_low, bounds_high=bounds_high, boundary_tol=1e-3,
        )

        assert list(result["at_bound"]) == [False, False, True]
        # the two free parameters recover their true sigma, not NaN
        assert result["stderr"][0] == pytest.approx(sigma[0], rel=1e-2)
        assert result["stderr"][1] == pytest.approx(sigma[1], rel=1e-2)
        # the pinned parameter is honestly reported as undefined, not a fabricated number
        assert np.isnan(result["stderr"][2])
        assert np.isfinite(result["hessian"]).all()  # bounds-aware step never blew up

    def test_no_bounds_falls_back_to_original_all_or_nothing_behavior(self):
        """Without bounds_low/bounds_high, behavior is unchanged from before this
        fix: at_bound is all False and a degenerate Hessian NaNs out everything."""
        def cost_fn(x):
            return 0.0

        result = parameter_uncertainty(cost_fn, x_best=np.array([0.0, 0.0]),
                                        eps=1e-3, likelihood="gaussian_gp_variance")
        assert not result["at_bound"].any()
        assert np.all(np.isnan(result["stderr"]))
