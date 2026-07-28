"""
Tests for the optimization pipeline modules:
  - autotune_gp.transforms  (fit_transform_X, fit_transform_Y, transform_obs)
  - autotune_gp.io           (split_zrg_obs)
  - autotune_gp.cost         (zrg_cost_function_mae_weighted)
  - autotune_gp.optimize     (optimize_parallel)

No GP training or real data required.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.backend import get_backend
from autotune_gp.transforms import fit_transform_X, fit_transform_Y, transform_obs
from autotune_gp.io import split_zrg_obs
from autotune_gp.cost import (
    zrg_cost_function_mae_weighted, zrg_cost_function_gaussian_nll,
    build_training_coverage_tree, training_distance_penalty,
)
from autotune_gp.optimize import optimize_parallel


# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

N_ZONAL   = 4
N_REGIONS = 3
N_VARS    = 4
N_FEAT    = 2 * (N_ZONAL + N_REGIONS + 1)   # 16

VAR_W  = {"PCP": 0.25, "TLWP": 0.25, "OSR": 0.25, "OLR": 0.25}
ZRG_W  = {"zonal": 1/3, "regional": 1/3, "global": 1/3}
DY_W   = {"DY1": 0.5,  "DY2": 0.5}

BACKEND = get_backend("numpy", "cpu")


def _uniform_arrays(value=0.0):
    """Return (preds, obs) of shape (1, N_FEAT, N_VARS) filled with value."""
    return (
        np.full((1, N_FEAT, N_VARS), value, dtype=float),
        np.full((1, N_FEAT, N_VARS), value, dtype=float),
    )


# ---------------------------------------------------------------------------
# transforms.fit_transform_X
# ---------------------------------------------------------------------------

class TestFitTransformX:

    def test_lower_bound_maps_to_zero(self):
        bounds = np.array([[0.0, 10.0], [1.0, 20.0]])   # shape (2, n_params): [lows, highs]
        X = np.array([[0.0, 10.0]])   # lower bound values
        _, X_norm = fit_transform_X(X, param_bounds=bounds)
        assert X_norm == pytest.approx(np.zeros((1, 2)), abs=1e-10)

    def test_upper_bound_maps_to_one(self):
        bounds = np.array([[0.0, 10.0], [1.0, 20.0]])
        X = np.array([[1.0, 20.0]])   # upper bound values
        _, X_norm = fit_transform_X(X, param_bounds=bounds)
        assert X_norm == pytest.approx(np.ones((1, 2)), abs=1e-10)

    def test_midpoint_maps_to_half(self):
        bounds = np.array([[0.0, 0.0], [2.0, 10.0]])    # lows=[0,0], highs=[2,10]
        X = np.array([[1.0, 5.0]])
        _, X_norm = fit_transform_X(X, param_bounds=bounds)
        assert X_norm == pytest.approx(np.array([[0.5, 0.5]]), abs=1e-10)

    def test_fallback_to_data_range(self):
        X = np.array([[0.0, 10.0], [1.0, 20.0]])
        _, X_norm = fit_transform_X(X)   # no param_bounds
        # min of each column → 0, max → 1
        assert X_norm[0] == pytest.approx([0.0, 0.0], abs=1e-10)
        assert X_norm[1] == pytest.approx([1.0, 1.0], abs=1e-10)

    def test_returns_scaler_and_array(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        sc, X_norm = fit_transform_X(X)
        assert hasattr(sc, "transform"), "First return value should be a scaler"
        assert X_norm.shape == X.shape


# ---------------------------------------------------------------------------
# transforms.fit_transform_Y  /  transform_obs
# ---------------------------------------------------------------------------

class TestFitTransformY:
    N_TRAIN = 20
    N_FEAT  = 10
    N_VARS  = 4

    def _make_Y(self, rng_seed=0):
        rng = np.random.RandomState(rng_seed)
        return rng.randn(self.N_TRAIN, self.N_FEAT, self.N_VARS) * 5 + 3

    def test_output_shape_preserved(self):
        Y = self._make_Y()
        _, Y_norm = fit_transform_Y(Y)
        assert Y_norm.shape == Y.shape

    def test_zero_mean_per_variable(self):
        Y = self._make_Y()
        _, Y_norm = fit_transform_Y(Y)
        for j in range(self.N_VARS):
            col_mean = Y_norm[:, :, j].mean()
            assert col_mean == pytest.approx(0.0, abs=1e-10), \
                f"Variable {j} mean after scaling should be 0, got {col_mean}"

    def test_unit_variance_per_variable(self):
        Y = self._make_Y()
        _, Y_norm = fit_transform_Y(Y)
        for j in range(self.N_VARS):
            col_std = Y_norm[:, :, j].std()
            assert col_std == pytest.approx(1.0, abs=1e-6), \
                f"Variable {j} std after scaling should be ~1, got {col_std}"

    def test_returns_correct_number_of_scalers(self):
        Y = self._make_Y()
        scalers, _ = fit_transform_Y(Y)
        assert len(scalers) == self.N_VARS

    def test_transform_obs_uses_same_scale_as_Y(self):
        """Training mean should transform to 0 when using the fitted scalers."""
        Y = self._make_Y()
        scalers, _ = fit_transform_Y(Y)
        # Build obs equal to the per-variable training mean
        obs_parts = []
        for j in range(self.N_VARS):
            mean_val = Y[:, :, j].mean(axis=0, keepdims=True)  # (1, N_FEAT)
            obs_parts.append(pd.DataFrame(mean_val))
        obs_norm = transform_obs(obs_parts, scalers)
        assert obs_norm == pytest.approx(
            np.zeros((1, self.N_FEAT, self.N_VARS)), abs=1e-6
        )

    def test_transform_obs_output_shape(self):
        Y = self._make_Y()
        scalers, _ = fit_transform_Y(Y)
        obs_parts = [pd.DataFrame(np.zeros((1, self.N_FEAT))) for _ in range(self.N_VARS)]
        obs_norm = transform_obs(obs_parts, scalers)
        assert obs_norm.shape == (1, self.N_FEAT, self.N_VARS)


# ---------------------------------------------------------------------------
# io.split_zrg_obs
# ---------------------------------------------------------------------------

class TestSplitZrgObs:
    N_VARS   = 4
    N_FEAT   = 16   # per variable
    N_COLS   = N_VARS * N_FEAT

    def _make_df(self):
        return pd.DataFrame(
            np.arange(self.N_COLS, dtype=float).reshape(1, -1)
        )

    def test_returns_n_vars_parts(self):
        parts = split_zrg_obs(self._make_df(), n_vars=self.N_VARS)
        assert len(parts) == self.N_VARS

    def test_each_part_has_correct_width(self):
        parts = split_zrg_obs(self._make_df(), n_vars=self.N_VARS)
        for i, part in enumerate(parts):
            assert part.shape[1] == self.N_FEAT, \
                f"Part {i} has {part.shape[1]} columns, expected {self.N_FEAT}"

    def test_columns_are_strings(self):
        parts = split_zrg_obs(self._make_df(), n_vars=self.N_VARS)
        for i, part in enumerate(parts):
            assert all(isinstance(c, str) for c in part.columns), \
                f"Part {i} columns are not all strings: {list(part.columns)}"

    def test_values_are_contiguous_blocks(self):
        parts = split_zrg_obs(self._make_df(), n_vars=self.N_VARS)
        for i, part in enumerate(parts):
            expected = np.arange(i * self.N_FEAT, (i + 1) * self.N_FEAT, dtype=float)
            assert part.values[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# cost.zrg_cost_function_mae_weighted
# ---------------------------------------------------------------------------

class TestCostMAEWeighted:

    def _call(self, preds, obs, dy_w=None, zonal_weights=None, regional_weights=None):
        return zrg_cost_function_mae_weighted(
            preds, obs, VAR_W, ZRG_W, dy_w or DY_W,
            n_zonal=N_ZONAL, n_regions=N_REGIONS,
            backend=BACKEND,
            zonal_weights=zonal_weights,
            regional_weights=regional_weights,
        )

    def test_zero_cost_when_preds_equal_obs(self):
        preds, obs = _uniform_arrays(1.0)
        assert self._call(preds, obs) == pytest.approx(0.0, abs=1e-12)

    def test_positive_cost_for_nonzero_error(self):
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.ones((1, N_FEAT, N_VARS))
        assert self._call(preds, obs) > 0.0

    def test_uniform_offset_gives_expected_cost(self):
        """
        With equal var/zrg/dy weights and a uniform offset delta=1,
        cost should equal 1.0 analytically:
          each MAE = 1 → w_mae = 1 → zrg total per snapshot = 1 → dy-weighted = 1
        """
        delta = 1.0
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.full((1, N_FEAT, N_VARS), delta)
        cost = self._call(preds, obs)
        assert cost == pytest.approx(delta, rel=1e-6)

    def test_cost_scales_linearly_with_offset(self):
        for delta in [0.5, 2.0, 5.0]:
            preds = np.zeros((1, N_FEAT, N_VARS))
            obs   = np.full((1, N_FEAT, N_VARS), delta)
            cost = self._call(preds, obs)
            assert cost == pytest.approx(delta, rel=1e-6), \
                f"Expected cost={delta} for uniform offset={delta}, got {cost}"

    def test_dy_weight_asymmetry(self):
        """Putting all weight on DY1 vs DY2 should give the same result when
        DY1 and DY2 errors are identical, but differ when they are not."""
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.zeros((1, N_FEAT, N_VARS))
        # Set DY2 half (second half of features) to have a large error
        n_per = N_ZONAL + N_REGIONS + 1
        obs[:, n_per:, :] = 2.0

        cost_dy1_heavy = self._call(preds, obs, dy_w={"DY1": 0.9, "DY2": 0.1})
        cost_dy2_heavy = self._call(preds, obs, dy_w={"DY1": 0.1, "DY2": 0.9})
        assert cost_dy2_heavy > cost_dy1_heavy

    def test_raises_on_wrong_n_obs(self):
        preds = np.zeros((2, N_FEAT, N_VARS))   # n_obs=2 is invalid
        obs   = np.zeros((2, N_FEAT, N_VARS))
        with pytest.raises(ValueError, match="n_obs=1"):
            self._call(preds, obs)

    def test_raises_on_wrong_n_vars(self):
        preds = np.zeros((1, N_FEAT, 3))   # n_vars=3 is invalid
        obs   = np.zeros((1, N_FEAT, 3))
        with pytest.raises(ValueError, match="4"):
            self._call(preds, obs)

    def test_raises_on_wrong_n_feat(self):
        preds = np.zeros((1, N_FEAT + 1, N_VARS))
        obs   = np.zeros((1, N_FEAT + 1, N_VARS))
        with pytest.raises(ValueError, match="layout mismatch"):
            self._call(preds, obs)

    def test_finite_output(self):
        rng = np.random.RandomState(42)
        preds = rng.randn(1, N_FEAT, N_VARS)
        obs   = rng.randn(1, N_FEAT, N_VARS)
        assert np.isfinite(self._call(preds, obs))


# ---------------------------------------------------------------------------
# cost.zrg_cost_function_gaussian_nll
# ---------------------------------------------------------------------------

class TestCostGaussianNLL:

    def _call(self, preds, pred_var, obs, dy_w=None, zonal_weights=None,
              regional_weights=None, var_floor=1e-6, variance_penalty_kappa=0.0):
        return zrg_cost_function_gaussian_nll(
            preds, pred_var, obs, VAR_W, ZRG_W, dy_w or DY_W,
            n_zonal=N_ZONAL, n_regions=N_REGIONS,
            backend=BACKEND,
            zonal_weights=zonal_weights,
            regional_weights=regional_weights,
            var_floor=var_floor,
            variance_penalty_kappa=variance_penalty_kappa,
        )

    def test_uniform_offset_matches_analytic_nll(self):
        """
        With equal var/zrg/dy weights, a uniform offset `delta`, and a uniform
        predictive variance `v` everywhere, every block/variable/snapshot term
        is identical, so the weighted combination (weights each summing to 1)
        collapses to the single per-location NLL term exactly.
        """
        delta, v = 1.0, 1.0
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.full((1, N_FEAT, N_VARS), delta)
        pred_var = np.full((1, N_FEAT, N_VARS), v)
        expected = 0.5 * delta ** 2 / v + 0.5 * np.log(2 * np.pi * v)
        assert self._call(preds, pred_var, obs) == pytest.approx(expected, rel=1e-6)

    def test_not_zero_at_zero_residual(self):
        """Unlike MAE, the NLL is not zero at zero residual — the log(var) term remains."""
        preds, obs = _uniform_arrays(1.0)
        pred_var = np.full((1, N_FEAT, N_VARS), 1.0)
        cost = self._call(preds, pred_var, obs)
        assert cost == pytest.approx(0.5 * np.log(2 * np.pi), rel=1e-6)

    def test_cost_increases_with_residual_for_fixed_variance(self):
        pred_var = np.full((1, N_FEAT, N_VARS), 1.0)
        costs = []
        for delta in [0.0, 0.5, 2.0, 5.0]:
            preds = np.zeros((1, N_FEAT, N_VARS))
            obs   = np.full((1, N_FEAT, N_VARS), delta)
            costs.append(self._call(preds, pred_var, obs))
        assert costs == sorted(costs)
        assert costs[-1] > costs[0]

    def test_var_floor_prevents_blowup(self):
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.full((1, N_FEAT, N_VARS), 1.0)
        pred_var = np.zeros((1, N_FEAT, N_VARS))   # would divide by zero unclipped
        cost = self._call(preds, pred_var, obs, var_floor=1e-6)
        assert np.isfinite(cost)

    def test_raises_on_wrong_n_obs(self):
        preds = np.zeros((2, N_FEAT, N_VARS))
        obs   = np.zeros((2, N_FEAT, N_VARS))
        pred_var = np.ones((2, N_FEAT, N_VARS))
        with pytest.raises(ValueError, match="n_obs=1"):
            self._call(preds, pred_var, obs)

    def test_raises_on_pred_var_shape_mismatch(self):
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.zeros((1, N_FEAT, N_VARS))
        pred_var = np.ones((1, N_FEAT, N_VARS + 1))
        with pytest.raises(ValueError, match="pred_var has shape"):
            self._call(preds, pred_var, obs)

    def test_raises_on_wrong_n_feat(self):
        preds = np.zeros((1, N_FEAT + 1, N_VARS))
        obs   = np.zeros((1, N_FEAT + 1, N_VARS))
        pred_var = np.ones((1, N_FEAT + 1, N_VARS))
        with pytest.raises(ValueError, match="layout mismatch"):
            self._call(preds, pred_var, obs)

    def test_finite_output(self):
        rng = np.random.RandomState(42)
        preds = rng.randn(1, N_FEAT, N_VARS)
        obs   = rng.randn(1, N_FEAT, N_VARS)
        pred_var = rng.rand(1, N_FEAT, N_VARS) + 0.1   # strictly positive
        assert np.isfinite(self._call(preds, pred_var, obs))

    def test_kappa_zero_matches_original_cost(self):
        """variance_penalty_kappa=0.0 (the default) must recover the cost
        exactly as it was before this parameter existed."""
        rng = np.random.RandomState(1)
        preds = rng.randn(1, N_FEAT, N_VARS)
        obs   = rng.randn(1, N_FEAT, N_VARS)
        pred_var = rng.rand(1, N_FEAT, N_VARS) + 0.1
        cost_default = self._call(preds, pred_var, obs)
        cost_explicit_zero = self._call(preds, pred_var, obs, variance_penalty_kappa=0.0)
        assert cost_default == pytest.approx(cost_explicit_zero, rel=1e-12)

    def test_kappa_penalizes_higher_variance_at_fixed_residual(self):
        """
        This is the failure mode kappa exists to close. For a fixed residual
        r=1 (resid**2=1), the uncapped (kappa=0) cost 0.5*r**2/v + 0.5*log(v)
        is minimized at v=r**2=1 and *decreases* monotonically for any v below
        that -- so moving from v=0.5 up to v=1.0 (still under-confident
        relative to the true residual) makes the kappa=0 cost go down, i.e.
        the optimizer is rewarded for finding a point where pred_var happens
        to be larger, without the fit itself improving at all.

        kappa=1 analytically shifts that minimum down to v=0.5 (solving
        kappa = 0.5*r**2/v**2 - 0.5/v at r=1 gives v=0.5), so the same
        v: 0.5 -> 1.0 move now makes the cost go up instead -- the reward
        for landing on a higher-variance point is gone.
        """
        delta = 1.0
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.full((1, N_FEAT, N_VARS), delta)

        def cost_at(v, kappa):
            pred_var = np.full((1, N_FEAT, N_VARS), v)
            return self._call(preds, pred_var, obs, variance_penalty_kappa=kappa)

        # Without the penalty: growing pred_var from 0.5 to 1.0 still lowers cost.
        assert cost_at(1.0, kappa=0.0) < cost_at(0.5, kappa=0.0)

        # With kappa=1: the same move now raises cost instead.
        assert cost_at(1.0, kappa=1.0) > cost_at(0.5, kappa=1.0)

    def test_kappa_negative_or_zero_never_worse_than_larger_kappa_at_high_variance(self):
        """Monotonicity sanity check: at a fixed (residual, variance) point,
        increasing kappa can only increase (never decrease) the cost, since
        the added term kappa*var_clipped >= 0 for kappa >= 0."""
        preds = np.zeros((1, N_FEAT, N_VARS))
        obs   = np.full((1, N_FEAT, N_VARS), 1.0)
        pred_var = np.full((1, N_FEAT, N_VARS), 5.0)
        costs = [self._call(preds, pred_var, obs, variance_penalty_kappa=k)
                 for k in (0.0, 0.1, 1.0, 10.0)]
        assert costs == sorted(costs)


# ---------------------------------------------------------------------------
# cost.build_training_coverage_tree / cost.training_distance_penalty
# ---------------------------------------------------------------------------

class TestTrainingDistancePenalty:

    def _grid_training_set(self):
        """A simple 2D grid of 'training points' at every combination of
        {0, 0.5, 1} x {0, 0.5, 1}, with bounds_low=0, bounds_high=1 (i.e.
        already in rescaled units) -- makes the expected nearest-neighbor
        geometry easy to reason about by hand."""
        xs, ys = np.meshgrid([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        X = np.stack([xs.ravel(), ys.ravel()], axis=1)
        bounds_low, bounds_high = [0.0, 0.0], [1.0, 1.0]
        return X, bounds_low, bounds_high

    def test_threshold_is_grids_own_max_internal_spacing(self):
        X, bl, bh = self._grid_training_set()
        _, threshold = build_training_coverage_tree([X], bl, bh)
        # every grid point's nearest OTHER grid point is 0.5 away (a neighbor
        # one step over on the grid), so the training set's own max internal
        # spacing should be exactly 0.5.
        assert threshold == pytest.approx(0.5, abs=1e-9)

    def test_lambda_zero_is_always_zero_without_querying_tree(self):
        X, bl, bh = self._grid_training_set()
        tree, threshold = build_training_coverage_tree([X], bl, bh)
        far_point = np.array([100.0, 100.0])   # would be enormous if queried
        assert training_distance_penalty(far_point, tree, threshold, 0.0, bl, bh) == 0.0

    def test_zero_penalty_inside_training_envelope(self):
        X, bl, bh = self._grid_training_set()
        tree, threshold = build_training_coverage_tree([X], bl, bh)
        # sits exactly on a training point -> distance 0, well under threshold
        on_point = np.array([0.5, 0.5])
        assert training_distance_penalty(on_point, tree, threshold, 1.0, bl, bh) == 0.0

    def test_positive_penalty_beyond_threshold(self):
        X, bl, bh = self._grid_training_set()
        tree, threshold = build_training_coverage_tree([X], bl, bh)
        # far outside the grid entirely
        far_point = np.array([5.0, 5.0])
        penalty = training_distance_penalty(far_point, tree, threshold, 1.0, bl, bh)
        assert penalty > 0.0

    def test_penalty_scales_linearly_with_lambda(self):
        X, bl, bh = self._grid_training_set()
        tree, threshold = build_training_coverage_tree([X], bl, bh)
        far_point = np.array([5.0, 5.0])
        p1 = training_distance_penalty(far_point, tree, threshold, 1.0, bl, bh)
        p2 = training_distance_penalty(far_point, tree, threshold, 2.0, bl, bh)
        assert p2 == pytest.approx(2 * p1, rel=1e-9)

    def test_rescaling_by_per_dimension_bounds(self):
        """A dimension with a much narrower [bounds_low, bounds_high] should
        contribute proportionally more to the rescaled distance for the same
        raw offset, exactly like the bounds_from_data convention used
        elsewhere in this project. Uses a hand-built single-point tree and
        threshold=0.0 directly, isolating the rescale arithmetic from
        build_training_coverage_tree's threshold selection (covered by
        test_threshold_is_grids_own_max_internal_spacing above)."""
        from scipy.spatial import cKDTree
        tree = cKDTree(np.array([[0.0, 0.0]]))   # one already-rescaled point, at the origin
        threshold = 0.0
        bl, bh = [0.0, 0.0], [1.0, 0.1]   # 2nd dim's sampled range is 10x narrower
        # offset by 0.1 in dim 0 (10% of its range) vs 0.1 in dim 1 (100% of its range)
        p_dim0 = training_distance_penalty(np.array([0.1, 0.0]), tree, threshold, 1.0, bl, bh)
        p_dim1 = training_distance_penalty(np.array([0.0, 0.1]), tree, threshold, 1.0, bl, bh)
        assert p_dim1 > p_dim0 > 0.0


# ---------------------------------------------------------------------------
# optimize.optimize_parallel
# ---------------------------------------------------------------------------

class TestOptimizeParallel:

    def test_finds_minimum_of_quadratic(self, tmp_path):
        """x^2 on [0,1] has its minimum at x=0. Each start should land near 0."""
        def cost_fn(x):
            return float(np.sum(np.asarray(x) ** 2))

        results, top_rows, csv_path = optimize_parallel(
            cost_fn=cost_fn,
            n_params=2,
            bounds_low=0.0,
            bounds_high=1.0,
            seed=0,
            n_xstarts=3,
            niter=5,
            method="L-BFGS-B",
            out_dir=str(tmp_path),
            max_workers=1,
        )
        best_params = results[top_rows[0], :-1]
        best_cost   = results[top_rows[0], -1]
        assert best_cost == pytest.approx(0.0, abs=1e-4)
        assert best_params == pytest.approx([0.0, 0.0], abs=1e-2)

    def test_csv_is_written(self, tmp_path):
        def cost_fn(x):
            return float(np.sum(np.asarray(x) ** 2))

        _, _, csv_path = optimize_parallel(
            cost_fn=cost_fn,
            n_params=2,
            bounds_low=0.0,
            bounds_high=1.0,
            seed=0,
            n_xstarts=2,
            niter=1,
            method="L-BFGS-B",
            out_dir=str(tmp_path),
            max_workers=1,
        )
        assert Path(csv_path).exists(), f"Expected CSV at {csv_path}"

    def test_results_shape(self, tmp_path):
        n_params  = 3
        n_xstarts = 4

        def cost_fn(x):
            return float(np.sum(np.asarray(x) ** 2))

        results, top_rows, _ = optimize_parallel(
            cost_fn=cost_fn,
            n_params=n_params,
            bounds_low=0.0,
            bounds_high=1.0,
            seed=0,
            n_xstarts=n_xstarts,
            niter=1,
            method="L-BFGS-B",
            out_dir=str(tmp_path),
            max_workers=1,
        )
        # results: (n_xstarts, n_params + 1)  — params + cost
        assert results.shape == (n_xstarts, n_params + 1)
        # top_rows: at most 10 indices
        assert len(top_rows) <= 10
