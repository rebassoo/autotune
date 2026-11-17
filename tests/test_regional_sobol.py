"""Regional Sobol estimators against functions with known indices."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from autotune_gp.regional_sobol import saltelli_points, sobol_from_outputs   # noqa: E402


def test_points_layout():
    X = saltelli_points([0, 10], [1, 20], n=8, seed=1)
    assert X.shape == (8 * 4, 2)
    A, B, AB = X[:8], X[8:16], X[16:].reshape(2, 8, 2)
    assert np.all((X[:, 0] >= 0) & (X[:, 0] <= 1) & (X[:, 1] >= 10) & (X[:, 1] <= 20))
    np.testing.assert_array_equal(AB[0][:, 0], B[:, 0])
    np.testing.assert_array_equal(AB[0][:, 1], A[:, 1])
    np.testing.assert_array_equal(AB[1][:, 1], B[:, 1])


def test_ishigami_known_indices():
    # Ishigami on [-pi, pi]^3: ST = (0.5576, 0.4424, 0.2437), S1 = (0.3139, 0.4424, 0)
    lo, hi = [-np.pi] * 3, [np.pi] * 3
    X = saltelli_points(lo, hi, n=2 ** 14, seed=0)
    f = lambda x: np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])
    ST, S1, VT, V = sobol_from_outputs(f(X), 2 ** 14, 3)
    np.testing.assert_allclose(ST, [0.5576, 0.4424, 0.2437], atol=0.02)
    np.testing.assert_allclose(S1, [0.3139, 0.4424, 0.0], atol=0.03)
    assert VT[0] == pytest.approx(ST[0] * V)


def test_vectorized_over_outputs_and_absolute_effect():
    # y = a*x0 + b*x1 on a box of widths w: VT_i = (coef_i * w_i)^2 / 12
    lo, hi = np.array([0.0, 0.0]), np.array([0.1, 0.2])
    X = saltelli_points(lo, hi, n=4096, seed=2)
    Y = np.stack([3 * X[:, 0] + X[:, 1], X[:, 1]], axis=1)      # two outputs
    ST, S1, VT, V = sobol_from_outputs(Y, 4096, 2)
    assert ST.shape == (2, 2)
    np.testing.assert_allclose(VT[:, 0], [(0.3) ** 2 / 12, (0.2) ** 2 / 12], rtol=0.03)
    np.testing.assert_allclose(VT[:, 1], [0.0, 0.2 ** 2 / 12], atol=1e-6, rtol=0.03)
    np.testing.assert_allclose(ST[:, 1], [0.0, 1.0], atol=0.02)
