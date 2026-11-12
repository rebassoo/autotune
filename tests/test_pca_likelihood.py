"""PCA likelihood: chi^2 calibration on synthetic errors and basic invariants."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from autotune_gp.pca_likelihood import build_basis, pca_cost   # noqa: E402


def _synthetic(n=4000, f=12, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(200, 3)) @ rng.normal(size=(3, f)) + 0.05 * rng.normal(size=(200, f))
    L = rng.normal(size=(f, f)) * 0.2
    E = rng.normal(size=(n, f)) @ L                     # correlated errors
    return Y, E


def test_chi2_matches_k_for_errors_from_the_same_distribution():
    Y, E = _synthetic()
    b = build_basis(Y, E[:3000], k=3, shrink=False)
    chi2 = [2 * pca_cost(e, b) for e in E[3000:]]
    assert np.mean(chi2) == pytest.approx(3, rel=0.1)


def test_zero_residual_costs_zero_and_scaling_is_quadratic():
    Y, E = _synthetic()
    b = build_basis(Y, E, k=3)
    r = E[0]
    assert pca_cost(np.zeros_like(r), b) == 0.0
    assert pca_cost(2 * r, b) == pytest.approx(4 * pca_cost(r, b))


def test_residual_layout_is_flattened_in_c_order():
    Y, E = _synthetic(f=12)
    b = build_basis(Y, E, k=3)
    assert pca_cost(E[1].reshape(1, 6, 2), b) == pytest.approx(pca_cost(E[1], b))


def test_rejects_bad_k_and_shapes():
    Y, E = _synthetic()
    with pytest.raises(ValueError):
        build_basis(Y, E, k=0)
    with pytest.raises(ValueError):
        build_basis(Y, E[:, :5], k=2)


def test_posterior_sampler_recovers_a_gaussian():
    from autotune_gp.posterior import sample_posterior
    mu, sd = np.array([0.3, 0.6]), np.array([0.05, 0.1])
    cost = lambda x: 0.5 * np.sum(((x - mu) / sd) ** 2)       # closure, like the GP cost
    out = sample_posterior(cost, mu, [0, 0], [1, 1], n_walkers=16, n_steps=1500, n_proc=2, seed=1)
    s = out["chain"][500:].reshape(-1, 2)
    np.testing.assert_allclose(s.mean(0), mu, atol=0.02)
    np.testing.assert_allclose(s.std(0), sd, rtol=0.15)


def test_posterior_sampler_starts_walkers_at_several_points():
    from autotune_gp.posterior import sample_posterior
    pts = np.array([[0.2, 0.2], [0.8, 0.8]])
    cost = lambda x: 0.5 * min(np.sum(((x - p) / 0.03) ** 2) for p in pts)   # two modes
    out = sample_posterior(cost, pts, [0, 0], [1, 1], n_walkers=8, n_steps=3, n_proc=2,
                           init_scale=0.001, seed=0)
    first = out["chain"][0]
    assert np.sum(np.linalg.norm(first - pts[0], axis=1) < 0.05) == 4
    assert np.sum(np.linalg.norm(first - pts[1], axis=1) < 0.05) == 4
