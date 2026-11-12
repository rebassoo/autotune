"""
Posterior sampling of the parameters with emcee, for a cost that is a true negative
log-likelihood (e.g. likelihood 'pca_gaussian'): log p(x) = -cost(x) inside the
box, -inf outside (uniform prior over the box).

The cost function is usually a closure over a trained GP, which cannot be pickled,
so worker processes are forked after it is stored in a module global; each worker
then calls it through the top-level _log_prob.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

_COST = None
_LO = _HI = None


def _log_prob(x):
    if np.any(x < _LO) or np.any(x > _HI):
        return -np.inf
    return -float(_COST(x))


def sample_posterior(cost_fn, x0, lo, hi, n_walkers=64, n_steps=2000, n_proc=32,
                     init_scale=0.02, seed=0, progress=False):
    """Run an emcee ensemble started in small balls around x0 (clipped to the box).

    x0 is one point (n_dim,) or several (n_points, n_dim), e.g. the distinct optima of
    a multi-start optimization; walkers are assigned to the points in turn, so a
    posterior with several modes is entered in each of them.

    Returns {'chain': (n_steps, n_walkers, n_dim), 'log_prob': (n_steps, n_walkers),
    'acceptance': (n_walkers,)}.
    """
    import emcee
    global _COST, _LO, _HI
    _COST, _LO, _HI = cost_fn, np.asarray(lo, float), np.asarray(hi, float)
    x0 = np.atleast_2d(np.asarray(x0, float))
    rng = np.random.default_rng(seed)
    width = _HI - _LO
    centres = x0[np.arange(n_walkers) % len(x0)]
    p0 = np.clip(centres + init_scale * width * rng.normal(size=centres.shape),
                 _LO + 1e-9 * width, _HI - 1e-9 * width)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc) as pool:
        s = emcee.EnsembleSampler(n_walkers, x0.shape[1], _log_prob, pool=pool)
        s.run_mcmc(p0, n_steps, progress=progress)
    return {"chain": s.get_chain(), "log_prob": s.get_log_prob(),
            "acceptance": s.acceptance_fraction}
