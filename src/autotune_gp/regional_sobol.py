"""
Sobol indices inside a sub-box of parameter space, vectorized over many outputs.

Sampling follows Saltelli (2010): two independent quasi-random matrices A and B
(n x d) inside the box, and for each parameter i the matrix AB_i = A with column i
from B; n*(d+2) model evaluations. Estimators, for every output at once:

    V     = Var([f(A); f(B)])
    VT_i  = mean((f(A) - f(AB_i))^2) / 2           (Jansen 1999; total effect)
    V1_i  = mean(f(B) * (f(AB_i) - f(A)))          (Saltelli 2010; first order)
    ST_i  = VT_i / V,   S1_i = V1_i / V

VT_i is also returned on its own: the variance the parameter accounts for, in the
output's units squared, which says whether the parameter matters in absolute terms
(a large ST_i of a tiny V does not).
"""
from __future__ import annotations

import numpy as np


def saltelli_points(lo, hi, n, seed=0):
    """(n*(d+2), d) points in the box [lo, hi]: rows A, then B, then AB_1..AB_d."""
    from scipy.stats import qmc
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = len(lo)
    u = qmc.Sobol(2 * d, scramble=True, seed=seed).random(n)
    A, B = lo + u[:, :d] * (hi - lo), lo + u[:, d:] * (hi - lo)
    AB = np.repeat(A[None], d, axis=0)
    AB[np.arange(d), :, np.arange(d)] = B.T
    return np.concatenate([A, B, AB.reshape(d * n, d)])


def sobol_from_outputs(Y, n, d):
    """Indices from outputs Y (n*(d+2), ...) evaluated at saltelli_points.

    Returns ST, S1, VT with shape (d, ...) and V with shape (...).
    """
    Y = np.asarray(Y, float)
    YA, YB = Y[:n], Y[n:2 * n]
    YAB = Y[2 * n:].reshape((d, n) + Y.shape[1:])
    V = np.var(np.concatenate([YA, YB]), axis=0)
    VT = 0.5 * np.mean((YA[None] - YAB) ** 2, axis=1)
    V1 = np.mean(YB[None] * (YAB - YA[None]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ST = np.where(V > 0, VT / V, 0.0)
        S1 = np.where(V > 0, V1 / V, 0.0)
    return ST, S1, VT, V
