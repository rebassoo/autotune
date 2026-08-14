# The Student-t multi-fidelity objective in `zrg_cost_function_student_t`

## What this document covers

The full, explicit scalar function `cost_fn(x)` that L-BFGS-B minimizes over
the 19 free parameters `x` in the MF Student-t runs (e.g.
`configs/aurora_mf_ne128_ne32_prod_studentt_1000starts_annual.yaml`), built
up step by step from `src/autotune_gp/cost.py`'s
`zrg_cost_function_student_t`, with this run's actual configuration
(dof=4, kappa=0, lambda=0, the PCP/OSR/OLR/TLWP weights) plugged in
throughout, not left as symbols.

## Step 1 — the per-location Student-t NLL term

For a predicted mean `p`, observed value `o`, and GP predictive variance `v`
at one (location, variable) pair, with residual `r = p - o`:

```
l(r, v) = 0.5*log(v_c)                              <- normalization
        + 0.5*(nu+1)*log(1 + r^2/(nu*v_c))           <- fit term
        + const(nu)                                  <- fixed, nu-dependent only
        + kappa*v_c                                  <- optional variance penalty
```

where `v_c = max(v, var_floor)`, `var_floor = 1e-6`, and

```
const(nu) = -lgamma((nu+1)/2) + lgamma(nu/2) + 0.5*log(nu*pi)
```

is the Student-t normalizing constant — fixed once `nu` (degrees of
freedom) is chosen, independent of the residual or the variance. It doesn't
affect where the minimum is (adding a constant doesn't move an argmin) but
makes the returned value a true negative log-likelihood.

**This run's numbers:** `nu = 4`, `kappa = 0`. The `kappa*v_c` term
vanishes entirely, and `const(4) = -lgamma(2.5) + lgamma(2) + 0.5*log(4*pi)
≈ -0.2847 + 0 + 1.2655 ≈ 0.9808`. So for this run the per-location term is
exactly:

```
l(r, v) = 0.5*log(v_c) + 2.5*log(1 + r^2/(4*v_c)) + 0.9808
```

As `nu -> infinity` this expression converges exactly to the Gaussian NLL
term `0.5*r^2/v_c + 0.5*log(2*pi*v_c)` — Student-t is not a different family
bolted on, it's the same Gaussian-variance idea with heavier tails, and
recovers Gaussian exactly in the limit.

## Step 2 — mean over locations, within one block, one variable

`l` is computed at every (zonal band, region, global point) location. Within
one ZRG block (zonal, or regional) and one variable, the per-location values
are averaged — a plain mean here, since `zonal_weights` and
`regional_weights` are both `null` in this config. The global block has a
single location, so no averaging is needed there — it's just `l` evaluated
once.

Zonal and regional locations are never pooled into the same mean — they are
sliced apart before any averaging happens (see the ZRG feature layout:
`[zonal_0 ... zonal_14, region_0 ... region_5, global]`).

## Step 3 — weighted sum over the 4 variables

At each block level (zonal, regional, global), the per-variable means are
combined via a weighted sum:

```
zonal_block    = sum_v  w_v * mean_l_zonal(v)
regional_block = sum_v  w_v * mean_l_regional(v)
global_block   = sum_v  w_v * l_global(v)
```

**This run's weights** (`weights.variables` in the config):

| Variable | Weight |
|---|---|
| PCP  | 0.3 |
| OSR  | 0.1 |
| OLR  | 0.3 |
| TLWP | 0.3 |

(OSR downweighted relative to the other three, per an earlier request in
this project's history.)

## Step 4 — weighted sum over zonal / regional / global

```
snap_cost = zrg_w_zonal * zonal_block + zrg_w_regional * regional_block + zrg_w_global * global_block
```

**This run's weights** (`weights.zrg`): zonal = regional = global = 1/3
each. This is a fixed 3-way split, not proportional to how many locations
are in each group — without it the 15 zonal bands would numerically
dominate the 6 regions and especially the single global point, just by
outnumbering them.

## Step 5 — weighted sum over snapshots

```
total_cost = sum_snapshots  dy_w[snapshot] * snap_cost[snapshot]
```

This run has one snapshot (`ANN`, weight 1.0), so this step is a no-op:
`total_cost = snap_cost`.

## Step 6 — the training-distance penalty (inert in this run)

The full `cost_fn(x)` used by the optimizer adds one more term on top:

```
cost_fn(x) = total_cost + lambda * max(0, d(x) - tau)
```

where `d(x)` is `x`'s Euclidean distance (in each parameter's own
`bounds_from_data`-rescaled space) to the nearest real training-ensemble
member, and `tau` is the training ensemble's own worst-case internal
spacing (see `docs/` — or the parameter-uncertainty conversation — for the
full derivation of this penalty). **This run sets `lambda = 0`**, so this
term is identically zero for every `x` — it does not affect this run's
optimization at all, but is included here because it is structurally part
of every call to `cost_fn`.

## The full assembled objective

Putting Steps 1-6 together, this is exactly what `cost_fn(x)` computes at
every L-BFGS-B iteration, for this run's configuration:

```
cost_fn(x) =
    (1/3) * sum_v w_v * mean_{15 zonal bands}    [ 0.5*log(v_c) + 2.5*log(1 + r^2/(4*v_c)) + 0.9808 ]
  + (1/3) * sum_v w_v * mean_{6 regions}          [ 0.5*log(v_c) + 2.5*log(1 + r^2/(4*v_c)) + 0.9808 ]
  + (1/3) * sum_v w_v *                            [ 0.5*log(v_c) + 2.5*log(1 + r^2/(4*v_c)) + 0.9808 ]_global
```

where `(m(x), v(x)) = gp.predict(x)` is the multi-fidelity AR1 GP's
high-fidelity posterior mean and variance at `x`, `r = m(x) - obs` at each
location, `v_c = max(v(x), 1e-6)`, and `w_v` are the PCP/OSR/OLR/TLWP
weights from the table above. This single scalar is what all 1000 L-BFGS-B
starts are independently minimizing over the 19 free parameters.

## This run's configuration, at a glance

| Setting | Value |
|---|---|
| Likelihood | `student_t` |
| Degrees of freedom (nu) | 4 |
| Variance floor | 1e-6 |
| Variance penalty (kappa) | 0 (off) |
| Training-distance penalty (lambda) | 0 (off) |
| Variable weights | PCP 0.3, OSR 0.1, OLR 0.3, TLWP 0.3 |
| ZRG weights | zonal 1/3, regional 1/3, global 1/3 |
| n_xstarts | 1000 |
| max_workers | 192 |

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press. Chapter 2 gives the fit-term-plus-
  normalization structure this per-location term generalizes.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359-378. Why both terms
  (fit + normalization) must be present for the score to be proper.
- Student-t as a Gaussian scale mixture: a Normal distribution with unknown
  variance, given an Inverse-Gamma prior on that variance and marginalized
  out, is exactly a Student-t distribution — the mathematical basis for
  reading "switch to Student-t" as "account for uncertainty in the GP's own
  fitted noise/kernel hyperparameters," not an ad hoc robustness patch.
