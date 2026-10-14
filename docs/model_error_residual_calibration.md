# Model-error calibration from residuals, then re-optimize

Status: **idea / not implemented** (written 2026-10-13). Nothing in the code
does this yet; this file records the plan so it can be picked up later.

## 1. Why

The likelihood costs (`zrg_cost_function_gaussian_nll`, `_laplace`,
`_student_t` in `src/autotune_gp/cost.py`) use only the **GP predictive
variance** as the noise term. There is no observational error and no
model (structural) error. Consequences:

- The fit is over-confident: it is dominated by locations where the GP
  happens to be confident, and parameter error bars are too narrow.
- Parameters may compensate for structural biases (pinning at bounds,
  equifinality-looking behaviour).
- Relative variable weights come from hand-set `weights.variables`, not from
  how well the model can actually match each field.

## 2. Where the error belongs: likelihood, not surrogate

| source | what it is | where it goes |
|---|---|---|
| emulator error | surrogate ≠ simulator | surrogate (GP predictive variance; already there) |
| internal variability | one 12-month run is one realization | surrogate (GP noise term, learned from PPE) |
| observation error ε | CERES / IMERG / MACLWP uncertainty | likelihood |
| structural error δ | simulator ≠ reality at any θ | likelihood |

Do **not** inflate the GP noise to absorb δ: the surrogate never sees the
observations, and inflating its noise just smooths away real parameter signal
(lower R², flatter cost for the wrong reason).

## 3. The equations (Kennedy & O'Hagan 2000 + 2001, modular form)

Levels l = 1, 2, 3 = ne32, ne128, ne256; s indexes an output feature
(variable × zonal band / region / global).

Simulator hierarchy (KO 2000, autoregressive multi-fidelity):

    η1(s, θ) ~ GP
    η_l(s, θ) = ρ_{l-1}(s) · η_{l-1}(s, θ) + γ_l(s, θ),   l = 2, 3,
    γ_l ~ GP, independent of η_{l-1}

Observations (KO 2001), tied to the **top level only**:

    z(s) = η3(s, θ*) + δ(s) + ε(s)
    δ(s) ~ N(0, σ²_δ,b(s))     # one variance per variable × block, see §5
    ε(s) ~ N(0, σ²_ε(s))       # published obs uncertainty

(KO 2001 also allows z = ρ·η + δ + ε; keep ρ = 1.)

Modular / plug-in likelihood (Bayarri et al. 2007; Liu, Bayarri & Berger
2009): train the MF emulator on the PPE only, then plug in its predictive
mean/variance. Observations do not feed back into the emulator, which avoids
making θ/δ confounding worse.

    μ3(s, θ)  = ρ2 μ2(s, θ) + μ_γ3(s, θ)            (recursive; exact for nested designs, Le Gratiet 2014)
    σ²3(s, θ) = ρ2² σ²2(s, θ) + σ²_γ3(s, θ)

    -log L(θ) = ½ Σ_s [ (z_s - μ3(s,θ))² / V_s(θ) + log 2π V_s(θ) ]
    V_s(θ)    = σ²3(s, θ) + σ²_δ,b(s) + σ²_ε(s)

The current Gaussian NLL is exactly this with σ²_δ = σ²_ε = 0
(`0.5·diff²/var + 0.5·log(2π·var)`, `var = max(pred_var, var_floor)`),
except that it takes a weighted **mean** over locations per block rather than
a sum, and multiplies by the hand variable weights.

## 4. Feature layout (what a "residual" is)

Per variable (`configs/perlmutter_ne256_all_annual.yaml`, `data:`):

| block | count | content |
|---|---|---|
| zonal | 18 | 10° latitude bands |
| regional | 6 | poles, extratropical land/ocean, tropical land, ascending/descending tropical ocean |
| global | 1 | global mean |

25 features per variable, 100 for PCP/OSR/OLR/TLWP. Every feature gets its
own residual r_s = z_s − μ3(s, θ̂).

**Units:** the cost works on StandardScaler-normalized targets
(`fit_transform_Y` in `transforms.py`; one scaler per variable, per-feature
`scale_`). `pred_var` is in those normalized units, so any σ_δ / σ_ε added to
it must be converted: σ²_norm(s) = σ²_phys(s) / scale_[s]². Estimating σ_δ
directly from normalized residuals avoids the conversion for δ; published obs
errors (physical units) need it.

## 5. Procedure: residual → error scale → re-optimize

This is coordinate descent on the joint likelihood in (θ, σ_δ).

0. **Start** with a non-zero σ_δ — e.g. published obs uncertainty, or
   20–30 % of each variable's PPE spread per block. Do *not* start from
   σ_δ = 0: then V_s = σ²3 alone, the fit is dominated by a few locations
   where the GP is very confident and can drift towards high-variance corners
   (the `variance_penalty_kappa` problem), so θ̂0 and its residuals are a
   poor starting point.
1. **Optimize** → θ̂_k (the usual 40-restart optimization).
2. **Residuals** at θ̂_k for all 100 features: r_s = z_s − μ3(s, θ̂_k).
3. **Update σ²_δ per variable × block** b ∈ {zonal, regional}:

       σ̂²_δ,b = max(0, wmean_{s∈b}[ r_s² − σ²3(s, θ̂_k) − σ²_ε(s) ])

   using the **same location weights the cost uses** (`zonal_weights`,
   `regional_weights`; currently null = equal). This is the approximate
   MLE given θ; the exact one solves the 1-D equation
   Σ_s w_s (r_s² − V_s) / V_s² = 0 for σ²_δ,b.
4. **Re-optimize** with V_s including σ²_δ → θ̂_{k+1}.
5. **Repeat 2–4** until θ̂ and σ_δ stop moving (typically 2–3 rounds).

That gives 8 estimated numbers (4 variables × {zonal, regional}). Few enough
to be stable, matched to the scale of what they are applied to.

### Pooling choices

- **Not one σ_δ per location**: it would absorb each residual exactly and
  remove all information about θ.
- **Not one σ_δ per variable over all 25**: zonal/regional errors are larger
  than global-mean errors (compensating regional biases cancel in the global
  mean), so a pooled value is too large for global and too small for some bands.
- **Global block cannot be estimated from residuals**: one residual per
  variable, and it is the number the fit drives hardest to zero. Specify it
  instead, from (a) published global-mean obs uncertainty (~1–2 W/m² for
  OSR/OLR), (b) the out-of-sample residual from the month split (§6), or
  (c) the zonal estimate scaled by ~1/√(effective number of independent bands).

### In-sample bias

Residuals at θ̂ are too small because θ̂ was tuned to shrink them. Rough
correction factor N / (N − p_eff); with 19 parameters and ~100 features that
is ~1.25, **but** neighbouring bands are correlated (effective N per zonal
block ~5–8, not 18), so the true underestimate is larger. Use §6 to avoid it.

## 6. Out-of-sample version: interleaved month split

Estimate σ_δ on months the fit never saw.

| set | months | seasons |
|---|---|---|
| A | Aug, Oct, Dec, Feb, Apr, Jun | JJA×2, SON, DJF×2, MAM |
| B | Sep, Nov, Jan, Mar, May, Jul | SON×2, DJF, MAM×2, JJA |

Optimize on B → θ̂_B; residuals on A at θ̂_B → σ_δ (no optimism); re-optimize
on B with it; swap A/B and average. The A/B disagreement measures how
month-dependent the estimate is.

Do **not** split contiguously and do **not** use 2 months: model errors are
seasonal and 2 months is a single season. A 6-month mean carries √2 the
internal-variability noise of the annual mean (2-month: √6); that noise ends
up in r and inflates σ_δ.

Cost: the surrogate must predict both halves → doubles the emulated outputs
→ roughly doubles 2-level training (~14 h → ~28 h; ~35 h with RESTOM). Needs a
`months:` selector on `SnapshotCfg` (sim and obs side); currently files are
chosen by `nc_glob` + first `max_files`. The snapshot / `dy` weight machinery
already handles multiple blocks.

## 7. Implementation (in-sample loop, §5)

**Cost.** `optimize.model_error` / `optimize.obs_error` in the config
(`src/autotune_gp/config.py`). `src/autotune_gp/model_error.py::extra_variance`
turns them into σ²_δ + σ²_ε per feature in normalized units, and
`run_two_stage.py` adds that to the GP's `pred_var` inside `cost_fn` before
the Gaussian / Laplace / Student-t cost, i.e. `var = max(pred_var + extra,
var_floor)`. Unset → bit-for-bit the old cost, and the checkpoint signature is
unchanged; set → the values go into the optimizer checkpoint signature, so
rounds with different σ never reuse each other's starts.

```yaml
optimize:
  model_error:            # normalized units = fraction of the PPE spread per feature
    PCP:  {zonal: 0.4, regional: 0.3, global: 0.25}
    OSR:  {zonal: 0.3, regional: 0.3, global_phys: 2.0}   # '<block>_phys' = physical units
  obs_error:              # always physical units
    TLWP: {zonal: 0.014, regional: 0.014}
```

**Loop.** `scripts/calibrate_model_error.py` (multi-fidelity surrogates):

1. starting σ: `--init-frac` (default 0.25 normalized, every block), or the
   config's `model_error`, or `--round0-dir` = reuse an existing optimization
   that had no model error as round 0;
2. each round runs `run_two_stage.py --stage 2 --skip-gp` with the current σ
   into `<out-dir>/round_K/` (the trained GP is symlinked in);
3. takes the best of the starts, predicts μ, σ²_GP there, and re-estimates
   σ_δ per variable × {zonal, regional} by a 1-D maximization of the **same**
   block likelihood the optimizer uses (`estimate_block_sigma2`; Gaussian
   method-of-moments value logged alongside). For Student-t the estimated
   number is on the squared-scale footing the cost uses;
4. global σ is held fixed (`--global VAR=SIGMA`, physical units);
5. stops when max|Δθ̂| < `--tol-theta` (0.02) and every σ moved by less than
   `--tol-sigma` (10 %) relative to max(σ, `--sigma-floor` 0.1);
6. `--sensitivity` then re-optimizes at σ_δ × 0.5 and × 2 (§8).

Outputs: `calibration_log.json` (θ̂, cost, σ used / next in normalized and
median physical units, per-block rms residual, rms GP sd, in-sample mean r²/V
— variance-corrected for Student-t), `model_error_final.yaml`,
`config_calibrated.yaml`. Resumable at round and at start level.

Batch: `jobs/calibrate_model_error_mf3.sh` (CONFIG / OUT / ROUND0 /
GLOBAL_ARGS via `--export`). With a 3-level GP a 40-start round is ~3.5–4.5 h,
so 3–4 rounds + sensitivity can exceed 24 h; resubmit and it continues.

Typical use when new ensemble members arrive: retrain as usual, run the
normal 40-start optimization (that is round 0, no model error), then

    sbatch --dependency=afterok:<opt job> \
      --export=ALL,CONFIG=configs/<cfg>.yaml,OUT=<dir>,ROUND0=<opt output_dir> \
      jobs/calibrate_model_error_mf3.sh

Not yet implemented: the out-of-sample month split (§6) and the profiled-out
cross-check (if σ_δ dominates V, maximizing over σ_δ gives
cost ≈ Σ_b (N_b / 2) · log RSS_b(θ); its argmin should agree with the loop).

## 8. Checks once it runs

- **Calibration**: out-of-sample mean(r² / V) ≈ 1 per variable × block
  (same z-score test as the colleague's `cover68` / `z_sd`).
- **Sensitivity**: rerun with σ_δ × 0.5 and × 2. If θ̂ and its intervals barely
  move, the exact value doesn't matter; if they move a lot, the calibration is
  driven by the error model, not the data.
- **Variable weights**: with σ_δ in the likelihood, relative variable weight
  comes from 1/V. Either drop `weights.variables` or keep it deliberately as a
  priority statement; keeping both double-weights.
- **Profile likelihood**: revisit `scripts/profile_likelihood.py --delta`.
  With V containing the real error budget, residuals are O(1) and the
  Wilks-style threshold becomes meaningful (currently the mean-over-locations
  cost makes Δ=0.5 far too strict; see earlier note on `--delta 0.02`).
- Cross-check σ_δ magnitudes against ne256 − ne128 / ne256 − ne32 differences
  at the same θ (`docs/resolution_bias*.png`) — a lower bound on the
  resolution-related part of δ — and against DYAMOND / CMIP spread.

## 9. What this does and doesn't fix

Does: honest variable weighting, stops over-weighting locations where the GP
is confident, error bars that reflect the real error budget, a usable
profile-likelihood threshold.

Doesn't: separate a persistent structural bias from a parameter shift
(θ/δ confounding, Brynjarsdóttir & O'Hagan 2014). σ_δ sets *how big* the error
is, not *which part* is parametric. For that, add discriminating observables —
e.g. seasonal snapshots (DJF/JJA) alongside ANN, where parameter effects and
structural errors often have different seasonal signatures — or the embedded
(parameter-space) error of Sargsyan, Huan & Najm, arXiv:1801.06768, which with
GPs can use closed-form moment matching for ARD-RBF inputs.

## References

- Kennedy & O'Hagan (2000), Predicting the output from a complex computer code
  when fast approximations are available, *Biometrika* 87, 1–13.
- Kennedy & O'Hagan (2001), Bayesian calibration of computer models,
  *JRSS-B* 63, 425–464.
- Bayarri et al. (2007), A framework for validation of computer models,
  *Technometrics* 49, 138–154.
- Liu, Bayarri & Berger (2009), Modularization in Bayesian analysis,
  *Bayesian Analysis* 4, 119–150.
- Le Gratiet (2014), Recursive co-kriging model for design of computer
  experiments with multiple levels of fidelity, *IJUQ* 4, 365–386.
- Brynjarsdóttir & O'Hagan (2014), Learning about physical parameters: the
  importance of model discrepancy, *Inverse Problems* 30, 114007.
- Williamson et al. (2015) / Hourdin et al. (2017): history matching and
  "tolerable model error" for climate-model tuning.
- Sargsyan, Huan & Najm (2019), Embedded model error representation for
  Bayesian model calibration, arXiv:1801.06768.
