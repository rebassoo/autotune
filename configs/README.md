# Configuration files

Each YAML config drives preprocessing (stage 1) and/or surrogate training and
optimization (stage 2) via `scripts/run_two_stage.py`.

## Aurora configs (added April 2026)

| File | Purpose |
|------|---------|
| `aurora_ne256_annual.yaml` | ne256 AMIP 20TR PPE on Aurora — annual-mean |
| `aurora_ne256_remapped_ne32pg2_annual.yaml` | Same ne256 data remapped to ne32pg2 grid |
| `scream_autocal.yaml` | Early prototype / scratch config |

## Perlmutter configs — original runs (added May 2026)

| File | Ensemble | Members | Purpose |
|------|----------|---------|---------|
| `perlmutter_ne32_annual.yaml` | ne32_ppe_2 (F2010) | ~175 | Single-fidelity LF surrogate |
| `perlmutter_multifidelity_annual.yaml` | ne256 AMIP (HF) + ne32_ppe_2 (LF) | 81 HF / 175 LF | AR1 multi-fidelity surrogate |
| `perlmutter_sf_ne256_annual.yaml` | ne256 AMIP 20TR | 81 | Single-fidelity HF surrogate (comparison baseline) |

These use `normranked_LH_sampling_base10.json` (256-entry LH sample from ne32_ppe_2).

Note: the ne256 AMIP and ne32 F2010 runs are different simulation types (not just
different resolutions), so the multi-fidelity model largely defaults to single-fidelity
HF behavior (ρ ≈ 0).

## Perlmutter configs — production runs (added June 2026)

| File | Ensemble | Members | Purpose |
|------|----------|---------|---------|
| `perlmutter_ne32_prod_annual.yaml` | ne32_ppe_prod (F2010) | 1024 | Single-fidelity or LF for MF |
| `perlmutter_ne128_prod_annual.yaml` | ne128_ppe_prod (F2010) | 256 | Single-fidelity or HF for MF |
| `perlmutter_mf_ne128_ne32_prod_annual.yaml` | ne128 (HF) + ne32 (LF), both F2010 | 256 HF / 1024 LF | AR1 multi-fidelity surrogate |

These use reconstructed params JSONs (see below). The ne128 output files are already
remapped to ne32pg2, so both fidelities share the same grid infrastructure
(control_file, regions_file, obs_files).  Using two F2010 runs at different native
resolutions is the physically correct multi-fidelity setup.

### Params JSONs for production runs

Built from `scream_input.yaml` in each member's run directory using
`scripts/build_params_json.py`:

```
normranked_ne32_prod_1024.json   — 1024 entries for ne32_ppe_prod
normranked_ne128_prod.json       — 256 entries for ne128_ppe_prod
```

### Preprocessing workflow

```bash
# 1. Build params JSONs (one-time)
python scripts/build_params_json.py \
    --ppe-dir /pscratch/sd/b/beydoun/e3sm_scratch/pm-gpu/ne32_ppe_prod \
    --output /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/normranked_ne32_prod_1024.json

python scripts/build_params_json.py \
    --ppe-dir /pscratch/sd/b/beydoun/e3sm_scratch/pm-gpu/ne128_ppe_prod \
    --output /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/normranked_ne128_prod.json

# 2. Preprocess each fidelity independently
python scripts/run_two_stage.py --config configs/perlmutter_ne32_prod_annual.yaml --stage 1 2>&1 | tee logs/preprocess_ne32_prod.log
python scripts/run_two_stage.py --config configs/perlmutter_ne128_prod_annual.yaml --stage 1 2>&1 | tee logs/preprocess_ne128_prod.log

# 3. Run multi-fidelity surrogate + optimization
python scripts/run_two_stage.py --config configs/perlmutter_mf_ne128_ne32_prod_annual.yaml --stage 2 --top-k-params 6 2>&1 | tee logs/stage2_mf_ne128_ne32.log
```

## Perlmutter configs — production runs v2 (added September 2026)

The 20260608 re-run of the production PPEs, on new paths. These are a
different design from the June 2026 (20260515) ensembles — the parameter
values differ — so they need their own params JSONs and preprocessing output.

| File | Ensemble | Members | Purpose |
|------|----------|---------|---------|
| `perlmutter_ne32_prod_v2_annual.yaml` | ne32_ppe_prod_v2 (F2010) | 984 of 1024 | Single-fidelity or LF for MF |
| `perlmutter_ne128_prod_v2_annual.yaml` | ne128_ppe_prod (F2010) | 218 of 256 | Single-fidelity or HF for MF |
| `perlmutter_mf_ne128_ne32_prod_v2_annual.yaml` | ne128 (HF) + ne32 (LF) | 218 HF / 984 LF | AR1 multi-fidelity surrogate |

Member data:

```
HF: /global/cfs/cdirs/e3sm/beydoun/ne128_ppe_prod
      PPEensemble_64node_full256.ne128pg2_ne128pg2.F2010-SCREAMv1.20260608.mNNN
LF: /pscratch/sd/b/beydoun/e3sm_scratch/pm-gpu/ne32_ppe_prod_v2
      PPEensemble_1node_full1024.ne32pg2_ne32pg2.F2010-SCREAMv1.20260608.mNNN
```

Members short of the full 13 monthly `1ma_ne32pg2.AVERAGE.nmonths_x1.*.nc`
files are dropped by the `min_files: 13` filter (38 of 256 HF, 40 of 1024 LF).
Each directory also holds one case dir without a `.mNNN` suffix, which
`build_run_list_generic` excludes from the ensemble and
`compute_default_run_zrg` picks up as the default/control run.

The HF design is exactly the first 256 rows of the LF design, so the two
fidelities are properly nested. Both are F2010 at different native
resolutions with output remapped to ne32pg2 — the physically correct
multi-fidelity setup.

The MF config differs from the v1 baseline in two research choices: the
Laplace GP-variance likelihood (`likelihood: laplace_gp_variance`, matching
the Aurora MF production runs) and OSR down-weighted to 0.1 in the cost.

### Workflow

```bash
# 1. Build params JSONs (one-time)
python scripts/build_params_json.py \
    --ppe-dir /global/cfs/cdirs/e3sm/beydoun/ne128_ppe_prod \
    --output <workdir>/normranked_ne128_prod_v2.json

python scripts/build_params_json.py \
    --ppe-dir /pscratch/sd/b/beydoun/e3sm_scratch/pm-gpu/ne32_ppe_prod_v2 \
    --output <workdir>/normranked_ne32_prod_v2_1024.json

# 2. Preprocess each fidelity independently
sbatch jobs/preprocess_ne128_prod_v2.sh
sbatch jobs/preprocess_ne32_prod_v2.sh

# 3. Multi-fidelity k-fold + AR1 GP training, then optimization.
#    Split in two because the phases want opposite threading: BLAS threads
#    for GP training, one thread per forked process for the 32-way optimizer.
sbatch jobs/stage2a_mf_ne128_ne32_prod_v2_train.sh
sbatch jobs/stage2b_mf_ne128_ne32_prod_v2_allparams_40starts_laplace.sh
```

All four job scripts use the `shared` QOS (max half a node) rather than
`regular`, which charges by fraction of a node and keeps the estimated cost
inside the per-user allocation cap.

### Single-fidelity ne128 comparison run

`perlmutter_ne128_prod_v2_sf_esem_mae_annual.yaml` runs the high-resolution
ne128 data on its own — no ne32 low-fidelity data — with the ESEm/GPflow
surrogate and the plain weighted-MAE cost:

```bash
sbatch jobs/stage2_ne128_prod_v2_sf_esem_mae_40starts.sh
```

It reads the same `preprocess_dir` as `perlmutter_ne128_prod_v2_annual.yaml`,
so that config's stage 1 is its only prerequisite, and it writes to its own
`output_dir` so its `sf_gp_trained.pkl` never collides with the MF run's
`mf_gp_trained.pkl`. Variable weights match the MF v2 config, so the two are
directly comparable: same HF data and same weights, with the fidelity,
surrogate backend and cost function being what differ.

Note that `run_stage2` (single-fidelity) has no `--skip-gp`/`--skip-optimize`
— those only feed the multi-fidelity branch — so k-fold, GP training and
optimization all run in one job rather than the split pair used for MF.

### Student-t (dof=4) noise-model runs

A matched multi-fidelity / single-fidelity pair under a Student-t noise model
with dof=4, replacing the squared-error growth of a Gaussian NLL with a term
that grows only ~logarithmically once the residual is large — so one
hard-to-fit variable or location cannot dominate the total cost:

| Config | Fidelity | Surrogate |
|---|---|---|
| `perlmutter_mf_ne128_ne32_prod_v2_studentt_dof4_annual.yaml` | ne128 + ne32 | AR1 (emukit/GPy) |
| `perlmutter_ne128_prod_v2_sf_gpy_studentt_dof4_annual.yaml` | ne128 only | GPy, one GP per output feature |

```bash
sbatch jobs/stage2_mf_ne128_ne32_prod_v2_studentt_dof4_40starts.sh
sbatch jobs/stage2_ne128_prod_v2_sf_gpy_studentt_dof4_40starts.sh
```

Two things make this pair a clean comparison. Both use the same weights and
the same `student_t_dof: 4.0`, and both use GPy underneath — the SF config
sets `sf_gp_backend: gpy` rather than the default ESEm, so its surrogate is
architecturally the AR1 model minus the low-fidelity data. The presence or
absence of the ne32 data is then the only difference between them.

The GPy backend is required rather than merely preferred: the Student-t cost
weights each location by that location's predictive variance, and ESEm trains
one *joint* GP with shared lengthscales, so it has no genuinely per-feature
variance to supply. GPy also pickles and survives `fork`, which is what makes
GP reuse and the process executor possible.

The AR1 GP does not depend on the cost function, so
`stage2_mf_ne128_ne32_prod_v2_studentt_dof4_40starts.sh` symlinks the
surrogate trained by `stage2a` for the Laplace run and passes `--skip-gp`
rather than retraining an identical model. It falls back to training its own
if that checkpoint is absent. No such reuse is possible on the
single-fidelity side, where the MAE run's ESEm `sf_gp_trained.pkl` is a
different backend and a different object from this run's
`sf_gp_gpy_trained.pkl`.
