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
