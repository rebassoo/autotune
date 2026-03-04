# autotune-gp-finn

This repo reproduces the **full surrogate training and optimization** workflow found in
`reference/Final_Surrogate_ToShare_Background.py`, organized as a small package + CLI.

The workflow covers:
- Loading and filtering SCREAM simulation runs (DY1 and DY2)
- Extracting target variables and masking to observation grid cells
- Computing zonal, regional, and global (ZRG) spatial averages
- K-fold surrogate skill evaluation (R² / RMSE per variable)
- Training a GP surrogate (`esem.gp_model`) on the full dataset
- Optimizing model parameters against observations via basinhopping

---

## Install

```bash
cd /global/u2/r/rebassoo/work/2026_02_02_Autotuning-Repo/autotune-gp
pip install -e .
```

If the package is not installed, set the Python path manually before running any script:

```bash
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
```

Activate the conda environment:

```bash
conda activate /pscratch/sd/r/rebassoo/conda/ESEm_env
```

---

## Configuration

All paths and parameters are controlled by a single YAML file:

```bash
configs/scream_autocal.yaml
```

Key sections:

| Section | What it controls |
|---|---|
| `paths.preprocess_dir` | Where preprocessing writes its output pickles |
| `paths.output_dir` | Where optimization results (CSV) are written |
| `preprocess.*` | Source paths for simulation data, observations, regions file |
| `data.*` | ZRG layout (n_zonal, regions, variables) |
| `weights.*` | Cost function weights per variable, ZRG component, and day |
| `optimize.*` | Basinhopping settings (seed, starts, iterations, bounds) |
| `runtime.*` | GP training flag, compute backend (numpy / torch / cupy), device |

---

## Option A — End-to-end (everything in memory, one command)

Runs the full pipeline from raw simulation files through to optimized parameters.
Nothing is written to disk between stages.

```bash
python scripts/run_end_to_end.py --config configs/scream_autocal.yaml
```

**Stages:**
1. Build run list — enumerate folders, remove bad runs, filter `p3_ice_sed_knob >= 1`, find DY1 ∩ DY2
2. Load simulation netCDFs and observation files, mask simulations to obs grid cells
3. Compute ZRG averages (zonal, regional, global) for all variables, both days
4. **K-fold surrogate evaluation** — prints R² and RMSE per variable in normalised and physical space for each fold, plus the mean across folds
5. Normalise full dataset (MinMaxScaler for X, StandardScaler per variable for Y)
6. Train final GP surrogate on full dataset
7. Optimize — writes CSV of top results to `paths.output_dir`

---

## Option B — Two-stage (preprocess once, re-run surrogate/optimize cheaply)

Useful when the preprocessing (steps 1–3 above) is slow. Run stage 1 once to save
the preprocessed data to disk, then re-run stage 2 as many times as needed (e.g.
to change optimization settings or the GP) without repeating the expensive data loading.

### Stage 1 — Preprocessing (slow, run once)

```bash
python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 1
```

Writes the following pickles to `paths.preprocess_dir`:

| File | Contents |
|---|---|
| `run_list.pkl` | Filtered run names and file paths |
| `zrg_data.pkl` | ZRG DataFrames per variable (for reference / k-fold standalone use) |
| `kfold_data.pkl` | K-fold train/test splits (loaded by stage 2 for evaluation) |
| `obs.pkl` | Observation ZRG — `{'zrg_obs': DataFrame}` |
| `gp_proj.pkl` | Full-dataset training arrays — `{'X_train': ndarray, 'Y_train': ndarray}` |

### Stage 2 — Evaluation + surrogate + optimization (fast, re-runnable)

```bash
python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 2
```

**Stages:**
1. Load pickles from `paths.preprocess_dir`
2. **K-fold surrogate evaluation** — prints R² and RMSE per variable for each fold + mean
3. Normalise full dataset
4. Train final GP surrogate on full dataset
5. Optimize — writes CSV of top results to `paths.output_dir`

### Run both stages in sequence

```bash
python scripts/run_two_stage.py --config configs/scream_autocal.yaml
```

---

## Standalone surrogate evaluation

If you want to re-run the k-fold evaluation independently (without re-running the
full optimization), use:

```bash
# All 5 folds
python scripts/evaluate_surrogate.py --config configs/scream_autocal.yaml

# Single fold (0-indexed)
python scripts/evaluate_surrogate.py --config configs/scream_autocal.yaml --fold 1
```

Requires `kfold_data.pkl` in `paths.preprocess_dir` (produced by stage 1 or
`preprocessing/04_kfold_and_stack.py`).

---

## Optimize only (using pre-existing surrogate pickles)

If you already have `obs.pkl` and `gp_proj.pkl` from a previous run or from
an external source, you can skip preprocessing entirely using the CLI:

```bash
autotune-gp optimize --config configs/scream_autocal.yaml
# or without installing:
python -m autotune_gp.cli optimize --config configs/scream_autocal.yaml
```

Set `paths.obs_pkl` and `paths.gp_proj_pkl` in the config to point to the
existing pickles.

---

## CPU / GPU

The **cost evaluation** can run on:
- `numpy` — CPU (default)
- `torch` — CPU or CUDA GPU
- `cupy` — GPU (if installed)

To enable GPU cost evaluation via PyTorch:

```yaml
runtime:
  backend: torch
  device: cuda
```

> Whether `esem.gp_model` prediction runs on GPU depends on the `esem` backend
> (TensorFlow / GPflow with CUDA). This repo makes the objective computation
> device-aware and avoids CPU-only sklearn metrics.

---

## Repository layout

```
configs/
  scream_autocal.yaml        # all settings and paths
preprocessing/
  pipeline.py                # importable preprocessing functions
  01_build_run_list.py       # standalone: enumerate + filter runs
  02_load_and_mask.py        # standalone: load netCDFs, mask to obs
  03_compute_zrg.py          # standalone: zonal/regional/global averages
  04_kfold_and_stack.py      # standalone: k-fold splits + stack arrays
reference/
  Final_Surrogate_ToShare_Background.py   # original notebook (verbatim)
scripts/
  run_end_to_end.py          # Option A: full pipeline in memory
  run_two_stage.py           # Option B: stage 1 (preprocess) + stage 2 (train+optimize)
  evaluate_surrogate.py      # standalone k-fold R² evaluation
  run_optimize.sh            # thin shell wrapper for optimize-only CLI
src/autotune_gp/
  transforms.py              # MinMaxScaler (X) and StandardScaler (Y) fits
  io.py                      # load obs / GP projection pickles
  gp.py                      # GPWrapper (esem.gp_model)
  cost.py                    # ZRG RMSE/L1 cost function (numpy / torch / cupy)
  optimize.py                # parallel basinhopping
  evaluate.py                # k-fold R² / RMSE evaluation
  config.py                  # dataclasses + YAML loader
  backend.py                 # numpy / torch / cupy abstraction
  cli.py                     # autotune-gp optimize entry point
```

---

## Notes

- ZRG layout: `n_feat == 2 * (n_zonal + n_regions + 1)` — 2 days × (18 zonal + 6 regional + 1 global) = 50 features per variable, 200 total across 4 variables.
- The standalone scripts in `preprocessing/` write intermediate pickles to the current working directory. Use the pipeline scripts (`run_end_to_end.py`, `run_two_stage.py`) for path-managed runs driven by the config.
