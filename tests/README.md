# Tests

## Setup

Activate the conda environment before running any tests:

```bash
mamba activate /lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env
```

Run all tests from the repo root:

```bash
cd /home/rebassoo/work/2026-04-17-Autotuning/autotune
python3 -m pytest tests/ -v
```

Run a single file:

```bash
python3 -m pytest tests/test_pipeline_generic.py -v
python3 -m pytest tests/test_optimization.py -v
```

---

## Test files

### `test_pipeline_generic.py`
Tests for the generic N-snapshot preprocessing pipeline (`preprocessing/pipeline.py`).
No real data files required — all inputs are synthetic.

| Class | Test | What it checks |
|---|---|---|
| `TestCalendarWeightedMean` | `test_weighted_differs_from_unweighted` | Weighted and unweighted monthly means differ when months have unequal length and different values |
| | `test_weighted_mean_matches_expected` | Weighted mean equals analytically computed value (Jan=0, Feb=1 → 28/59) |
| | `test_unweighted_mean_is_half` | Unweighted mean of [0, 1] is 0.5 regardless of month length |
| | `test_uniform_data_gives_same_result` | When all values are equal, weighted and unweighted means agree |
| `TestDropNanSymmetric` | `test_n_zonal_decrements` | `n_zonal` decreases by 1 when one zonal band is all-NaN |
| | `test_mirror_column_also_dropped` | All-NaN column in snapshot 1 causes the same positional column to be dropped in snapshot 2 |
| | `test_all_variables_drop_same_columns` | All variables lose the same columns after dropping |
| | `test_zrg_obs_columns_also_dropped` | `zrg_obs` loses the same columns as the per-variable DataFrames |
| | `test_no_drop_when_no_nans` | No columns dropped and `n_zonal` unchanged when data has no all-NaN columns |
| `TestDropNanExplicit` | `test_explicit_band_dropped_despite_no_nans` | A band listed in `explicit_drop_zonal` is removed even when data is fully finite |
| | `test_explicit_drop_is_symmetric` | Explicit band removal applies to all snapshots, not just the first |
| | `test_multiple_explicit_bands` | Two explicit bands are each removed symmetrically across all snapshots |

---

### `test_optimization.py`
Tests for the optimization pipeline modules. No GP training or real data required.

| Class | Test | What it checks |
|---|---|---|
| `TestFitTransformX` | `test_lower_bound_maps_to_zero` | Parameter at its physical lower bound normalises to 0.0 |
| | `test_upper_bound_maps_to_one` | Parameter at its physical upper bound normalises to 1.0 |
| | `test_midpoint_maps_to_half` | Midpoint of bounds normalises to 0.5 |
| | `test_fallback_to_data_range` | Without `param_bounds`, scaler fits to data min/max |
| | `test_returns_scaler_and_array` | Return type is (scaler, ndarray) |
| `TestFitTransformY` | `test_output_shape_preserved` | Shape unchanged after StandardScaler normalisation |
| | `test_zero_mean_per_variable` | Each variable has zero mean after scaling |
| | `test_unit_variance_per_variable` | Each variable has unit variance after scaling |
| | `test_returns_correct_number_of_scalers` | One scaler returned per variable |
| | `test_transform_obs_uses_same_scale_as_Y` | Obs equal to training mean transforms to 0 (same scale as Y) |
| | `test_transform_obs_output_shape` | Output shape is (1, n_feat, n_vars) |
| `TestSplitZrgObs` | `test_returns_n_vars_parts` | DataFrame splits into exactly n_vars blocks |
| | `test_each_part_has_correct_width` | Each block has n_feat columns |
| | `test_columns_are_strings` | Column labels are cast to str |
| | `test_values_are_contiguous_blocks` | Blocks preserve column ordering from original DataFrame |
| `TestCostMAEWeighted` | `test_zero_cost_when_preds_equal_obs` | Cost is exactly 0 when predictions match observations |
| | `test_positive_cost_for_nonzero_error` | Non-zero error gives positive cost |
| | `test_uniform_offset_gives_expected_cost` | Uniform offset of δ gives cost = δ analytically |
| | `test_cost_scales_linearly_with_offset` | Cost scales linearly with error magnitude |
| | `test_dy_weight_asymmetry` | Higher dy weight on the snapshot with larger error increases cost |
| | `test_raises_on_wrong_n_obs` | ValueError if n_obs ≠ 1 |
| | `test_raises_on_wrong_n_vars` | ValueError if n_vars ≠ 4 |
| | `test_raises_on_wrong_n_feat` | ValueError on ZRG layout mismatch |
| | `test_finite_output` | Random inputs give a finite scalar |
| `TestOptimizeParallel` | `test_finds_minimum_of_quadratic` | Finds minimum near [0,0] for x²+y² on [0,1]² |
| | `test_csv_is_written` | Results CSV is written to output directory |
| | `test_results_shape` | Results array has shape (n_xstarts, n_params+1) |

---

### `test_layout.py`
Smoke test for the ZRG cost function array layout.
Verifies that `zrg_cost_function_rmse_like_reference` runs without error on zero-valued inputs of the correct shape.

**Note:** requires `autotune_gp` on the Python path. Run from the repo root or with `PYTHONPATH=src`.

```bash
PYTHONPATH=src python3 -m pytest tests/test_layout.py -v
```

---

### `test_cost_equivalence.py`
Verifies that the refactored cost function in `autotune_gp/cost.py` produces numerically identical results to the original reference script (`reference/Final_Surrogate_ToShare_Background.py`).

**Note:** requires `autotune_gp` on the Python path and the reference script to be present.

```bash
PYTHONPATH=src python3 -m pytest tests/test_cost_equivalence.py -v
```

---

### `test_cost_equivalence_update.py`
Same as `test_cost_equivalence.py` but uses `sklearn.metrics.root_mean_squared_error` directly as the reference instead of the original script's inline implementation.

```bash
PYTHONPATH=src python3 -m pytest tests/test_cost_equivalence_update.py -v
```

---

### `test.py`
Standalone script (not a pytest test) that prints RMSE values computed two ways to verify they agree. Run directly:

```bash
python3 tests/test.py
```
