"""
Two-stage pipeline: preprocessing → save to disk → evaluation → surrogate → optimization.

Stage 1 writes pickles to cfg.paths.preprocess_dir:
    obs.pkl        {'zrg_obs': pd.DataFrame}         (readable by load_obs)
    gp_proj.pkl    {'X_train': DataFrame,             (readable by load_gp_proj)
                    'Y_train': ndarray}
    scalers.pkl    {'X_pipeline': MinMaxScaler,       fitted scalers (physical → ML units)
                    'Y_pipeline_<VAR>': StandardScaler, ...}
    kfold_data.pkl {'folds': list}                   (k-fold splits for evaluation)
    run_list.pkl   (intermediate)
    zrg_data.pkl   (intermediate; per-variable ZRG DataFrames)

Stage 2 loads those pickles and runs:
    k-fold surrogate evaluation (R² / RMSE per variable)
    → train final surrogate on full dataset
    → optimize

You can re-run stage 2 alone (skipping slow preprocessing) with --stage 2.

Usage:
    python scripts/run_two_stage.py --config configs/scream_autocal.yaml
    python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 1
    python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 2

    # Process PPE only (obs files not yet available):
    python scripts/run_two_stage.py --config configs/aurora_ne256_annual.yaml --stage 1 --preprocess-mode ppe

    # Later, once obs files are ready — process obs only and align to existing column mask:
    python scripts/run_two_stage.py --config configs/aurora_ne256_annual.yaml --stage 1 --preprocess-mode obs
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

try:
    import cloudpickle as _cpickle
except ImportError:
    _cpickle = None
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import load_config
from autotune_gp.backend import get_backend
from autotune_gp.io import load_obs, load_gp_proj, split_zrg_obs
from autotune_gp.transforms import fit_transform_X, fit_transform_Y, transform_obs
from autotune_gp.gp import GPWrapper
from autotune_gp.cost import zrg_cost_function_mae_weighted
from autotune_gp.optimize import optimize_parallel
from autotune_gp.evaluate import run_kfold_evaluation, select_top_params_hf

from preprocessing.pipeline import (
    build_run_list,
    load_and_mask,
    compute_zrg,
    drop_nan_zrg_features,
    build_run_list_generic,
    load_and_mask_generic,
    compute_zrg_generic,
    compute_obs_zrg_generic,
    compute_default_run_zrg,
    drop_nan_zrg_features_generic,
    stack_all_data,
    make_folds,
)


def run_stage1(cfg, preprocess_mode: str = "both", make_plots: bool = False):
    """Preprocessing: build ZRG arrays and save to preprocess_dir.

    preprocess_mode:
        'both' — process PPE and obs together (default, original behaviour)
        'ppe'  — process PPE only; saves gp_proj/scalers/kfold/column_mask pickles.
                 Skips obs. Use on systems where obs files are not yet available.
        'obs'  — process obs only; loads column_mask.pkl written by a prior 'ppe'
                 run to align columns, then saves obs.pkl.
    """
    pp = cfg.preprocess
    if pp is None:
        raise ValueError("Config must include a [preprocess] section for stage 1.")

    var_names = list(pp.variables.keys())
    out_dir = Path(cfg.paths.preprocess_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Obs-only mode: load obs, apply saved column mask, save obs.pkl
    # ------------------------------------------------------------------
    if preprocess_mode == "obs":
        if pp.snapshots is None:
            raise ValueError("obs mode requires the generic (snapshots) pipeline.")
        print("=== Stage 1 (obs-only): Compute obs ZRG averages ===")
        obs_result = compute_obs_zrg_generic(
            snapshots=pp.snapshots,
            variables=pp.variables,
            control_file=pp.control_file,
            regions_file=pp.regions_file,
        )
        zrg_obs = obs_result["zrg_obs"]

        mask_path = out_dir / "column_mask.pkl"
        if mask_path.exists():
            print(f"  Applying column mask from {mask_path}")
            with open(mask_path, "rb") as f:
                mask = pickle.load(f)
            n_feat_orig = mask["n_feat_original"]
            valid       = mask["valid_feat_indices"]
            n_vars_mask = mask["n_vars"]
            if n_vars_mask != len(var_names):
                raise ValueError(
                    f"column_mask.pkl has n_vars={n_vars_mask} but config has "
                    f"{len(var_names)} variables. Re-run PPE preprocessing."
                )
            all_valid = [v * n_feat_orig + col for v in range(n_vars_mask) for col in valid]
            zrg_obs = zrg_obs.iloc[:, all_valid]
            print(f"  zrg_obs after mask: {zrg_obs.shape}")
        else:
            print(f"  No column_mask.pkl found in {out_dir} — using all obs columns as-is.")

        with open(out_dir / "obs.pkl", "wb") as f:
            pickle.dump({"zrg_obs": zrg_obs}, f)
        print(f"  Saved obs.pkl  (zrg_obs shape: {zrg_obs.shape})")
        print(f"Stage 1 (obs-only) complete. Output in: {out_dir}")
        return

    # ------------------------------------------------------------------
    # PPE mode or both: build run list, load simulations, compute ZRG
    # ------------------------------------------------------------------
    print("=== Stage 1: Build run list ===")
    if pp.snapshots is not None:
        param_names = list(cfg.optimize.param_physical_bounds.keys()) \
                      if cfg.optimize.param_physical_bounds else None
        run_list = build_run_list_generic(
            params_json=pp.params_json,
            snapshots=pp.snapshots,
            param_names=param_names,
        )
    else:
        run_list = build_run_list(
            params_json=pp.params_json,
            DY1_sim_dir=pp.DY1_sim_dir,
            DY1_nc_suffix=pp.DY1_nc_suffix,
            DY2_sim_dir=pp.DY2_sim_dir,
            DY2_nc_suffix=pp.DY2_nc_suffix,
        )
    with open(out_dir / "run_list.pkl", "wb") as f:
        pickle.dump(run_list, f)
    print(f"  Saved run_list.pkl  ({len(run_list['sim_names'])} runs)")

    print("=== Stage 1: Load simulations and observations ===")
    if pp.snapshots is not None:
        sim_data = load_and_mask_generic(
            run_list=run_list,
            snapshots=pp.snapshots,
            variables=pp.variables,
            n_workers=int(os.environ.get("PREPROCESS_WORKERS", 0)),
        )
    else:
        sim_data = load_and_mask(
            run_list=run_list,
            DY1_obs_dir=pp.DY1_obs_dir,
            DY2_obs_dir=pp.DY2_obs_dir,
            variables=pp.variables,
        )

    print("=== Stage 1: Compute ZRG averages ===")
    if pp.snapshots is not None:
        zrg_result = compute_zrg_generic(
            sim_names=run_list["sim_names"],
            ppe_dataset_small=sim_data["ppe_dataset_small"],
            obs_dict=sim_data,
            control_file=pp.control_file,
            regions_file=pp.regions_file,
            variables=pp.variables,
            snapshots=pp.snapshots,
        )
    else:
        zrg_result = compute_zrg(
            sim_names=run_list["sim_names"],
            ppe_dataset_small=sim_data["ppe_dataset_small"],
            obs_dict=sim_data,
            control_file=pp.control_file,
            regions_file=pp.regions_file,
            variables=pp.variables,
        )
    with open(out_dir / "zrg_data.pkl", "wb") as f:
        pickle.dump(zrg_result, f)
    print("  Saved zrg_data.pkl")

    print("=== Stage 1: Drop all-NaN ZRG features ===")
    if pp.snapshots is not None:
        # In PPE-only mode obs rows are all-NaN placeholders; skip obs NaN scan.
        check_obs = (preprocess_mode == "both")
        zrg_result, _ = drop_nan_zrg_features_generic(
            zrg_result,
            var_names=var_names,
            n_zonal=cfg.data.n_zonal,
            n_regions=len(cfg.data.regions_list),
            regions_list=cfg.data.regions_list,
            snapshots=pp.snapshots,
            explicit_drop_zonal=pp.drop_zonal_bands,
            check_obs_nan=check_obs,
        )
    else:
        zrg_result, _ = drop_nan_zrg_features(
            zrg_result,
            var_names=var_names,
            n_zonal=cfg.data.n_zonal,
            n_regions=len(cfg.data.regions_list),
            regions_list=cfg.data.regions_list,
            explicit_drop_zonal=pp.drop_zonal_bands,
        )

    # Save post-drop ZRG data (used by the standalone plot script)
    with open(out_dir / "zrg_data_clean.pkl", "wb") as f:
        pickle.dump(zrg_result, f)
    print("  Saved zrg_data_clean.pkl")

    # Save column mask so obs-only preprocessing can align to the same columns later
    if pp.snapshots is not None:
        n_snaps = len(pp.snapshots)
        n_per_snap = cfg.data.n_zonal + len(cfg.data.regions_list) + 1
        n_feat_original = n_per_snap * n_snaps
        valid_feat_indices = zrg_result.get("_valid_feat_indices", list(range(n_feat_original)))
        mask_data = {
            "valid_feat_indices": valid_feat_indices,
            "n_feat_original":    n_feat_original,
            "n_vars":             len(var_names),
        }
        with open(out_dir / "column_mask.pkl", "wb") as f:
            pickle.dump(mask_data, f)
        print(f"  Saved column_mask.pkl  ({len(valid_feat_indices)}/{n_feat_original} features kept)")

        print("=== Stage 1: Compute default/control run ZRG ===")
        default_result = compute_default_run_zrg(run_list, mask_data, pp, var_names)
        if default_result is not None:
            with open(out_dir / "default_run_zrg.pkl", "wb") as f:
                pickle.dump(default_result, f)
            print(f"  Saved default_run_zrg.pkl  (default_name={default_result['default_name']})")

    # Optional ZRG diagnostic plot
    if make_plots and pp.snapshots is not None:
        from preprocessing.plots import plot_ppe_zrg
        print("=== Stage 1: Plot ZRG diagnostics ===")
        plot_ppe_zrg(
            zrg_result=zrg_result,
            var_names=var_names,
            n_regions=len(cfg.data.regions_list),
            regions_list=cfg.data.regions_list,
            snapshots=pp.snapshots,
            out_dir=str(out_dir),
        )

    print("=== Stage 1: Stack training arrays ===")
    X_train, Y_train_ZRG = stack_all_data(zrg_result, run_list["ppe_params"],
                                          var_names=var_names)

    print("=== Stage 1: Build k-fold splits ===")
    folds = make_folds(zrg_result, run_list["ppe_params"], var_names=var_names)
    with open(out_dir / "kfold_data.pkl", "wb") as f:
        pickle.dump({"folds": folds}, f)
    print(f"  Saved kfold_data.pkl  ({len(folds)} folds)")

    with open(out_dir / "gp_proj.pkl", "wb") as f:
        pickle.dump({"X_train": X_train, "Y_train": Y_train_ZRG}, f)
    print(f"  Saved gp_proj.pkl  (X: {X_train.shape}, Y: {Y_train_ZRG.shape})")

    if preprocess_mode == "both":
        with open(out_dir / "obs.pkl", "wb") as f:
            pickle.dump({"zrg_obs": zrg_result["zrg_obs"]}, f)
        print(f"  Saved obs.pkl  (zrg_obs shape: {zrg_result['zrg_obs'].shape})")

    print("=== Stage 1: Fit and save scalers ===")
    phys = cfg.optimize.param_physical_bounds
    if phys and hasattr(X_train, "columns"):
        param_bounds_arr = np.array([phys[col] for col in X_train.columns])
        X_sc, _ = fit_transform_X(X_train, param_bounds=param_bounds_arr.T)
    else:
        X_sc, _ = fit_transform_X(X_train)
    Y_scalers, _ = fit_transform_Y(Y_train_ZRG)
    scalers_dict = {
        "X_pipeline": X_sc,
        **{f"Y_pipeline_{v}": Y_scalers[i] for i, v in enumerate(var_names)},
    }
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers_dict, f)
    print(f"  Saved scalers.pkl  (X: MinMaxScaler on bounds, Y: StandardScaler x{len(var_names)})")
    print(f"Stage 1 complete. All outputs in: {out_dir}")


def _zonal_center_lats(cfg, n_zonal_surviving):
    """Return the actual band-centre latitudes for surviving zonal bands.

    Uses the original n_zonal from cfg plus column_mask.pkl to find which
    bands were kept after dropping polar bands in stage 1.  Falls back to
    None (triggers approximate relabelling in diagnostics) if the mask file
    is missing.
    """
    col_mask_path = Path(cfg.paths.preprocess_dir) / "column_mask.pkl"
    if not col_mask_path.exists():
        return None
    with open(col_mask_path, "rb") as f:
        col_mask = pickle.load(f)
    n_zonal_orig = int(cfg.data.n_zonal)
    orig_edges   = np.linspace(-90, 90, n_zonal_orig + 1)
    orig_centers = 0.5 * (orig_edges[:-1] + orig_edges[1:])
    surviving    = [i for i in col_mask["valid_feat_indices"] if i < n_zonal_orig]
    return [orig_centers[i] for i in surviving]


def _resolve_opt_bounds(cfg, X_norm_list, param_names, n_params, log=print):
    """Return per-parameter (bounds_low, bounds_high) lists of length n_params.

    By default this just broadcasts the flat cfg.optimize.bounds to every
    parameter. When cfg.optimize.bounds_from_data is set, each parameter is
    additionally clamped to the range the training data actually covers.

    This matters because param_physical_bounds is wider than the range the PPE
    actually sampled for several parameters (e.g. length_fac is declared
    [0.1, 10] but sampled only up to ~1.9). Searching the full declared box
    lets the optimizer wander into regions with no training data, where an RBF
    GP just reverts to its prior mean — and the cost uses only the predictive
    mean, so nothing penalises being there.

    X_norm_list: list of normalised training-X arrays whose union defines the
                 supported region (for multi-fidelity, pass both HF and LF).
    """
    _bl = cfg.optimize.bounds["low"]
    _bh = cfg.optimize.bounds["high"]
    lo = np.array([_bl[i] if hasattr(_bl, "__getitem__") else _bl
                   for i in range(n_params)], dtype=float)
    hi = np.array([_bh[i] if hasattr(_bh, "__getitem__") else _bh
                   for i in range(n_params)], dtype=float)

    if not getattr(cfg.optimize, "bounds_from_data", False):
        return lo.tolist(), hi.tolist()

    stacked = np.vstack([np.asarray(X, dtype=float) for X in X_norm_list])
    new_lo = np.maximum(lo, stacked.min(axis=0))
    new_hi = np.minimum(hi, stacked.max(axis=0))

    names = list(param_names) if param_names else [f"p{i}" for i in range(n_params)]
    w = max(len(n) for n in names)
    log("  Optimizer bounds restricted to the sampled range "
        "(optimize.bounds_from_data=true):")
    for i, name in enumerate(names):
        span = hi[i] - lo[i]
        frac = (new_hi[i] - new_lo[i]) / span if span > 0 else 1.0
        flag = "  <-- narrowed" if frac < 0.99 else ""
        log(f"    {name:<{w}}  [{new_lo[i]:.4f}, {new_hi[i]:.4f}]"
            f"  ({frac * 100:5.1f}% of declared range){flag}")
    return new_lo.tolist(), new_hi.tolist()


def _load_default_zrg(preprocess_dir):
    """Load Y_default_ZRG from default_run_zrg.pkl if present, else None."""
    path = Path(preprocess_dir) / "default_run_zrg.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["Y_default_ZRG"]


def run_stage2(cfg, _preprocess_pkls=None):
    """K-fold evaluation + surrogate training + optimization.

    Path priority for obs/gp_proj pickles:
      1. _preprocess_pkls  — passed explicitly when stage 1 just ran (both-stage mode)
      2. cfg.paths.obs_pkl / cfg.paths.gp_proj_pkl  — config-specified paths
      3. preprocess_dir/obs.pkl / gp_proj.pkl  — fallback if config paths absent

    K-fold evaluation is skipped if kfold_data.pkl is not found in preprocess_dir.
    """
    out_dir = Path(cfg.paths.preprocess_dir)
    kfold_pkl = out_dir / "kfold_data.pkl"

    if _preprocess_pkls is not None:
        obs_pkl     = _preprocess_pkls["obs_pkl"]
        gp_proj_pkl = _preprocess_pkls["gp_proj_pkl"]
    elif cfg.paths.obs_pkl and Path(cfg.paths.obs_pkl).exists():
        obs_pkl     = cfg.paths.obs_pkl
        gp_proj_pkl = cfg.paths.gp_proj_pkl
    else:
        obs_pkl     = str(out_dir / "obs.pkl")
        gp_proj_pkl = str(out_dir / "gp_proj.pkl")

    print(f"=== Stage 2: Load data ===")
    print(f"  obs:     {obs_pkl}")
    print(f"  gp_proj: {gp_proj_pkl}")

    obs_loaded  = load_obs(obs_pkl)
    zrg_obs     = obs_loaded["zrg_obs"]
    obs_parts   = split_zrg_obs(zrg_obs, n_vars=len(cfg.data.variables))

    gp_loaded   = load_gp_proj(gp_proj_pkl)
    X_train     = gp_loaded["X_train"]
    Y_train_ZRG = gp_loaded["Y_train"]

    print("=== Stage 2: K-fold surrogate evaluation ===")
    if not kfold_pkl.exists():
        print(f"  kfold_data.pkl not found in {out_dir} — skipping k-fold evaluation.")
        print("  (Run stage 1 first to generate k-fold splits.)")
    else:
        with open(kfold_pkl, "rb") as f:
            kfold_data = pickle.load(f)
        run_kfold_evaluation(kfold_data["folds"], train_gp=cfg.runtime.train_gp)

    print("=== Stage 2: Normalise (full dataset) ===")
    var_names = cfg.data.variables
    # Try to load saved scalers: first from gp_proj_pkl, then from scalers.pkl
    X_sc   = gp_loaded.get("X_pipeline")
    Y_scalers = [gp_loaded.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is None or not all(s is not None for s in Y_scalers):
        scalers_pkl_path = out_dir / "scalers.pkl"
        if scalers_pkl_path.exists():
            with open(scalers_pkl_path, "rb") as f:
                saved = pickle.load(f)
            X_sc      = saved.get("X_pipeline")
            Y_scalers = [saved.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is not None and all(s is not None for s in Y_scalers):
        X_train_norm  = X_sc.transform(X_train)
        Y_train_norm  = np.stack(
            [Y_scalers[j].transform(Y_train_ZRG[:, :, j]) for j in range(len(var_names))],
            axis=0).transpose(1, 2, 0)
        obs_norm = transform_obs(obs_parts, Y_scalers)
        print("  Using saved scalers (exact match with training)")
    else:
        print("  Warning: no saved scalers found — refitting from scratch")
        phys = cfg.optimize.param_physical_bounds
        if phys and hasattr(X_train, "columns"):
            param_bounds = np.array([phys[col] for col in X_train.columns])
            X_sc, X_train_norm = fit_transform_X(X_train, param_bounds=param_bounds.T)
        else:
            X_sc, X_train_norm = fit_transform_X(X_train)
        Y_scalers, Y_train_norm = fit_transform_Y(Y_train_ZRG)
        obs_norm = transform_obs(obs_parts, Y_scalers)

    _sf_gp_ckpt = Path(cfg.paths.output_dir) / "sf_gp_trained.pkl"
    _sf_gp_ckpt.parent.mkdir(parents=True, exist_ok=True)
    gp = None
    if _sf_gp_ckpt.exists():
        print(f"=== Stage 2: Loading saved GP from {_sf_gp_ckpt} ===")
        try:
            _loader = _cpickle if _cpickle is not None else pickle
            with open(_sf_gp_ckpt, "rb") as f:
                gp = _loader.load(f)
            print("  GP loaded.")
        except Exception as e:
            print(f"  Warning: could not load saved GP ({e}); retraining.")
            _sf_gp_ckpt.unlink(missing_ok=True)
            gp = None
    if gp is None:
        print("=== Stage 2: Train GP surrogate (full dataset) ===")
        gp = GPWrapper(X_train_norm, Y_train_norm)
        if cfg.runtime.train_gp:
            gp.train(tf_determinism=cfg.runtime.tf_determinism)
            _saver = _cpickle if _cpickle is not None else pickle
            try:
                with open(_sf_gp_ckpt, "wb") as f:
                    _saver.dump(gp, f)
                print(f"  Trained GP saved → {_sf_gp_ckpt}")
            except Exception as e:
                print(f"  Warning: could not save GP ({e}); will retrain on next run.")
                _sf_gp_ckpt.unlink(missing_ok=True)

    print("=== Stage 2: Optimize ===")
    backend   = get_backend(cfg.runtime.backend, cfg.runtime.device)
    n_regions = len(cfg.data.regions_list)
    # Derive n_zonal from actual data shape in case bands were dropped during stage 1
    n_snaps   = len(cfg.preprocess.snapshots) if (cfg.preprocess and cfg.preprocess.snapshots) else 2
    n_zonal   = Y_train_ZRG.shape[1] // n_snaps - n_regions - 1
    if n_zonal != int(cfg.data.n_zonal):
        print(f"  Note: n_zonal={n_zonal} (derived from data shape; "
              f"config has {cfg.data.n_zonal} — bands were dropped during stage 1)")
    var_w            = cfg.weights.variables
    zrg_w            = cfg.weights.zrg
    dy_w             = cfg.weights.dy
    zonal_weights    = cfg.weights.zonal_weights
    regional_weights = cfg.weights.regional_weights

    # Build parameter ordering constraint index pairs from column names
    param_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    constraint_pairs = []
    for pair in (cfg.optimize.param_ordering_constraints or []):
        low_name, high_name = pair
        if param_names and low_name in param_names and high_name in param_names:
            constraint_pairs.append((param_names.index(low_name), param_names.index(high_name)))

    def cost_fn(x):
        x = np.asarray(x, dtype=float)
        for low_idx, high_idx in constraint_pairs:
            violation = x[low_idx] - x[high_idx]
            if violation > 0:
                return 1e2 + 1e2 * violation
        m, _ = gp.predict(x)
        return zrg_cost_function_mae_weighted(
            m, obs_norm, var_w, zrg_w, dy_w,
            n_zonal=n_zonal, n_regions=n_regions, backend=backend,
            var_names=list(var_names),
            zonal_weights=zonal_weights, regional_weights=regional_weights,
        )

    n_opt_params = len(param_names) if param_names else cfg.optimize.n_params
    bounds_low, bounds_high = _resolve_opt_bounds(
        cfg, [X_train_norm], param_names, n_opt_params)

    results, top_rows, csv_path = optimize_parallel(
        cost_fn=cost_fn,
        n_params=n_opt_params,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        seed=cfg.optimize.seed,
        n_xstarts=cfg.optimize.n_xstarts,
        niter=cfg.optimize.niter,
        method=cfg.optimize.method,
        out_dir=cfg.paths.output_dir,
        max_workers=cfg.optimize.max_workers,
        executor=cfg.optimize.executor,
        checkpoint_dir=str(Path(cfg.paths.output_dir) / "optimize_checkpoints"),
    )

    print(f"Done. Results: {csv_path}")

    if cfg.diagnostics.enabled:
        from autotune_gp.diagnostics import run_diagnostics
        diag_dir = Path(cfg.diagnostics.output_dir) if cfg.diagnostics.output_dir \
                   else Path(cfg.paths.output_dir) / "diagnostics"
        print(f"=== Stage 2: Diagnostics ===")
        run_diagnostics(
            results=results,
            top_rows=top_rows,
            gp=gp,
            Y_train_ZRG=Y_train_ZRG,
            Y_scalers=Y_scalers,
            obs_parts=obs_parts,
            param_names=param_names,
            var_names=var_names,
            n_zonal=n_zonal,
            n_regions=n_regions,
            regions_list=cfg.data.regions_list,
            out_dir=diag_dir,
            suffix=f"_seed{cfg.optimize.seed}",
            zonal_center_lats=_zonal_center_lats(cfg, n_zonal),
            hf_default_ZRG=_load_default_zrg(cfg.paths.preprocess_dir),
        )


def run_stage2_multifidelity(cfg, top_k_params=None, skip_gp=False, skip_optimize=False,
                             vary_params=None):
    """Stage 2 with AR1 multi-fidelity GP (emukit + GPy).

    High-fidelity data comes from cfg.paths.preprocess_dir (same as the
    single-fidelity path).  Low-fidelity data comes from
    cfg.multi_fidelity.low_fidelity_dir.

    Y scalers are fitted on high-fidelity data only and applied to both
    fidelities.  Obs and the cost function are identical to single-fidelity.
    """
    from autotune_gp.gp_multifidelity import MultiFidelityGPWrapper, align_Y_to_hf_layout

    mf      = cfg.multi_fidelity
    out_dir = Path(cfg.paths.preprocess_dir)
    lf_dir  = Path(mf.low_fidelity_dir)
    var_names = list(cfg.data.variables)

    def _ts(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # ------------------------------------------------------------------
    _ts("=== Stage 2 (multi-fidelity): Load data ===")
    gp_high    = load_gp_proj(str(out_dir / "gp_proj.pkl"))
    X_high     = gp_high["X_train"]
    Y_high     = gp_high["Y_train"]
    _ts(f"  HF: X={np.asarray(X_high).shape}, Y={Y_high.shape}")

    gp_low     = load_gp_proj(str(lf_dir / "gp_proj.pkl"))
    X_low      = gp_low["X_train"]
    Y_low      = gp_low["Y_train"]
    _ts(f"  LF: X={np.asarray(X_low).shape}, Y={Y_low.shape}")

    # ------------------------------------------------------------------
    hf_mask_path = out_dir / "column_mask.pkl"
    lf_mask_path = lf_dir  / "column_mask.pkl"
    if hf_mask_path.exists() and lf_mask_path.exists():
        _ts("=== Stage 2 (multi-fidelity): Align LF columns to HF layout ===")
        with open(hf_mask_path, "rb") as f:
            hf_mask = pickle.load(f)
        with open(lf_mask_path, "rb") as f:
            lf_mask = pickle.load(f)
        Y_low = align_Y_to_hf_layout(Y_low, lf_mask, hf_mask)
        _ts(f"  LF Y after alignment: {Y_low.shape}")
    elif Y_low.shape[1] != Y_high.shape[1]:
        raise ValueError(
            f"LF and HF feature counts differ ({Y_low.shape[1]} vs {Y_high.shape[1]}) "
            "but no column_mask.pkl found to align them. "
            "Re-run stage 1 with the current code to generate column_mask.pkl."
        )

    # ------------------------------------------------------------------
    hf_default_ZRG = _load_default_zrg(out_dir)
    lf_default_ZRG = _load_default_zrg(lf_dir)
    if lf_default_ZRG is not None and hf_mask_path.exists() and lf_mask_path.exists():
        lf_default_ZRG = align_Y_to_hf_layout(
            lf_default_ZRG[np.newaxis], lf_mask, hf_mask
        )[0]

    # ------------------------------------------------------------------
    if cfg.runtime.train_gp and not skip_gp:
        _ts("=== Stage 2 (multi-fidelity): K-fold evaluation ===")
        from autotune_gp.evaluate import run_kfold_evaluation_mf
        _kfold_n_zonal   = Y_high.shape[1] // (
            len(cfg.preprocess.snapshots) if (cfg.preprocess and cfg.preprocess.snapshots) else 2
        ) - len(cfg.data.regions_list) - 1
        _kfold_lats      = _zonal_center_lats(cfg, _kfold_n_zonal)
        _kfold_feat_lbls = (
            ([f"{c:.0f}" for c in _kfold_lats] if _kfold_lats
             else [str(i) for i in range(_kfold_n_zonal)])
            + list(cfg.data.regions_list) + ["global"]
        ) * (len(cfg.preprocess.snapshots) if (cfg.preprocess and cfg.preprocess.snapshots) else 2)
        _kfold_ckpt_dir = str(Path(cfg.paths.output_dir) / "kfold_checkpoints")
        run_kfold_evaluation_mf(
            X_high, Y_high,
            X_low,  Y_low,
            var_names=var_names,
            k=5,
            feat_labels=_kfold_feat_lbls,
            checkpoint_dir=_kfold_ckpt_dir,
            n_workers=int(os.environ.get("KFOLD_WORKERS", 0)),
        )

        if top_k_params is not None:
            param_names_full = (list(X_high.columns)
                                if hasattr(X_high, "columns")
                                else [str(i) for i in range(np.asarray(X_high).shape[1])])
            top_idx, _ = select_top_params_hf(
                X_high, Y_high, var_names,
                k=top_k_params, param_names=param_names_full,
            )
            X_high_red = (X_high.iloc[:, top_idx]
                          if hasattr(X_high, "iloc") else np.asarray(X_high)[:, top_idx])
            X_low_red  = (X_low.iloc[:, top_idx]
                          if hasattr(X_low,  "iloc") else np.asarray(X_low)[:, top_idx])
            _ts(f"=== Stage 2 (multi-fidelity): K-fold with top-{top_k_params} params ===")
            run_kfold_evaluation_mf(
                X_high_red, Y_high,
                X_low_red,  Y_low,
                var_names=var_names,
                k=5,
                feat_labels=_kfold_feat_lbls,
            )
    else:
        _ts("=== Stage 2 (multi-fidelity): K-fold and GP training skipped (--skip-gp) ===" if skip_gp
            else "=== Stage 2 (multi-fidelity): K-fold evaluation skipped (train_gp=false) ===")
        top_idx = None

    # Compute top_idx even when the k-fold branch above was skipped (either
    # because train_gp=False in the config, or --skip-gp was passed on the
    # CLI) so the surrogate still uses the reduced params.
    if top_k_params is not None and (not cfg.runtime.train_gp or skip_gp):
        param_names_full = (list(X_high.columns)
                            if hasattr(X_high, "columns")
                            else [str(i) for i in range(np.asarray(X_high).shape[1])])
        top_idx, _ = select_top_params_hf(
            X_high, Y_high, var_names,
            k=top_k_params, param_names=param_names_full,
        )

    # ------------------------------------------------------------------
    _ts("=== Stage 2 (multi-fidelity): Normalise ===")
    with open(out_dir / "scalers.pkl", "rb") as f:
        saved = pickle.load(f)
    X_sc      = saved["X_pipeline"]
    Y_scalers = [saved[f"Y_pipeline_{v}"] for v in var_names]

    # Normalise with the full parameter set — the saved X scaler was fit on all
    # n_params physical bounds, so it must be handed every column. Parameter
    # reduction happens afterwards; MinMaxScaler is per-feature, so
    # normalise-then-slice is identical to slice-then-normalise.
    X_high_norm = X_sc.transform(X_high)
    X_low_norm  = X_sc.transform(X_low)

    # Apply parameter reduction for surrogate training and optimization
    if top_k_params is not None:
        X_high = (X_high.iloc[:, top_idx]
                  if hasattr(X_high, "iloc") else np.asarray(X_high)[:, top_idx])
        X_low  = (X_low.iloc[:, top_idx]
                  if hasattr(X_low,  "iloc") else np.asarray(X_low)[:, top_idx])
        X_high_norm = np.asarray(X_high_norm)[:, top_idx]
        X_low_norm  = np.asarray(X_low_norm)[:, top_idx]
        _ts(f"  Surrogate and optimization will use top-{top_k_params} params.")

    def _norm_Y(Y):
        return np.stack(
            [Y_scalers[j].transform(Y[:, :, j]) for j in range(len(var_names))],
            axis=0,
        ).transpose(1, 2, 0)

    Y_high_norm = _norm_Y(Y_high)
    Y_low_norm  = _norm_Y(Y_low)
    _ts(f"  HF norm: X={X_high_norm.shape}, Y={Y_high_norm.shape}")
    _ts(f"  LF norm: X={X_low_norm.shape},  Y={Y_low_norm.shape}")

    # ------------------------------------------------------------------
    obs_loaded = load_obs(str(out_dir / "obs.pkl"))
    zrg_obs    = obs_loaded["zrg_obs"]
    obs_parts  = split_zrg_obs(zrg_obs, n_vars=len(var_names))
    obs_norm   = transform_obs(obs_parts, Y_scalers)

    # ------------------------------------------------------------------
    _gp_ckpt = Path(cfg.paths.output_dir) / "mf_gp_trained.pkl"
    _gp_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if _gp_ckpt.exists():
        _ts(f"=== Stage 2 (multi-fidelity): Loading saved GP from {_gp_ckpt} ===")
        with open(_gp_ckpt, "rb") as f:
            gp = pickle.load(f)
        _ts("  GP loaded.")
    else:
        if skip_gp:
            raise FileNotFoundError(
                f"--skip-gp was set but no saved GP found at {_gp_ckpt}. "
                "Run without --skip-gp first to train and save the GP."
            )
        _ts("=== Stage 2 (multi-fidelity): Train AR1 GP ===")
        _t_gp = time.time()
        gp = MultiFidelityGPWrapper(X_low_norm, Y_low_norm, X_high_norm, Y_high_norm)
        gp.train()
        _ts(f"  AR1 GP training complete in {(time.time() - _t_gp)/60:.1f} min.")
        with open(_gp_ckpt, "wb") as f:
            pickle.dump(gp, f)
        _ts(f"  Trained GP saved → {_gp_ckpt}")

    if skip_optimize:
        _ts("=== Stage 2 (multi-fidelity): Optimization skipped (--skip-optimize) ===")
        return

    _ts("=== Stage 2 (multi-fidelity): Save hyperparameters ===")
    n_regions = len(cfg.data.regions_list)
    n_snaps   = len(cfg.preprocess.snapshots) if (cfg.preprocess and cfg.preprocess.snapshots) else 2
    n_zonal   = Y_high.shape[1] // n_snaps - n_regions - 1
    _zonal_lats = _zonal_center_lats(cfg, n_zonal)
    _snap_labels = (([f"{c:.0f}" for c in _zonal_lats] if _zonal_lats else
                     [str(i) for i in range(n_zonal)])
                    + list(cfg.data.regions_list) + ["global"]) * n_snaps
    hp_dir = (Path(cfg.diagnostics.output_dir) if cfg.diagnostics.output_dir
              else Path(cfg.paths.output_dir) / "diagnostics")
    gp.save_hyperparameters(
        path=str(hp_dir / "hyperparameters.pkl"),
        var_names=var_names,
        feat_labels=_snap_labels,
    )

    # ------------------------------------------------------------------
    print("=== Stage 2 (multi-fidelity): Optimize ===")
    backend   = get_backend(cfg.runtime.backend, cfg.runtime.device)
    n_regions = len(cfg.data.regions_list)
    n_snaps   = len(cfg.preprocess.snapshots) if (cfg.preprocess and cfg.preprocess.snapshots) else 2
    n_zonal   = Y_high.shape[1] // n_snaps - n_regions - 1
    var_w            = cfg.weights.variables
    zrg_w            = cfg.weights.zrg
    dy_w             = cfg.weights.dy
    zonal_weights    = cfg.weights.zonal_weights
    regional_weights = cfg.weights.regional_weights

    param_names = list(X_high.columns) if hasattr(X_high, "columns") else None
    constraint_pairs = []
    for pair in (cfg.optimize.param_ordering_constraints or []):
        low_name, high_name = pair
        if param_names and low_name in param_names and high_name in param_names:
            constraint_pairs.append((param_names.index(low_name), param_names.index(high_name)))

    # --- Optional subset optimization -------------------------------------
    # Optimize only `vary_params`, freezing every other parameter at the
    # control-run default. This reuses the full-dimension GP (no retraining):
    # the cost function expands the reduced vector back to full length, filling
    # frozen slots with the default, before calling gp.predict. Constraints are
    # checked on the full vector.
    x_default_norm, vary_idx = None, None
    if vary_params:
        if param_names is None:
            raise ValueError("--vary-params needs named parameters (X_train must be a DataFrame).")
        missing = [p for p in vary_params if p not in param_names]
        if missing:
            raise ValueError(f"--vary-params names not in the parameter set: {missing}")
        with open(out_dir / "default_params.pkl", "rb") as f:
            _dp = pickle.load(f)
        if list(_dp["names"]) != list(param_names):
            raise ValueError("default_params.pkl parameter order does not match the GP's.")
        x_default_norm = np.asarray(_dp["norm"], dtype=float)
        vary_idx = [param_names.index(p) for p in vary_params]
        _ts(f"=== Subset optimization: varying {len(vary_idx)} params, "
            f"freezing {len(param_names) - len(vary_idx)} at control-run default "
            f"({_dp['default_name']}) ===")
        for p in vary_params:
            _ts(f"    vary {p}")

        # Clip each frozen default to the sampled range the GP actually covers
        # (HF ∪ LF — the same support bounds_from_data uses), so no cost
        # evaluation ever asks the GP to extrapolate. A few control-run defaults
        # sit a hair past the PPE's sampled edge (they are round numbers at the
        # physical bounds); this snaps them to the nearest sampled value.
        _stacked = np.vstack([np.asarray(X_high_norm, dtype=float),
                              np.asarray(X_low_norm, dtype=float)])
        _samp_lo, _samp_hi = _stacked.min(axis=0), _stacked.max(axis=0)
        _clipped = np.clip(x_default_norm, _samp_lo, _samp_hi)
        for i in range(len(param_names)):
            if not np.isclose(_clipped[i], x_default_norm[i]):
                _ts(f"    clip {param_names[i]}: default norm "
                    f"{x_default_norm[i]:.4f} -> {_clipped[i]:.4f} (nearest sampled)")
        x_default_norm = _clipped

    def cost_fn(x):
        x = np.asarray(x, dtype=float)
        if vary_idx is not None:
            full = x_default_norm.copy()
            full[vary_idx] = x
        else:
            full = x
        for low_idx, high_idx in constraint_pairs:
            if full[low_idx] - full[high_idx] > 0:
                return 1e2 + 1e2 * (full[low_idx] - full[high_idx])
        m, _ = gp.predict(full)
        return zrg_cost_function_mae_weighted(
            m, obs_norm, var_w, zrg_w, dy_w,
            n_zonal=n_zonal, n_regions=n_regions, backend=backend,
            var_names=list(var_names),
            zonal_weights=zonal_weights, regional_weights=regional_weights,
        )

    # Resolve full-dimension bounds, then slice to the varied params.
    n_full = len(param_names) if param_names else cfg.optimize.n_params
    bl_full, bh_full = _resolve_opt_bounds(
        cfg, [X_high_norm, X_low_norm], param_names, n_full, log=_ts)
    if vary_idx is not None:
        n_opt_params = len(vary_idx)
        bounds_low  = [bl_full[i] for i in vary_idx]
        bounds_high = [bh_full[i] for i in vary_idx]
    else:
        n_opt_params = n_full
        bounds_low, bounds_high = bl_full, bh_full

    results, top_rows, csv_path = optimize_parallel(
        cost_fn=cost_fn,
        n_params=n_opt_params,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        seed=cfg.optimize.seed,
        n_xstarts=cfg.optimize.n_xstarts,
        niter=cfg.optimize.niter,
        method=cfg.optimize.method,
        out_dir=cfg.paths.output_dir,
        max_workers=cfg.optimize.max_workers,
        executor=cfg.optimize.executor,
        checkpoint_dir=str(Path(cfg.paths.output_dir) / "optimize_checkpoints"),
    )
    print(f"Done. Results: {csv_path}")

    # Expand reduced results back to full parameter dimension so diagnostics
    # (barcode, ZRG projection, projection_data.pkl) see all 19 params — the
    # frozen ones appear as constant columns at their default value.
    if vary_idx is not None:
        full_res = np.tile(np.append(x_default_norm, 0.0), (results.shape[0], 1))
        full_res[:, vary_idx] = results[:, :-1]
        full_res[:, -1]       = results[:, -1]
        results = full_res

    if cfg.diagnostics.enabled:
        from autotune_gp.diagnostics import run_diagnostics
        diag_dir = Path(cfg.diagnostics.output_dir) if cfg.diagnostics.output_dir \
                   else Path(cfg.paths.output_dir) / "diagnostics"
        print("=== Stage 2 (multi-fidelity): Diagnostics ===")
        run_diagnostics(
            results=results,
            top_rows=top_rows,
            gp=gp,
            Y_train_ZRG=Y_high,
            Y_scalers=Y_scalers,
            obs_parts=obs_parts,
            param_names=param_names,
            var_names=var_names,
            n_zonal=n_zonal,
            n_regions=n_regions,
            regions_list=cfg.data.regions_list,
            out_dir=diag_dir,
            suffix=f"_seed{cfg.optimize.seed}",
            n_snaps=n_snaps,
            Y_low_ZRG=Y_low,
            zonal_center_lats=_zonal_center_lats(cfg, n_zonal),
            hf_default_ZRG=hf_default_ZRG,
            lf_default_ZRG=lf_default_ZRG,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only stage 1 (preprocess) or stage 2 (surrogate+optimize). "
             "Default: run both.",
    )
    p.add_argument(
        "--preprocess-mode",
        choices=["ppe", "obs", "both"],
        default="both",
        help=(
            "Controls what stage 1 preprocesses. "
            "'both' (default): process PPE and obs together. "
            "'ppe': process PPE only — saves gp_proj/scalers/kfold/column_mask pickles, "
            "skips obs.pkl. Use on systems where obs files are not yet available. "
            "'obs': process obs only — loads column_mask.pkl from a prior 'ppe' run "
            "to align columns, then saves obs.pkl."
        ),
    )
    p.add_argument("--plot", action="store_true", default=False,
                   help="Save a ZRG diagnostic PNG at the end of stage 1.")
    p.add_argument("--seed",      type=int, default=None,
                   help="Override optimize.seed from config.")
    p.add_argument("--n-xstarts", type=int, default=None,
                   help="Override optimize.n_xstarts from config.")
    p.add_argument("--top-k-params", type=int, default=None,
                   metavar="K",
                   help="(Multi-fidelity only) Select the top-K parameters by HF "
                        "Pearson correlation before training. Runs both the full "
                        "19-param k-fold and a reduced K-param k-fold for comparison, "
                        "then trains the surrogate and runs optimization in the reduced "
                        "K-param space.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override paths.output_dir from config.")
    p.add_argument("--skip-gp", action="store_true", default=False,
                   help="(Multi-fidelity only) Skip k-fold and GP training; load saved "
                        "mf_gp_trained.pkl and go straight to optimization.")
    p.add_argument("--skip-optimize", action="store_true", default=False,
                   help="(Multi-fidelity only) Run k-fold and GP training, then stop "
                        "before optimization.")
    p.add_argument("--vary-params", type=str, default=None, metavar="P1,P2,...",
                   help="(Multi-fidelity only) Optimize only these comma-separated "
                        "parameters, freezing all others at the control-run default. "
                        "Reuses the full GP (use with --skip-gp); no retraining. "
                        "Requires default_params.pkl in the preprocess dir. Mutually "
                        "exclusive with --top-k-params.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.optimize.seed = args.seed
    if args.n_xstarts is not None:
        cfg.optimize.n_xstarts = args.n_xstarts
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir
    top_k_params = args.top_k_params
    vary_params = ([p.strip() for p in args.vary_params.split(",") if p.strip()]
                   if args.vary_params else None)
    if top_k_params is not None and vary_params:
        raise ValueError("--top-k-params and --vary-params are mutually exclusive.")
    if not cfg.paths.preprocess_dir:
        raise ValueError("cfg.paths.preprocess_dir must be set in the config.")

    use_mf = cfg.multi_fidelity is not None

    if args.stage == 1:
        run_stage1(cfg, preprocess_mode=args.preprocess_mode, make_plots=args.plot)
    elif args.stage == 2:
        if use_mf:
            run_stage2_multifidelity(cfg, top_k_params=top_k_params,
                                     skip_gp=args.skip_gp,
                                     skip_optimize=args.skip_optimize,
                                     vary_params=vary_params)
        else:
            run_stage2(cfg)
    else:
        run_stage1(cfg, preprocess_mode=args.preprocess_mode, make_plots=args.plot)
        if use_mf:
            run_stage2_multifidelity(cfg, top_k_params=top_k_params)
        else:
            out_dir = Path(cfg.paths.preprocess_dir)
            run_stage2(cfg, _preprocess_pkls={
                "obs_pkl":     str(out_dir / "obs.pkl"),
                "gp_proj_pkl": str(out_dir / "gp_proj.pkl"),
            })


if __name__ == "__main__":
    main()
