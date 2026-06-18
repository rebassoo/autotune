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
import pickle
import sys
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
from autotune_gp.evaluate import run_kfold_evaluation

from preprocessing.pipeline import (
    build_run_list,
    load_and_mask,
    compute_zrg,
    drop_nan_zrg_features,
    build_run_list_generic,
    load_and_mask_generic,
    compute_zrg_generic,
    compute_obs_zrg_generic,
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

    print("=== Stage 2: Train GP surrogate (full dataset) ===")
    gp = GPWrapper(X_train_norm, Y_train_norm)
    if cfg.runtime.train_gp:
        gp.train(tf_determinism=cfg.runtime.tf_determinism)

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

    results, top_rows, csv_path = optimize_parallel(
        cost_fn=cost_fn,
        n_params=cfg.optimize.n_params,
        bounds_low=cfg.optimize.bounds["low"],
        bounds_high=cfg.optimize.bounds["high"],
        seed=cfg.optimize.seed,
        n_xstarts=cfg.optimize.n_xstarts,
        niter=cfg.optimize.niter,
        method=cfg.optimize.method,
        out_dir=cfg.paths.output_dir,
        max_workers=cfg.optimize.max_workers,
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
        )


def run_stage2_multifidelity(cfg):
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

    # ------------------------------------------------------------------
    print("=== Stage 2 (multi-fidelity): Load data ===")
    gp_high    = load_gp_proj(str(out_dir / "gp_proj.pkl"))
    X_high     = gp_high["X_train"]
    Y_high     = gp_high["Y_train"]
    print(f"  HF: X={np.asarray(X_high).shape}, Y={Y_high.shape}")

    gp_low     = load_gp_proj(str(lf_dir / "gp_proj.pkl"))
    X_low      = gp_low["X_train"]
    Y_low      = gp_low["Y_train"]
    print(f"  LF: X={np.asarray(X_low).shape}, Y={Y_low.shape}")

    # ------------------------------------------------------------------
    hf_mask_path = out_dir / "column_mask.pkl"
    lf_mask_path = lf_dir  / "column_mask.pkl"
    if hf_mask_path.exists() and lf_mask_path.exists():
        print("=== Stage 2 (multi-fidelity): Align LF columns to HF layout ===")
        with open(hf_mask_path, "rb") as f:
            hf_mask = pickle.load(f)
        with open(lf_mask_path, "rb") as f:
            lf_mask = pickle.load(f)
        Y_low = align_Y_to_hf_layout(Y_low, lf_mask, hf_mask)
        print(f"  LF Y after alignment: {Y_low.shape}")
    elif Y_low.shape[1] != Y_high.shape[1]:
        raise ValueError(
            f"LF and HF feature counts differ ({Y_low.shape[1]} vs {Y_high.shape[1]}) "
            "but no column_mask.pkl found to align them. "
            "Re-run stage 1 with the current code to generate column_mask.pkl."
        )

    # ------------------------------------------------------------------
    if cfg.runtime.train_gp:
        print("=== Stage 2 (multi-fidelity): K-fold evaluation ===")
        from autotune_gp.evaluate import run_kfold_evaluation_mf
        run_kfold_evaluation_mf(
            X_high, Y_high,
            X_low,  Y_low,
            var_names=var_names,
            k=5,
        )
    else:
        print("=== Stage 2 (multi-fidelity): K-fold evaluation skipped (train_gp=false) ===")

    # ------------------------------------------------------------------
    print("=== Stage 2 (multi-fidelity): Normalise ===")
    with open(out_dir / "scalers.pkl", "rb") as f:
        saved = pickle.load(f)
    X_sc      = saved["X_pipeline"]
    Y_scalers = [saved[f"Y_pipeline_{v}"] for v in var_names]

    X_high_norm = X_sc.transform(X_high)
    X_low_norm  = X_sc.transform(X_low)

    def _norm_Y(Y):
        return np.stack(
            [Y_scalers[j].transform(Y[:, :, j]) for j in range(len(var_names))],
            axis=0,
        ).transpose(1, 2, 0)

    Y_high_norm = _norm_Y(Y_high)
    Y_low_norm  = _norm_Y(Y_low)
    print(f"  HF norm: X={X_high_norm.shape}, Y={Y_high_norm.shape}")
    print(f"  LF norm: X={X_low_norm.shape},  Y={Y_low_norm.shape}")

    # ------------------------------------------------------------------
    obs_loaded = load_obs(str(out_dir / "obs.pkl"))
    zrg_obs    = obs_loaded["zrg_obs"]
    obs_parts  = split_zrg_obs(zrg_obs, n_vars=len(var_names))
    obs_norm   = transform_obs(obs_parts, Y_scalers)

    # ------------------------------------------------------------------
    print("=== Stage 2 (multi-fidelity): Train AR1 GP ===")
    gp = MultiFidelityGPWrapper(X_low_norm, Y_low_norm, X_high_norm, Y_high_norm)
    gp.train()

    print("=== Stage 2 (multi-fidelity): Save hyperparameters ===")
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

    def cost_fn(x):
        x = np.asarray(x, dtype=float)
        for low_idx, high_idx in constraint_pairs:
            if x[low_idx] - x[high_idx] > 0:
                return 1e2 + 1e2 * (x[low_idx] - x[high_idx])
        m, _ = gp.predict(x)
        return zrg_cost_function_mae_weighted(
            m, obs_norm, var_w, zrg_w, dy_w,
            n_zonal=n_zonal, n_regions=n_regions, backend=backend,
            var_names=list(var_names),
            zonal_weights=zonal_weights, regional_weights=regional_weights,
        )

    results, top_rows, csv_path = optimize_parallel(
        cost_fn=cost_fn,
        n_params=cfg.optimize.n_params,
        bounds_low=cfg.optimize.bounds["low"],
        bounds_high=cfg.optimize.bounds["high"],
        seed=cfg.optimize.seed,
        n_xstarts=cfg.optimize.n_xstarts,
        niter=cfg.optimize.niter,
        method=cfg.optimize.method,
        out_dir=cfg.paths.output_dir,
        max_workers=cfg.optimize.max_workers,
    )
    print(f"Done. Results: {csv_path}")

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
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.optimize.seed = args.seed
    if args.n_xstarts is not None:
        cfg.optimize.n_xstarts = args.n_xstarts
    if not cfg.paths.preprocess_dir:
        raise ValueError("cfg.paths.preprocess_dir must be set in the config.")

    use_mf = cfg.multi_fidelity is not None

    if args.stage == 1:
        run_stage1(cfg, preprocess_mode=args.preprocess_mode, make_plots=args.plot)
    elif args.stage == 2:
        if use_mf:
            run_stage2_multifidelity(cfg)
        else:
            run_stage2(cfg)
    else:
        run_stage1(cfg, preprocess_mode=args.preprocess_mode, make_plots=args.plot)
        if use_mf:
            run_stage2_multifidelity(cfg)
        else:
            out_dir = Path(cfg.paths.preprocess_dir)
            run_stage2(cfg, _preprocess_pkls={
                "obs_pkl":     str(out_dir / "obs.pkl"),
                "gp_proj_pkl": str(out_dir / "gp_proj.pkl"),
            })


if __name__ == "__main__":
    main()
