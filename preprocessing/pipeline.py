"""
Core preprocessing functions shared by scripts/run_end_to_end.py and
scripts/run_two_stage.py.

Each function mirrors one of the standalone 01-04 scripts but returns
data rather than writing to disk, so callers decide whether to save.

Variable names, obs filenames, and simulation field names are all driven
by the `variables` dict from cfg.preprocess.variables (VariableCfg objects).
Adding, removing, or renaming variables requires only a config change.
"""
from __future__ import annotations

import glob as _glob_module
import json as _json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Constants that are independent of variable choice
# ---------------------------------------------------------------------------
DY1_BAD_RUNS = [
    "m0024", "m0025", "m0061", "optmar22hd",
    "m0262", "m0263", "m0264", "m0266", "m0267", "m0270",
    "m0272", "m0274", "m0275", "m0279", "m0289", "m0290",
    "m0292", "m0293", "m0294", "m0295", "m0296", "m0299",
    "m0300",
    "optmar15seed0", "optmar27a", "optmar20dayAll",
    "optmar20day2-fail", "optmar15b", "optmar20day2-ltend",
    "m0230", "optmar20day5",
]
DY2_BAD_RUNS = ["m0230", "optmar20day5", "m0024", "m0025", "m0061", "optmar22hd",
                "optmar15", "optmar20day2"]

REGIONS_LIST = [
    "poles",
    "extratropical_land",
    "extratropical_ocean",
    "tropical_land",
    "ascending_tropical_ocean",
    "descending_tropical_ocean",
]


# ---------------------------------------------------------------------------
# Shared helpers used by both DY1/DY2 and generic pipelines
# ---------------------------------------------------------------------------

def _raw_fields_needed(variables):
    needed = set()
    for var_cfg in variables.values():
        if var_cfg.sim_components:
            needed.update(var_cfg.sim_components)
        else:
            needed.add(var_cfg.sim_field)
    return needed


def _prep_sim(ds, prefix, variables):
    """Drop unneeded vars, compute derived fields, rename with snapshot prefix."""
    raw_needed = _raw_fields_needed(variables)
    to_drop = [v for v in ds.data_vars if v not in raw_needed]
    ds = ds.drop_vars(to_drop, errors="ignore")
    ds = ds.drop_vars("p_levs", errors="ignore")
    direct_fields = {vc.sim_field for vc in variables.values() if not vc.sim_components}
    for var_cfg in variables.values():
        if var_cfg.sim_components:
            ds[var_cfg.sim_field] = sum(ds[c] for c in var_cfg.sim_components)
            for comp in var_cfg.sim_components:
                if comp not in direct_fields:
                    ds = ds.drop_vars(comp, errors="ignore")
    return ds.rename({v: f"{prefix}_{v}" for v in ds.data_vars})


def _mask_sim(ds, prefix, variables, obs):
    """Mask sim grid cells where the corresponding obs is NaN."""
    ds = ds.copy(deep=True)
    for var_name, var_cfg in variables.items():
        field_key = f"{prefix}_{var_cfg.sim_field}"
        obs_key   = f"{prefix}_{var_name}_obs"
        ds[field_key] = ds[field_key].where(~np.isnan(obs[obs_key]))
    return ds


# ---------------------------------------------------------------------------
# Step 1: Build run lists
# ---------------------------------------------------------------------------
def build_run_list(
    params_json: str,
    DY1_sim_dir: str,
    DY1_nc_suffix: str,
    DY2_sim_dir: str,
    DY2_nc_suffix: str,
) -> dict:
    """
    Enumerate runs, filter bad/incomplete runs and p3_ice_sed_knob < 1,
    return the intersection of valid DY1 and DY2 runs.
    """
    ppe_params_all = pd.read_json(params_json)

    def _enumerate(sim_dir, bad_runs):
        folders = []
        for m in range(0, 301):
            folder = "m{:04}".format(m)
            if os.path.exists(sim_dir + folder):
                folders.append(m)
        folders = ["m{:04}".format(m) for m in folders]
        for f in os.listdir(sim_dir):
            if f.startswith("opt"):
                folders.append(f)
        for bad in bad_runs:
            if bad in folders:
                folders.remove(bad)
        folders = sorted(folders)
        return folders

    def _filter_ice_sed(folders, sim_dir, nc_suffix):
        check = np.zeros(len(folders), dtype=bool)
        for i, member in enumerate(folders):
            if float(ppe_params_all["p3_ice_sed_knob"][member]) >= 1.0:
                check[i] = True
        names = np.array(folders)[check]
        files = np.array([sim_dir + f + "/" + nc_suffix for f in folders])[check]
        return names, files

    DY1_folders = _enumerate(DY1_sim_dir, DY1_BAD_RUNS)
    DY2_folders = _enumerate(DY2_sim_dir, DY2_BAD_RUNS)

    DY1_sim_names, DY1_filename_list = _filter_ice_sed(
        DY1_folders, DY1_sim_dir, DY1_nc_suffix
    )
    DY2_sim_names, DY2_filename_list = _filter_ice_sed(
        DY2_folders, DY2_sim_dir, DY2_nc_suffix
    )

    sim_names = [s for s in DY1_sim_names if s in DY2_sim_names]
    ppe_params = ppe_params_all.loc[sim_names]   # .loc preserves sim_names order
    assert list(ppe_params.index) == list(sim_names), \
        "ppe_params row order does not match sim_names"

    print(f"DY1 runs: {len(DY1_sim_names)}, DY2 runs: {len(DY2_sim_names)}, "
          f"intersection: {len(sim_names)}")

    return {
        "sim_names": sim_names,
        "DY1_sim_names": DY1_sim_names,
        "DY2_sim_names": DY2_sim_names,
        "DY1_filename_list": DY1_filename_list,
        "DY2_filename_list": DY2_filename_list,
        "ppe_params": ppe_params,
        "ppe_params_all": ppe_params_all,
    }


# ---------------------------------------------------------------------------
# Step 2: Load simulations and observations, mask
# ---------------------------------------------------------------------------
def load_and_mask(
    run_list: dict,
    DY1_obs_dir: str,
    DY2_obs_dir: str,
    variables: dict,          # {var_name: VariableCfg}
) -> dict:
    """
    Open simulation netCDFs, load obs, extract target variables,
    mask sim cells where obs is NaN, merge DY1+DY2 into one dataset.

    `variables` drives which simulation fields to keep, which obs files to
    open, and any obs unit scaling. Adding a new variable requires only a
    new entry in the config — no changes here.

    Returns dict with keys:
        ppe_dataset_small,
        DY1_{var_name}_obs and DY2_{var_name}_obs for each var in variables
    """
    sim_names     = run_list["sim_names"]
    DY1_sim_names = run_list["DY1_sim_names"]
    DY2_sim_names = run_list["DY2_sim_names"]
    DY1_files     = run_list["DY1_filename_list"]
    DY2_files     = run_list["DY2_filename_list"]

    # --- Load simulations ---
    DY1_ds = (
        xr.open_mfdataset(DY1_files, concat_dim="run_label", combine="nested")
        .assign_coords(run_label=("run_label", DY1_sim_names))
        .squeeze("time")
    )
    DY2_ds = (
        xr.open_mfdataset(DY2_files, concat_dim="run_label", combine="nested")
        .assign_coords(run_label=("run_label", DY2_sim_names))
        .squeeze("time")
    )

    # --- Load observations (driven entirely by variables config) ---
    obs = {}
    for var_name, var_cfg in variables.items():
        dy1 = (
            xr.open_dataset(DY1_obs_dir + var_cfg.obs_file_DY1)
            .variables[var_cfg.obs_nc_var]
            .squeeze("time")
        ) * var_cfg.obs_scale
        dy2 = (
            xr.open_dataset(DY2_obs_dir + var_cfg.obs_file_DY2)
            .variables[var_cfg.obs_nc_var]
            .squeeze("time")
        ) * var_cfg.obs_scale
        obs[f"DY1_{var_name}_obs"] = dy1
        obs[f"DY2_{var_name}_obs"] = dy2

    # --- Extract target variables and mask simulations ---
    DY1_small = _prep_sim(DY1_ds, "DY1", variables)
    DY2_small = _prep_sim(DY2_ds, "DY2", variables)
    DY1_small = _mask_sim(DY1_small, "DY1", variables, obs)
    DY2_small = _mask_sim(DY2_small, "DY2", variables, obs)

    # --- Merge DY1 + DY2, filter to intersection ---
    ppe_dataset_small = (
        DY1_small.sel(run_label=sim_names)
        .combine_first(DY2_small.sel(run_label=sim_names))
    )

    # Verify run_label ordering matches sim_names throughout
    assert list(DY1_small.sel(run_label=sim_names).run_label.values) == sim_names, \
        "DY1 run_label order does not match sim_names"
    assert list(DY2_small.sel(run_label=sim_names).run_label.values) == sim_names, \
        "DY2 run_label order does not match sim_names"

    return {"ppe_dataset_small": ppe_dataset_small, **obs}


# ---------------------------------------------------------------------------
# Step 3: Compute ZRG averages
# ---------------------------------------------------------------------------
def _safe_weighted_mean(d, a):
    """Area-weighted mean excluding NaN cells from both numerator and denominator."""
    valid_a = np.where(np.isnan(d), np.nan, a)
    denom = np.nansum(valid_a)
    return np.nansum(d * a) / denom if denom > 0 else np.nan


def _zonal_means(data, area, lat):
    lat_bands = np.linspace(-90, 90, 19)
    result = {}
    for i in range(len(lat_bands) - 1):
        mask = (lat >= lat_bands[i]) & (lat < lat_bands[i + 1]).squeeze()
        d = np.where(mask > 0, data.squeeze(), np.nan)
        a = np.where(mask > 0, area.squeeze(), np.nan)
        center = abs(lat_bands[i] - lat_bands[i + 1]) / 2 + lat_bands[i]
        result[center] = _safe_weighted_mean(d, a)
    return result


def _regional_means(data, area, regions_file):
    region_data = xr.open_dataset(regions_file)
    result = {}
    for reg in REGIONS_LIST:
        mask = region_data[reg].squeeze()
        d = np.where(mask > 0, data.squeeze(), np.nan)
        a = np.where(mask > 0, area.squeeze(), np.nan)
        result[reg] = _safe_weighted_mean(d, a)
    return result


def _global_mean(data, area):
    d = np.asarray(data).squeeze()
    a = np.asarray(area).squeeze()
    return _safe_weighted_mean(d, a)


def drop_nan_zrg_features(zrg_result, var_names, n_zonal, n_regions, regions_list,
                           explicit_drop_zonal=None):
    """
    Drop ZRG feature columns that are all-NaN across all runs for any variable
    or in the obs, plus any zonal bands explicitly listed by centre latitude in
    explicit_drop_zonal (e.g. [-85, -75, 85]).  Dropping is always symmetric:
    if a position is removed it is removed from both DY1 and DY2.
    Prints which bands/regions are dropped.

    Returns updated zrg_result and new n_zonal.
    """
    n_per_day = n_zonal + n_regions + 1
    n_feat = n_per_day * 2
    n_vars = len(var_names)

    # Build human-readable labels for each feature position
    lat_bands = np.linspace(-90, 90, n_zonal + 1)
    zonal_labels = [f"zonal {(lat_bands[i] + lat_bands[i+1]) / 2:.0f}deg"
                    for i in range(n_zonal)]
    per_day_labels = zonal_labels + list(regions_list) + ["global"]
    day_labels = ["DY1"] * n_per_day + ["DY2"] * n_per_day
    feature_labels = per_day_labels * 2

    # Find feature positions (0..n_feat-1) all-NaN for any variable (sim or obs)
    raw_nan = set()
    for var in var_names:
        df = zrg_result[f"{var}_zrg_ppedataset"]
        for i in range(n_feat):
            if df.iloc[:, i].isna().all():
                raw_nan.add(i)
    for v_idx in range(n_vars):
        obs_block = zrg_result["zrg_obs"].iloc[:, v_idx * n_feat:(v_idx + 1) * n_feat]
        for i in range(n_feat):
            if obs_block.iloc[:, i].isna().all():
                raw_nan.add(i)

    # Add explicitly requested zonal bands (matched by closest centre latitude)
    if explicit_drop_zonal:
        lat_bands = np.linspace(-90, 90, n_zonal + 1)
        centres = np.array([(lat_bands[i] + lat_bands[i + 1]) / 2 for i in range(n_zonal)])
        for target in explicit_drop_zonal:
            idx = int(np.argmin(np.abs(centres - target)))
            raw_nan.add(idx)
            print(f"  Explicitly dropping zonal band centre {centres[idx]:.0f}° "
                  f"(requested {target}°)")

    # Enforce symmetric dropping: DY1 and DY2 must remain structurally identical.
    # If position i in DY1 or position i in DY2 is all-NaN, drop both.
    nan_cols = set()
    for i in raw_nan:
        # Determine structural position (0..n_per_day-1)
        pos = i % n_per_day
        nan_cols.add(pos)            # DY1 position
        nan_cols.add(pos + n_per_day)  # DY2 mirror

    if not nan_cols:
        print("  No all-NaN ZRG feature columns found.")
        return zrg_result, n_zonal

    sorted_nan = sorted(nan_cols)
    print(f"  Dropping {len(sorted_nan)} all-NaN ZRG feature column(s) (symmetric DY1/DY2):")
    for i in sorted_nan:
        nan_vars = [v for v in var_names
                    if zrg_result[f"{v}_zrg_ppedataset"].iloc[:, i].isna().all()]
        print(f"    {day_labels[i]} {feature_labels[i]}"
              f"  (all-NaN for: {', '.join(nan_vars) if nan_vars else 'obs only'})")

    valid = [i for i in range(n_feat) if i not in nan_cols]

    # Update per-variable sim DataFrames
    updated = dict(zrg_result)
    for var in var_names:
        updated[f"{var}_zrg_ppedataset"] = zrg_result[f"{var}_zrg_ppedataset"].iloc[:, valid]

    # Update concatenated zrg_ppedataset and zrg_obs
    # (layout: n_vars blocks of n_feat columns each)
    all_drop = sorted({v_idx * n_feat + col for v_idx in range(n_vars) for col in sorted_nan})
    valid_global = [i for i in range(n_vars * n_feat) if i not in set(all_drop)]
    updated["zrg_ppedataset"] = zrg_result["zrg_ppedataset"].iloc[:, valid_global]
    updated["zrg_obs"]        = zrg_result["zrg_obs"].iloc[:, valid_global]

    # New n_zonal = DY1 zonal positions that were not dropped (DY2 mirrors, so count is same)
    new_n_zonal = len([i for i in range(n_zonal) if i not in nan_cols])
    print(f"  n_zonal updated: {n_zonal} → {new_n_zonal}")

    return updated, new_n_zonal


def _zrg_df(z_dict, r_dict, global_val, global_col):
    df = pd.concat(
        [pd.DataFrame.from_dict(z_dict, orient="index"),
         pd.DataFrame.from_dict(r_dict, orient="index")],
        axis=1,
    )
    df[global_col] = global_val
    return df


def compute_zrg(
    sim_names: list,
    ppe_dataset_small: xr.Dataset,
    obs_dict: dict,
    control_file: str,
    regions_file: str,
    variables: dict,          # {var_name: VariableCfg}
) -> dict:
    """
    Compute area-weighted zonal, regional, and global averages for all
    variables over both DY1 and DY2, for simulations and observations.

    Returns dict with keys:
        zrg_ppedataset, zrg_obs,
        {var_name}_zrg_ppedataset for each variable
    """
    ctrl = xr.open_dataset(control_file)
    area = ctrl.variables["area"][:]
    lat  = ctrl.variables["lat"][:]

    var_names = list(variables.keys())

    # --- Simulations ---
    per_var_zrg = {}
    for var_name, var_cfg in variables.items():
        sim_field = var_cfg.sim_field
        dy1_z, dy1_r, dy1_g = {}, {}, []
        dy2_z, dy2_r, dy2_g = {}, {}, []

        for run in sim_names:
            ds = ppe_dataset_small.sel(run_label=run)
            d1 = ds[f"DY1_{sim_field}"]
            d2 = ds[f"DY2_{sim_field}"]
            dy1_z[run] = _zonal_means(d1, area, lat)
            dy1_r[run] = _regional_means(d1, area, regions_file)
            dy1_g.append(_global_mean(d1, area))
            dy2_z[run] = _zonal_means(d2, area, lat)
            dy2_r[run] = _regional_means(d2, area, regions_file)
            dy2_g.append(_global_mean(d2, area))

        dy1_df = _zrg_df(dy1_z, dy1_r, dy1_g, "DY1_global")
        dy2_df = _zrg_df(dy2_z, dy2_r, dy2_g, "DY2_global")
        per_var_zrg[var_name] = pd.concat([dy1_df, dy2_df], axis=1)

    zrg_ppedataset = pd.concat(list(per_var_zrg.values()), axis=1)

    # --- Observations ---
    per_var_zrg_obs = {}
    for var_name in var_names:
        dy1_obs = obs_dict[f"DY1_{var_name}_obs"]
        dy2_obs = obs_dict[f"DY2_{var_name}_obs"]

        dy1_df = _zrg_df(
            {"obs": _zonal_means(dy1_obs, area, lat)},
            {"obs": _regional_means(dy1_obs, area, regions_file)},
            [_global_mean(dy1_obs, area)],
            "DY1_global",
        )
        dy2_df = _zrg_df(
            {"obs": _zonal_means(dy2_obs, area, lat)},
            {"obs": _regional_means(dy2_obs, area, regions_file)},
            [_global_mean(dy2_obs, area)],
            "DY2_global",
        )
        per_var_zrg_obs[var_name] = pd.concat([dy1_df, dy2_df], axis=1)

    zrg_obs = pd.concat(list(per_var_zrg_obs.values()), axis=1)

    print(f"zrg_ppedataset: {zrg_ppedataset.shape}, zrg_obs: {zrg_obs.shape}")

    return {
        "zrg_ppedataset": zrg_ppedataset,
        "zrg_obs":        zrg_obs,
        **{f"{v}_zrg_ppedataset": per_var_zrg[v] for v in var_names},
    }


# ---------------------------------------------------------------------------
# Generic N-snapshot pipeline (Step 1–3)
# ---------------------------------------------------------------------------

def build_run_list_generic(
    params_json: str,
    snapshots,                          # List[SnapshotCfg]
    param_names: Optional[List[str]] = None,
) -> dict:
    """
    Enumerate case directories for each snapshot, filter by file count, and
    return the intersection of valid members across all snapshots.

    Handles two params JSON formats:
      - list-of-lists (new): [[p0, p1, ...], ...], 1-indexed member dirs
      - dict-keyed (old): {"member_name": {"param": value, ...}, ...}

    Returns:
        sim_names:            sorted list of member dir names
        ppe_params:           DataFrame indexed by member name
        per_snapshot_files:   {label: {member: [file, ...]}}
    """
    with open(params_json) as fh:
        raw = _json.load(fh)

    if isinstance(raw, list):
        params_array = np.array(raw)     # (n_members, n_params)
        params_list_mode = True
    else:
        ppe_params_all = pd.read_json(params_json)
        params_list_mode = False

    per_snapshot_files: Dict[str, Dict[str, List[str]]] = {}

    for snap in snapshots:
        sim_dir = snap.sim_dir.rstrip("/") + "/"
        member_files: Dict[str, List[str]] = {}

        for entry in sorted(os.listdir(sim_dir)):
            entry_path = os.path.join(sim_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            if snap.nc_glob:
                run_dir = os.path.join(entry_path, "run")
                if not os.path.isdir(run_dir):
                    continue
                matched = sorted(_glob_module.glob(os.path.join(run_dir, snap.nc_glob)))
                if len(matched) < snap.min_files:
                    continue
                if snap.max_files is not None:
                    matched = matched[:snap.max_files]
                member_files[entry] = matched
            elif snap.nc_suffix:
                f = os.path.join(entry_path, snap.nc_suffix)
                if os.path.exists(f):
                    member_files[entry] = [f]

        per_snapshot_files[snap.label] = member_files
        print(f"  Snapshot {snap.label}: {len(member_files)} valid members")

    valid_sets = [set(per_snapshot_files[s.label].keys()) for s in snapshots]
    sim_names = sorted(set.intersection(*valid_sets))
    print(f"  Intersection across {len(snapshots)} snapshot(s): {len(sim_names)} members")

    if params_list_mode:
        def _member_to_idx(name):
            return int(name.split(".")[0]) - 1   # "001.xxx" → index 0

        indices = [_member_to_idx(m) for m in sim_names]
        data = params_array[indices]
        if param_names is None:
            param_names = [str(i) for i in range(data.shape[1])]
        ppe_params = pd.DataFrame(data, index=sim_names, columns=param_names)
    else:
        ppe_params = ppe_params_all.loc[sim_names]

    return {
        "sim_names":           sim_names,
        "ppe_params":          ppe_params,
        "per_snapshot_files":  per_snapshot_files,
    }


def load_and_mask_generic(
    run_list: dict,
    snapshots,              # List[SnapshotCfg]
    variables: dict,        # {var_name: VariableCfg}
) -> dict:
    """
    Load simulation files for each snapshot (averaging over multiple files),
    load observations where available, prep and mask, then merge all snapshots
    into one xr.Dataset keyed by run_label.

    Snapshots with obs_dir=None or obs_files={label: None} are loaded but
    not masked (masking requires valid obs).

    Returns dict with:
        ppe_dataset_small
        {label}_{var_name}_obs  for each snapshot/variable (may be None)
    """
    sim_names          = run_list["sim_names"]
    per_snapshot_files = run_list["per_snapshot_files"]

    snap_datasets: Dict[str, xr.Dataset] = {}
    for snap in snapshots:
        member_files = per_snapshot_files[snap.label]
        member_ds_list = []
        for member in sim_names:
            files = member_files[member]
            ds = xr.open_mfdataset(files, combine="by_coords")
            if "time" in ds.dims:
                weights = ds.time.dt.days_in_month.astype(float)
                ds = ds.weighted(weights).mean(dim="time")
            member_ds_list.append(ds)
        snap_ds = xr.concat(member_ds_list, dim="run_label").assign_coords(
            run_label=("run_label", sim_names)
        )
        snap_datasets[snap.label] = snap_ds

    # Load observations (None if not yet available)
    obs: Dict[str, Optional[xr.DataArray]] = {}
    for snap in snapshots:
        for var_name, var_cfg in variables.items():
            key = f"{snap.label}_{var_name}_obs"
            obs_filename = None
            if snap.obs_dir and var_cfg.obs_files:
                obs_filename = var_cfg.obs_files.get(snap.label)
            if obs_filename and snap.obs_dir:
                obs_path = os.path.join(snap.obs_dir, obs_filename)
                obs_da = xr.open_dataset(obs_path)[var_cfg.obs_nc_var]
                if "time" in obs_da.dims:
                    weights = obs_da.time.dt.days_in_month.astype(float)
                    obs_da = obs_da.weighted(weights).mean(dim="time")
                else:
                    obs_da = obs_da.squeeze()
                obs[key] = obs_da * var_cfg.obs_scale
            else:
                obs[key] = None

    # Prep fields and conditionally mask
    snap_small: Dict[str, xr.Dataset] = {}
    for snap in snapshots:
        ds = _prep_sim(snap_datasets[snap.label], snap.label, variables)
        has_all_obs = all(
            obs.get(f"{snap.label}_{v}_obs") is not None for v in variables
        )
        if has_all_obs:
            ds = _mask_sim(ds, snap.label, variables, obs)
        snap_small[snap.label] = ds

    # Merge all snapshots into a single dataset
    merged = snap_small[snapshots[0].label]
    for snap in snapshots[1:]:
        merged = merged.merge(snap_small[snap.label])

    return {"ppe_dataset_small": merged, **obs}


def compute_zrg_generic(
    sim_names: list,
    ppe_dataset_small: xr.Dataset,
    obs_dict: dict,
    control_file: str,
    regions_file: str,
    variables: dict,        # {var_name: VariableCfg}
    snapshots,              # List[SnapshotCfg]
) -> dict:
    """
    Compute area-weighted ZRG averages for all variables over all snapshots,
    for both simulations and observations.

    Snapshot obs values that are None are stored as NaN rows in zrg_obs.

    Returns dict with:
        zrg_ppedataset, zrg_obs,
        {var_name}_zrg_ppedataset  for each variable
    """
    ctrl = xr.open_dataset(control_file)
    area = ctrl.variables["area"][:]
    lat  = ctrl.variables["lat"][:]

    var_names = list(variables.keys())

    per_var_zrg: Dict[str, pd.DataFrame] = {}
    for var_name, var_cfg in variables.items():
        sim_field = var_cfg.sim_field
        snap_dfs = []
        for snap in snapshots:
            z_dict, r_dict, g_vals = {}, {}, []
            for run in sim_names:
                d = ppe_dataset_small.sel(run_label=run)[f"{snap.label}_{sim_field}"]
                z_dict[run] = _zonal_means(d, area, lat)
                r_dict[run] = _regional_means(d, area, regions_file)
                g_vals.append(_global_mean(d, area))
            snap_dfs.append(_zrg_df(z_dict, r_dict, g_vals, f"{snap.label}_global"))
        per_var_zrg[var_name] = pd.concat(snap_dfs, axis=1)

    zrg_ppedataset = pd.concat(list(per_var_zrg.values()), axis=1)

    per_var_zrg_obs: Dict[str, pd.DataFrame] = {}
    for var_name in var_names:
        snap_obs_dfs = []
        for snap in snapshots:
            obs_da = obs_dict.get(f"{snap.label}_{var_name}_obs")
            if obs_da is not None:
                snap_obs_dfs.append(_zrg_df(
                    {"obs": _zonal_means(obs_da, area, lat)},
                    {"obs": _regional_means(obs_da, area, regions_file)},
                    [_global_mean(obs_da, area)],
                    f"{snap.label}_global",
                ))
            else:
                # Placeholder NaN row so column layout matches simulations
                dummy = per_var_zrg[var_name].iloc[:1].copy() * np.nan
                dummy.index = ["obs"]
                n_per_snap = dummy.shape[1] // len(snapshots)
                dummy_snap = dummy.iloc[:, :n_per_snap].copy()
                dummy_snap.columns = per_var_zrg[var_name].columns[:n_per_snap]
                snap_obs_dfs.append(dummy_snap)
        per_var_zrg_obs[var_name] = pd.concat(snap_obs_dfs, axis=1)

    zrg_obs = pd.concat(list(per_var_zrg_obs.values()), axis=1)

    print(f"zrg_ppedataset: {zrg_ppedataset.shape}, zrg_obs: {zrg_obs.shape}")

    return {
        "zrg_ppedataset": zrg_ppedataset,
        "zrg_obs":        zrg_obs,
        **{f"{v}_zrg_ppedataset": per_var_zrg[v] for v in var_names},
    }


def drop_nan_zrg_features_generic(
    zrg_result: dict,
    var_names: list,
    n_zonal: int,
    n_regions: int,
    regions_list: list,
    snapshots,                              # List[SnapshotCfg]
    explicit_drop_zonal: Optional[List[float]] = None,
) -> Tuple[dict, int]:
    """
    Generic version of drop_nan_zrg_features for N snapshots.

    Drops feature columns (zonal band / region / global) that are all-NaN
    across all runs or obs for any variable, plus any explicitly listed zonal
    bands.  Dropping is symmetric: if a position is removed in one snapshot
    it is removed from all snapshots.
    """
    n_snaps    = len(snapshots)
    n_per_snap = n_zonal + n_regions + 1
    n_feat     = n_per_snap * n_snaps

    lat_bands     = np.linspace(-90, 90, n_zonal + 1)
    zonal_labels  = [f"zonal {(lat_bands[i] + lat_bands[i+1]) / 2:.0f}deg"
                     for i in range(n_zonal)]
    per_snap_labels = zonal_labels + list(regions_list) + ["global"]
    snap_labels  = [s.label for s in snapshots for _ in range(n_per_snap)]
    feat_labels  = per_snap_labels * n_snaps

    raw_nan: set = set()
    for var in var_names:
        df = zrg_result[f"{var}_zrg_ppedataset"]
        for i in range(n_feat):
            if df.iloc[:, i].isna().all():
                raw_nan.add(i)
    n_vars = len(var_names)
    for v_idx in range(n_vars):
        obs_block = zrg_result["zrg_obs"].iloc[:, v_idx * n_feat:(v_idx + 1) * n_feat]
        for i in range(n_feat):
            if obs_block.iloc[:, i].isna().all():
                raw_nan.add(i)

    if explicit_drop_zonal:
        centres = np.array([(lat_bands[i] + lat_bands[i + 1]) / 2 for i in range(n_zonal)])
        for target in explicit_drop_zonal:
            idx = int(np.argmin(np.abs(centres - target)))
            raw_nan.add(idx)
            print(f"  Explicitly dropping zonal band centre {centres[idx]:.0f}° "
                  f"(requested {target}°)")

    # Symmetric: drop a position across all snapshots if bad in any
    nan_cols: set = set()
    for i in raw_nan:
        pos = i % n_per_snap
        for s in range(n_snaps):
            nan_cols.add(s * n_per_snap + pos)

    if not nan_cols:
        print("  No all-NaN ZRG feature columns found.")
        return zrg_result, n_zonal

    sorted_nan = sorted(nan_cols)
    print(f"  Dropping {len(sorted_nan)} all-NaN ZRG feature column(s) (symmetric across snapshots):")
    for i in sorted_nan:
        nan_vars = [v for v in var_names
                    if zrg_result[f"{v}_zrg_ppedataset"].iloc[:, i].isna().all()]
        print(f"    {snap_labels[i]} {feat_labels[i]}"
              f"  (all-NaN for: {', '.join(nan_vars) if nan_vars else 'obs only'})")

    valid = [i for i in range(n_feat) if i not in nan_cols]

    updated = dict(zrg_result)
    for var in var_names:
        updated[f"{var}_zrg_ppedataset"] = zrg_result[f"{var}_zrg_ppedataset"].iloc[:, valid]

    all_drop = sorted({v_idx * n_feat + col for v_idx in range(n_vars) for col in sorted_nan})
    valid_global = [i for i in range(n_vars * n_feat) if i not in set(all_drop)]
    updated["zrg_ppedataset"] = zrg_result["zrg_ppedataset"].iloc[:, valid_global]
    updated["zrg_obs"]        = zrg_result["zrg_obs"].iloc[:, valid_global]

    dropped_positions = {col % n_per_snap for col in nan_cols}
    new_n_zonal = len([p for p in range(n_zonal) if p not in dropped_positions])
    print(f"  n_zonal updated: {n_zonal} → {new_n_zonal}")

    return updated, new_n_zonal


# ---------------------------------------------------------------------------
# Step 4: Stack ZRG into training arrays (all data, no fold split)
# ---------------------------------------------------------------------------
def stack_all_data(
    zrg_result: dict,
    ppe_params: pd.DataFrame,
    var_names: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack all runs into X (n_runs, n_params) and
    Y (n_runs, n_features, n_vars).
    """
    run_labels = ppe_params.index.to_list()
    dfs = []
    for v in var_names:
        df = zrg_result[f"{v}_zrg_ppedataset"].loc[run_labels].copy()
        df.columns = df.columns.astype(str)
        dfs.append(df)

    Y_train_ZRG = np.transpose(np.stack(dfs, axis=0), (1, 2, 0))
    X_train = ppe_params   # keep as DataFrame to preserve column names for constraint lookup

    print(f"X_train: {X_train.shape}, Y_train_ZRG: {Y_train_ZRG.shape}")
    return X_train, Y_train_ZRG


# ---------------------------------------------------------------------------
# K-fold splits (in-memory, same format as 04_kfold_and_stack.py)
# ---------------------------------------------------------------------------
def make_folds(
    zrg_result: dict,
    ppe_params: pd.DataFrame,
    var_names: list,
    n_folds: int = 5,
    random_state: int = 2,
) -> list:
    """
    Generate k-fold train/test splits from ZRG data.

    Returns a list of fold dicts ready for
    autotune_gp.evaluate.run_kfold_evaluation.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []

    for k, (train_idx, test_idx) in enumerate(kf.split(ppe_params)):
        X_train_df = ppe_params.iloc[train_idx]
        X_test_df  = ppe_params.iloc[test_idx]
        train_labels = X_train_df.index.to_list()
        test_labels  = X_test_df.index.to_list()

        per_var = {}
        for var in var_names:
            df = zrg_result[f"{var}_zrg_ppedataset"]
            train_df = df.loc[train_labels].copy()
            test_df  = df.loc[test_labels].copy()
            train_df.columns = train_df.columns.astype(str)
            test_df.columns  = test_df.columns.astype(str)
            per_var[var] = (train_df, test_df)

        Y_train_ZRG = np.transpose(
            np.stack([per_var[v][0] for v in var_names], axis=0), (1, 2, 0)
        )
        Y_test_ZRG = np.transpose(
            np.stack([per_var[v][1] for v in var_names], axis=0), (1, 2, 0)
        )

        folds.append({
            "k":                k,
            "train_run_labels": train_labels,
            "test_run_labels":  test_labels,
            "X_train":          X_train_df.to_numpy(),
            "X_test":           X_test_df.to_numpy(),
            "Y_train_ZRG":      Y_train_ZRG,
            "Y_test_ZRG":       Y_test_ZRG,
            "var_names":        var_names,
            **{f"{v}_train": per_var[v][0] for v in var_names},
            **{f"{v}_test":  per_var[v][1] for v in var_names},
        })

    return folds
