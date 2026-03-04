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

import os
from typing import Dict, List, Tuple

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

    # --- Extract target variables from simulations ---
    # Determine which raw sim fields are needed
    def _raw_fields_needed(variables):
        needed = set()
        for var_cfg in variables.values():
            if var_cfg.sim_components:
                needed.update(var_cfg.sim_components)
            else:
                needed.add(var_cfg.sim_field)
        return needed

    def _prep_sim(ds, prefix, variables):
        raw_needed = _raw_fields_needed(variables)
        # Drop everything not needed (keeps the dataset small)
        to_drop = [v for v in ds.data_vars if v not in raw_needed]
        ds = ds.drop_vars(to_drop, errors="ignore")
        ds = ds.drop_vars("p_levs", errors="ignore")

        # Compute derived fields (e.g. TotalLiqWaterPath = sum of components)
        direct_fields = {vc.sim_field for vc in variables.values()
                         if not vc.sim_components}
        for var_cfg in variables.values():
            if var_cfg.sim_components:
                ds[var_cfg.sim_field] = sum(
                    ds[c] for c in var_cfg.sim_components
                )
                # Drop components that are not also direct sim_fields
                for comp in var_cfg.sim_components:
                    if comp not in direct_fields:
                        ds = ds.drop_vars(comp, errors="ignore")

        return ds.rename({v: f"{prefix}_{v}" for v in ds.data_vars})

    DY1_small = _prep_sim(DY1_ds, "DY1", variables)
    DY2_small = _prep_sim(DY2_ds, "DY2", variables)

    # --- Mask simulations to cells where obs is available ---
    def _mask(ds, prefix, variables, obs):
        ds = ds.copy(deep=True)
        for var_name, var_cfg in variables.items():
            field_key = f"{prefix}_{var_cfg.sim_field}"
            obs_key   = f"{prefix}_{var_name}_obs"
            ds[field_key] = ds[field_key].where(~np.isnan(obs[obs_key]))
        return ds

    DY1_small = _mask(DY1_small, "DY1", variables, obs)
    DY2_small = _mask(DY2_small, "DY2", variables, obs)

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
def _zonal_means(data, area, lat):
    lat_bands = np.linspace(-90, 90, 19)
    result = {}
    for i in range(len(lat_bands) - 1):
        mask = (lat >= lat_bands[i]) & (lat < lat_bands[i + 1]).squeeze()
        d = np.where(mask > 0, data.squeeze(), np.nan)
        a = np.where(mask > 0, area.squeeze(), np.nan)
        center = abs(lat_bands[i] - lat_bands[i + 1]) / 2 + lat_bands[i]
        result[center] = np.nansum(d * a) / np.nansum(a)
    return result


def _regional_means(data, area, regions_file):
    region_data = xr.open_dataset(regions_file)
    result = {}
    for reg in REGIONS_LIST:
        mask = region_data[reg].squeeze()
        d = np.where(mask > 0, data.squeeze(), np.nan)
        a = np.where(mask > 0, area.squeeze(), np.nan)
        result[reg] = np.nansum(d * a) / np.nansum(a)
    return result


def _global_mean(data, area):
    return np.nanmean(data * area) / np.nanmean(area)


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
