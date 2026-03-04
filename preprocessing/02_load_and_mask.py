"""
Step 2: Load simulation netCDFs and observation files, extract target
variables, mask simulations to the grid cells where observations exist.

Inputs:  run_list.pkl  (from 01_build_run_list.py)

Outputs: sim_data.pkl
  {
    'ppe_dataset_small': xr.Dataset,  # DY1+DY2 merged, obs-masked
    'DY1_PCP_obs':  xr.DataArray,
    'DY1_TLWP_obs': xr.DataArray,
    'DY1_OSR_obs':  xr.DataArray,
    'DY1_OLR_obs':  xr.DataArray,
    'DY2_PCP_obs':  xr.DataArray,
    'DY2_TLWP_obs': xr.DataArray,
    'DY2_OSR_obs':  xr.DataArray,
    'DY2_OLR_obs':  xr.DataArray,
  }

Notes:
- TotalLiqWaterPath = LiqWaterPath + RainWaterPath (computed here)
- Simulation variables are renamed with DY1_/DY2_ prefixes
- Masking uses obs NaN positions (where obs is NaN, sim is set to NaN)
- The reference script initialised DY2_ppe_dataset_mask from DY1; this
  script correctly initialises it from DY2_ppe_dataset_small.
"""

import pickle
import numpy as np
import xarray as xr

INPUT_PKL = "run_list.pkl"
OUTPUT_PKL = "sim_data.pkl"

# Variables to drop from the raw simulation output
VARS_TO_DROP = [
    "SW_flux_dn", "SW_flux_dn_at_model_bot", "SW_flux_up",
    "SW_flux_up_at_model_bot", "SW_flux_dn_at_model_top", "T_2m",
    "T_mid", "precip_ice_surf_mass_flux", "precip_liq_surf_mass_flux",
    "ps", "qc", "qi", "qm", "qr", "qv", "qv_2m", "LW_flux_up",
    "LW_flux_up_at_model_bot", "IceWaterPath", "LW_flux_dn",
    "LW_flux_dn_at_model_bot", "time_bnds", "LongwaveCloudForcing",
    "MeridionalVapFlux", "ShortwaveCloudForcing", "U", "V",
    "VapWaterPath", "ZonalVapFlux", "bm", "eddy_diff_mom",
    "eff_radius_qc_at_cldtop", "eff_radius_qi_at_cldtop",
    "homme_T_mid_tend", "homme_qv_tend", "horiz_winds_at_model_bot",
    "nc", "ni", "nr", "omega", "p3_T_mid_tend", "p3_qv_tend",
    "rrtmgp_T_mid_tend", "sgs_buoy_flux", "shoc_T_mid_tend",
    "shoc_qv_tend", "surf_evap", "surf_mom_flux", "surf_radiative_T",
    "surf_sens_flux", "surface_upward_latent_heat_flux",
    "avg_count_ncol", "avg_count_ncol_lev", "avg_count_ncol_dim",
    "area", "lat", "lon",
]

# ---------------------------------------------------------------------------
# Observation file paths
# ---------------------------------------------------------------------------
DY1_OBS_DIR = "/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/obs/"
DY1_PCP_OBS_FILE  = DY1_OBS_DIR + "IMERG.precip_total_surf_mass_flux.daily_AVERAGE.ne30pg2.20160807_mahf708.nc"
DY1_LW_OBS_FILE   = DY1_OBS_DIR + "CERES.LW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807_mahf708.nc"
DY1_SW_UP_OBS_FILE = DY1_OBS_DIR + "CERES.SW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807_mahf708.nc"
DY1_LWP_FILE      = DY1_OBS_DIR + "mac.clwp-tlwp-wvp.20160807.ne30pg2.nc"

DY2_OBS_DIR = "/global/cfs/projectdirs/e3smdata/simulations/SCREAM.2024-autocal-00.ne1024pg2/obs/"
DY2_PCP_OBS_FILE  = DY2_OBS_DIR + "IMERG.precip_total_surf_mass_flux.AVERAGE.ne30pg2.20200126.nc"
DY2_LW_OBS_FILE   = DY2_OBS_DIR + "CERES.LW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc"
DY2_SW_UP_OBS_FILE = DY2_OBS_DIR + "CERES.SW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc"
DY2_LWP_FILE      = DY2_OBS_DIR + "mac.clwp-tlwp-wvp.20200126.ne30pg2.nc"

# ---------------------------------------------------------------------------
# Load run list
# ---------------------------------------------------------------------------
with open(INPUT_PKL, "rb") as f:
    run_list = pickle.load(f)

sim_names         = run_list["sim_names"]
DY1_sim_names     = run_list["DY1_sim_names"]
DY2_sim_names     = run_list["DY2_sim_names"]
DY1_filename_list = run_list["DY1_filename_list"]
DY2_filename_list = run_list["DY2_filename_list"]

# ---------------------------------------------------------------------------
# Load simulation datasets
# ---------------------------------------------------------------------------
DY1_concat = xr.open_mfdataset(
    DY1_filename_list, concat_dim="run_label", combine="nested"
)
DY1_ppe_dataset = (
    DY1_concat
    .assign_coords(run_label=("run_label", DY1_sim_names))
    .squeeze("time")
)

DY2_concat = xr.open_mfdataset(
    DY2_filename_list, concat_dim="run_label", combine="nested"
)
DY2_ppe_dataset = (
    DY2_concat
    .assign_coords(run_label=("run_label", DY2_sim_names))
    .squeeze("time")
)

# ---------------------------------------------------------------------------
# Load observations
# ---------------------------------------------------------------------------
DY1_PCP_obs  = xr.open_dataset(DY1_PCP_OBS_FILE).variables["precip_total_surf_mass_flux"].squeeze("time")
DY1_TLWP_obs = (xr.open_dataset(DY1_LWP_FILE).variables["tlwp"] * 1e-3).squeeze("time")
DY1_OSR_obs  = xr.open_dataset(DY1_SW_UP_OBS_FILE).variables["SW_flux_up_at_model_top"].squeeze("time")
DY1_OLR_obs  = xr.open_dataset(DY1_LW_OBS_FILE).variables["LW_flux_up_at_model_top"].squeeze("time")

DY2_PCP_obs  = xr.open_dataset(DY2_PCP_OBS_FILE).variables["precip_total_surf_mass_flux"].squeeze("time")
DY2_TLWP_obs = (xr.open_dataset(DY2_LWP_FILE).variables["tlwp"] * 1e-3).squeeze("time")
DY2_OSR_obs  = xr.open_dataset(DY2_SW_UP_OBS_FILE).variables["SW_flux_up_at_model_top"].squeeze("time")
DY2_OLR_obs  = xr.open_dataset(DY2_LW_OBS_FILE).variables["LW_flux_up_at_model_top"].squeeze("time")

# ---------------------------------------------------------------------------
# Extract target variables
# ---------------------------------------------------------------------------
DY1_ppe_dataset_small = DY1_ppe_dataset.drop_vars(VARS_TO_DROP, errors="ignore")
DY1_ppe_dataset_small["TotalLiqWaterPath"] = (
    DY1_ppe_dataset_small.LiqWaterPath + DY1_ppe_dataset_small.RainWaterPath
)
DY1_ppe_dataset_small = DY1_ppe_dataset_small.drop_vars("p_levs", errors="ignore")
DY1_ppe_dataset_small = DY1_ppe_dataset_small.rename(
    {var: f"DY1_{var}" for var in DY1_ppe_dataset_small.data_vars}
)

DY2_ppe_dataset_small = DY2_ppe_dataset.drop_vars(VARS_TO_DROP, errors="ignore")
DY2_ppe_dataset_small["TotalLiqWaterPath"] = (
    DY2_ppe_dataset_small.LiqWaterPath + DY2_ppe_dataset_small.RainWaterPath
)
DY2_ppe_dataset_small = DY2_ppe_dataset_small.rename(
    {var: f"DY2_{var}" for var in DY2_ppe_dataset_small.data_vars}
)

# ---------------------------------------------------------------------------
# Mask simulations to cells where observations are available
# ---------------------------------------------------------------------------
DY1_ppe_dataset_mask = DY1_ppe_dataset_small.copy(deep=True)
DY2_ppe_dataset_mask = DY2_ppe_dataset_small.copy(deep=True)

DY1_ppe_dataset_mask["DY1_precip_total_surf_mass_flux"] = (
    DY1_ppe_dataset_small.DY1_precip_total_surf_mass_flux
    .where(~np.isnan(DY1_PCP_obs))
)
DY1_ppe_dataset_mask["DY1_TotalLiqWaterPath"] = (
    DY1_ppe_dataset_small.DY1_TotalLiqWaterPath
    .where(~np.isnan(DY1_TLWP_obs))
)
DY1_ppe_dataset_mask["DY1_SW_flux_up_at_model_top"] = (
    DY1_ppe_dataset_small.DY1_SW_flux_up_at_model_top
    .where(~np.isnan(DY1_OSR_obs))
)
DY1_ppe_dataset_mask["DY1_LW_flux_up_at_model_top"] = (
    DY1_ppe_dataset_small.DY1_LW_flux_up_at_model_top
    .where(~np.isnan(DY1_OLR_obs))
)

DY2_ppe_dataset_mask["DY2_precip_total_surf_mass_flux"] = (
    DY2_ppe_dataset_small.DY2_precip_total_surf_mass_flux
    .where(~np.isnan(DY2_PCP_obs))
)
DY2_ppe_dataset_mask["DY2_TotalLiqWaterPath"] = (
    DY2_ppe_dataset_small.DY2_TotalLiqWaterPath
    .where(~np.isnan(DY2_TLWP_obs))
)
DY2_ppe_dataset_mask["DY2_SW_flux_up_at_model_top"] = (
    DY2_ppe_dataset_small.DY2_SW_flux_up_at_model_top
    .where(~np.isnan(DY2_OSR_obs))
)
DY2_ppe_dataset_mask["DY2_LW_flux_up_at_model_top"] = (
    DY2_ppe_dataset_small.DY2_LW_flux_up_at_model_top
    .where(~np.isnan(DY2_OLR_obs))
)

DY1_ppe_dataset_small = DY1_ppe_dataset_mask
DY2_ppe_dataset_small = DY2_ppe_dataset_mask

# ---------------------------------------------------------------------------
# Merge DY1 and DY2 into a single dataset, filtered to intersection runs
# ---------------------------------------------------------------------------
ppe_dataset_small = (
    DY1_ppe_dataset_small.sel(run_label=sim_names)
    .combine_first(DY2_ppe_dataset_small.sel(run_label=sim_names))
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(
        {
            "ppe_dataset_small": ppe_dataset_small,
            "DY1_PCP_obs":  DY1_PCP_obs,
            "DY1_TLWP_obs": DY1_TLWP_obs,
            "DY1_OSR_obs":  DY1_OSR_obs,
            "DY1_OLR_obs":  DY1_OLR_obs,
            "DY2_PCP_obs":  DY2_PCP_obs,
            "DY2_TLWP_obs": DY2_TLWP_obs,
            "DY2_OSR_obs":  DY2_OSR_obs,
            "DY2_OLR_obs":  DY2_OLR_obs,
        },
        f,
    )
print(f"Saved {OUTPUT_PKL}")
print("ppe_dataset_small:", ppe_dataset_small)
