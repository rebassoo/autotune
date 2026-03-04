"""
Step 3: Compute zonal, regional, and global (ZRG) averages for both
simulation runs and observations.

Inputs:
  run_list.pkl  (from 01_build_run_list.py)
  sim_data.pkl  (from 02_load_and_mask.py)

Outputs: zrg_data.pkl
  {
    'zrg_ppedataset': pd.DataFrame,  # (n_runs, 200) — sims
    'zrg_obs':        pd.DataFrame,  # (1, 200)       — observations
    'PCP_zrg_ppedataset':  pd.DataFrame,
    'TLWP_zrg_ppedataset': pd.DataFrame,
    'OSR_zrg_ppedataset':  pd.DataFrame,
    'OLR_zrg_ppedataset':  pd.DataFrame,
  }

Column layout per variable (25 cols each, 4 vars → 100 cols per day → 200 total):
  DY1: 18 zonal bands | 6 regions | 1 global
  DY2: 18 zonal bands | 6 regions | 1 global
"""

import pickle
import numpy as np
import pandas as pd
import xarray as xr

INPUT_RUN_LIST = "run_list.pkl"
INPUT_SIM_DATA = "sim_data.pkl"
OUTPUT_PKL     = "zrg_data.pkl"

CONTROL_FILE = (
    "/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/"
    "SCREAM.2024-autocal-00.ne1024pg2/m0000/"
    "SCREAM.2024-autocal-00.ne1024pg2/run/"
    "output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24."
    "2020-01-26-00000.nc"
)
REGIONS_FILE = (
    "/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc"
)
REGIONS_LIST = [
    "poles",
    "extratropical_land",
    "extratropical_ocean",
    "tropical_land",
    "ascending_tropical_ocean",
    "descending_tropical_ocean",
]

# ---------------------------------------------------------------------------
# Spatial averaging functions
# ---------------------------------------------------------------------------

def zonal_means_native(data, area, lat, lon):
    """Area-weighted mean in 18 equal 10-degree latitude bands."""
    lat_bands = np.linspace(-90, 90, 19)
    zonal_means = {}
    for i in range(len(lat_bands) - 1):
        mask_zone = (lat > lat_bands[i]) & (lat < lat_bands[i + 1]).squeeze()
        data_zone = np.where(mask_zone > 0, data.squeeze(), np.nan)
        area_zone = np.where(mask_zone > 0, area.squeeze(), np.nan)
        zone_mean = np.nansum(data_zone * area_zone) / np.nansum(area_zone)
        zone_center = abs(lat_bands[i] - lat_bands[i + 1]) / 2 + lat_bands[i]
        zonal_means[zone_center] = zone_mean
    return zonal_means


def regional_means_native(data, area):
    """Area-weighted mean for each of the 6 pre-defined regions."""
    region_data = xr.open_dataset(REGIONS_FILE)
    region_means = {}
    for reg_name in REGIONS_LIST:
        mask_reg = region_data[reg_name].squeeze()
        data_reg = np.where(mask_reg > 0, data.squeeze(), np.nan)
        area_reg = np.where(mask_reg > 0, area.squeeze(), np.nan)
        reg_mean = np.nansum(data_reg * area_reg) / np.nansum(area_reg)
        region_means[reg_name] = reg_mean
    return region_means


def global_means_native(data, area):
    """Area-weighted global mean."""
    return np.nanmean(data * area) / np.nanmean(area)


# ---------------------------------------------------------------------------
# Load area / lat / lon from the control run
# ---------------------------------------------------------------------------
control = xr.open_dataset(CONTROL_FILE)
area = control.variables["area"][:]
lat  = control.variables["lat"][:]
lon  = control.variables["lon"][:]

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
with open(INPUT_RUN_LIST, "rb") as f:
    run_list = pickle.load(f)
sim_names = run_list["sim_names"]

with open(INPUT_SIM_DATA, "rb") as f:
    sim_data = pickle.load(f)

ppe_dataset_small = sim_data["ppe_dataset_small"]
DY1_PCP_obs  = sim_data["DY1_PCP_obs"]
DY1_TLWP_obs = sim_data["DY1_TLWP_obs"]
DY1_OSR_obs  = sim_data["DY1_OSR_obs"]
DY1_OLR_obs  = sim_data["DY1_OLR_obs"]
DY2_PCP_obs  = sim_data["DY2_PCP_obs"]
DY2_TLWP_obs = sim_data["DY2_TLWP_obs"]
DY2_OSR_obs  = sim_data["DY2_OSR_obs"]
DY2_OLR_obs  = sim_data["DY2_OLR_obs"]

# ---------------------------------------------------------------------------
# Compute ZRG averages for simulations
# ---------------------------------------------------------------------------
DY1_PCP_zonal  = {}; DY1_TLWP_zonal  = {}; DY1_OSR_zonal  = {}; DY1_OLR_zonal  = {}
DY2_PCP_zonal  = {}; DY2_TLWP_zonal  = {}; DY2_OSR_zonal  = {}; DY2_OLR_zonal  = {}

DY1_PCP_regional  = {}; DY1_TLWP_regional  = {}; DY1_OSR_regional  = {}; DY1_OLR_regional  = {}
DY2_PCP_regional  = {}; DY2_TLWP_regional  = {}; DY2_OSR_regional  = {}; DY2_OLR_regional  = {}

DY1_PCP_global  = []; DY1_TLWP_global  = []; DY1_OSR_global  = []; DY1_OLR_global  = []
DY2_PCP_global  = []; DY2_TLWP_global  = []; DY2_OSR_global  = []; DY2_OLR_global  = []

for run in sim_names:
    ds = ppe_dataset_small.sel(run_label=run)

    DY1_precip = ds["DY1_precip_total_surf_mass_flux"]
    DY1_tlwp   = ds["DY1_TotalLiqWaterPath"]
    DY1_sw     = ds["DY1_SW_flux_up_at_model_top"]
    DY1_lw     = ds["DY1_LW_flux_up_at_model_top"]

    DY2_precip = ds["DY2_precip_total_surf_mass_flux"]
    DY2_tlwp   = ds["DY2_TotalLiqWaterPath"]
    DY2_sw     = ds["DY2_SW_flux_up_at_model_top"]
    DY2_lw     = ds["DY2_LW_flux_up_at_model_top"]

    DY1_PCP_zonal[run]  = zonal_means_native(DY1_precip, area, lat, lon)
    DY1_TLWP_zonal[run] = zonal_means_native(DY1_tlwp,   area, lat, lon)
    DY1_OSR_zonal[run]  = zonal_means_native(DY1_sw,     area, lat, lon)
    DY1_OLR_zonal[run]  = zonal_means_native(DY1_lw,     area, lat, lon)

    DY2_PCP_zonal[run]  = zonal_means_native(DY2_precip, area, lat, lon)
    DY2_TLWP_zonal[run] = zonal_means_native(DY2_tlwp,   area, lat, lon)
    DY2_OSR_zonal[run]  = zonal_means_native(DY2_sw,     area, lat, lon)
    DY2_OLR_zonal[run]  = zonal_means_native(DY2_lw,     area, lat, lon)

    DY1_PCP_regional[run]  = regional_means_native(DY1_precip, area)
    DY1_TLWP_regional[run] = regional_means_native(DY1_tlwp,   area)
    DY1_OSR_regional[run]  = regional_means_native(DY1_sw,     area)
    DY1_OLR_regional[run]  = regional_means_native(DY1_lw,     area)

    DY2_PCP_regional[run]  = regional_means_native(DY2_precip, area)
    DY2_TLWP_regional[run] = regional_means_native(DY2_tlwp,   area)
    DY2_OSR_regional[run]  = regional_means_native(DY2_sw,     area)
    DY2_OLR_regional[run]  = regional_means_native(DY2_lw,     area)

    DY1_PCP_global.append(global_means_native(DY1_precip, area))
    DY1_TLWP_global.append(global_means_native(DY1_tlwp,  area))
    DY1_OSR_global.append(global_means_native(DY1_sw,     area))
    DY1_OLR_global.append(global_means_native(DY1_lw,     area))

    DY2_PCP_global.append(global_means_native(DY2_precip, area))
    DY2_TLWP_global.append(global_means_native(DY2_tlwp,  area))
    DY2_OSR_global.append(global_means_native(DY2_sw,     area))
    DY2_OLR_global.append(global_means_native(DY2_lw,     area))


# ---------------------------------------------------------------------------
# Build per-variable ZRG DataFrames for simulations
# ---------------------------------------------------------------------------
def _build_zrg_df(dy1_zonal, dy1_regional, dy1_global, dy1_tag,
                  dy2_zonal, dy2_regional, dy2_global, dy2_tag):
    dy1_z = pd.DataFrame.from_dict(dy1_zonal,    orient="index")
    dy1_r = pd.DataFrame.from_dict(dy1_regional, orient="index")
    dy1_df = pd.concat([dy1_z, dy1_r], axis=1)
    dy1_df[dy1_tag + "_global"] = dy1_global

    dy2_z = pd.DataFrame.from_dict(dy2_zonal,    orient="index")
    dy2_r = pd.DataFrame.from_dict(dy2_regional, orient="index")
    dy2_df = pd.concat([dy2_z, dy2_r], axis=1)
    dy2_df[dy2_tag + "_global"] = dy2_global

    return dy1_df, dy2_df, pd.concat([dy1_df, dy2_df], axis=1)


DY1_PCP_zrg,  DY2_PCP_zrg,  PCP_zrg_ppedataset  = _build_zrg_df(
    DY1_PCP_zonal, DY1_PCP_regional, DY1_PCP_global, "DY1",
    DY2_PCP_zonal, DY2_PCP_regional, DY2_PCP_global, "DY2",
)
DY1_TLWP_zrg, DY2_TLWP_zrg, TLWP_zrg_ppedataset = _build_zrg_df(
    DY1_TLWP_zonal, DY1_TLWP_regional, DY1_TLWP_global, "DY1",
    DY2_TLWP_zonal, DY2_TLWP_regional, DY2_TLWP_global, "DY2",
)
DY1_OSR_zrg,  DY2_OSR_zrg,  OSR_zrg_ppedataset  = _build_zrg_df(
    DY1_OSR_zonal, DY1_OSR_regional, DY1_OSR_global, "DY1",
    DY2_OSR_zonal, DY2_OSR_regional, DY2_OSR_global, "DY2",
)
DY1_OLR_zrg,  DY2_OLR_zrg,  OLR_zrg_ppedataset  = _build_zrg_df(
    DY1_OLR_zonal, DY1_OLR_regional, DY1_OLR_global, "DY1",
    DY2_OLR_zonal, DY2_OLR_regional, DY2_OLR_global, "DY2",
)

# Ordering matters: PCP, TLWP, OSR, OLR
zrg_ppedataset = pd.concat(
    [PCP_zrg_ppedataset, TLWP_zrg_ppedataset, OSR_zrg_ppedataset, OLR_zrg_ppedataset],
    axis=1,
)

# ---------------------------------------------------------------------------
# Compute ZRG averages for observations
# ---------------------------------------------------------------------------
def _obs_zrg(pcp_obs, tlwp_obs, osr_obs, olr_obs, day_tag):
    pcp_z  = {"obs": zonal_means_native(pcp_obs,  area, lat, lon)}
    tlwp_z = {"obs": zonal_means_native(tlwp_obs, area, lat, lon)}
    osr_z  = {"obs": zonal_means_native(osr_obs,  area, lat, lon)}
    olr_z  = {"obs": zonal_means_native(olr_obs,  area, lat, lon)}

    pcp_r  = {"obs": regional_means_native(pcp_obs,  area)}
    tlwp_r = {"obs": regional_means_native(tlwp_obs, area)}
    osr_r  = {"obs": regional_means_native(osr_obs,  area)}
    olr_r  = {"obs": regional_means_native(olr_obs,  area)}

    pcp_g  = [global_means_native(pcp_obs,  area)]
    tlwp_g = [global_means_native(tlwp_obs, area)]
    osr_g  = [global_means_native(osr_obs,  area)]
    olr_g  = [global_means_native(olr_obs,  area)]

    def _df(z, r, g):
        df = pd.concat(
            [pd.DataFrame.from_dict(z, orient="index"),
             pd.DataFrame.from_dict(r, orient="index")],
            axis=1,
        )
        df[day_tag + "_global"] = g
        return df

    return _df(pcp_z, pcp_r, pcp_g), _df(tlwp_z, tlwp_r, tlwp_g), \
           _df(osr_z, osr_r, osr_g), _df(olr_z, olr_r, olr_g)


DY1_PCP_zrg_obs,  DY1_TLWP_zrg_obs,  DY1_OSR_zrg_obs,  DY1_OLR_zrg_obs  = \
    _obs_zrg(DY1_PCP_obs, DY1_TLWP_obs, DY1_OSR_obs, DY1_OLR_obs, "DY1")
DY2_PCP_zrg_obs,  DY2_TLWP_zrg_obs,  DY2_OSR_zrg_obs,  DY2_OLR_zrg_obs  = \
    _obs_zrg(DY2_PCP_obs, DY2_TLWP_obs, DY2_OSR_obs, DY2_OLR_obs, "DY2")

PCP_zrg_obs  = pd.concat([DY1_PCP_zrg_obs,  DY2_PCP_zrg_obs],  axis=1)
TLWP_zrg_obs = pd.concat([DY1_TLWP_zrg_obs, DY2_TLWP_zrg_obs], axis=1)
OSR_zrg_obs  = pd.concat([DY1_OSR_zrg_obs,  DY2_OSR_zrg_obs],  axis=1)
OLR_zrg_obs  = pd.concat([DY1_OLR_zrg_obs,  DY2_OLR_zrg_obs],  axis=1)

# Ordering must match zrg_ppedataset: PCP, TLWP, OSR, OLR
zrg_obs = pd.concat([PCP_zrg_obs, TLWP_zrg_obs, OSR_zrg_obs, OLR_zrg_obs], axis=1)

print("zrg_ppedataset shape:", zrg_ppedataset.shape)  # (n_runs, 200)
print("zrg_obs shape:       ", zrg_obs.shape)          # (1, 200)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(
        {
            "zrg_ppedataset":      zrg_ppedataset,
            "zrg_obs":             zrg_obs,
            "PCP_zrg_ppedataset":  PCP_zrg_ppedataset,
            "TLWP_zrg_ppedataset": TLWP_zrg_ppedataset,
            "OSR_zrg_ppedataset":  OSR_zrg_ppedataset,
            "OLR_zrg_ppedataset":  OLR_zrg_ppedataset,
        },
        f,
    )
print(f"Saved {OUTPUT_PKL}")
