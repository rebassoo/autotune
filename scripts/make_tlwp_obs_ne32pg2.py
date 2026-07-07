"""
Remap TLWP (Total Liquid Water Path) observations from 1-degree lat-lon to
ne32pg2 and save a period-matched annual-mean file.

Input:  maclwp_totallwpave_2019_v2.nc4  (12 monthly means, Jan-Dec 2019)
        maclwp_totallwpave_2020_v2.nc4  (12 monthly means, Jan-Dec 2020)
Output: TLWP_MACLWP_ne32pg2_201908_202007.nc
        Variable: totallwp  (g/m²), single time step (annual mean Aug 2019-Jul 2020)

Period Aug 2019–Jul 2020 matches the existing CERES and IMERG obs files.

The sim field is LiqWaterPath + RainWaterPath (both kg/m²), so use
obs_scale: 0.001 in the config to convert g/m² → kg/m².

Usage:
    python scripts/make_tlwp_obs_ne32pg2.py
"""
from __future__ import annotations

import numpy as np
import netCDF4 as nc
from scipy.sparse import csr_matrix

# ---- paths ----------------------------------------------------------------
TLWP_DIR   = "/global/homes/r/rebassoo/work/2026_06_29_TLWP"
MAP_FILE   = "/global/cfs/cdirs/e3sm/beydoun/autotune_utils/map_180x360_to_ne32pg2.nc"
OUT_FILE   = "/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/TLWP_MACLWP_ne32pg2_201908_202007.nc"

# Aug-Dec from 2019 file (0-based month indices 7..11), Jan-Jul from 2020 (0..6)
MONTHS_2019 = list(range(7, 12))   # Aug, Sep, Oct, Nov, Dec
MONTHS_2020 = list(range(0, 7))    # Jan, Feb, Mar, Apr, May, Jun, Jul
DAYS_IN_MONTH = [31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 31]
# Aug=31, Sep=30, Oct=31, Nov=30, Dec=31, Jan=31, Feb=28, Mar=31, Apr=30, May=31, Jun=30, Jul=31
DAYS = [31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31]

# ---- load remap weights ---------------------------------------------------
print("Loading remap weights ...")
with nc.Dataset(MAP_FILE) as wf:
    row = wf.variables["row"][:] - 1   # 1-indexed → 0-indexed
    col = wf.variables["col"][:] - 1
    S   = wf.variables["S"][:]
    n_b = len(wf.dimensions["n_b"])    # 24576 (ne32pg2)
    n_a = len(wf.dimensions["n_a"])    # 64800 (180×360)

W = csr_matrix((S, (row, col)), shape=(n_b, n_a))
print(f"  Weights matrix: {W.shape}, {W.nnz} non-zeros")

# ---- load and subset TLWP data -------------------------------------------
print("Loading TLWP data ...")
with nc.Dataset(f"{TLWP_DIR}/maclwp_totallwpave_2019_v2.nc4") as ds:
    data_2019 = np.ma.filled(ds.variables["totallwp"][MONTHS_2019, :, :].astype(float), np.nan)
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]

with nc.Dataset(f"{TLWP_DIR}/maclwp_totallwpave_2020_v2.nc4") as ds:
    data_2020 = np.ma.filled(ds.variables["totallwp"][MONTHS_2020, :, :].astype(float), np.nan)

# Concatenate: Aug 2019 ... Jul 2020  (12 months)
data = np.concatenate([data_2019, data_2020], axis=0)   # (12, 180, 360)
days = np.array(DAYS, dtype=float)
print(f"  Loaded {data.shape[0]} months, shape {data.shape}")

# ---- weighted annual mean (before remapping, to save memory) -------------
print("Computing weighted annual mean ...")
weights = days[:, None, None]

w_sum  = np.where(np.isfinite(data), weights, 0.0).sum(axis=0)
annual = np.nansum(data * weights, axis=0) / np.where(w_sum > 0, w_sum, np.nan)
# annual shape: (180, 360)
print(f"  Annual mean range: {np.nanmin(annual):.2f} – {np.nanmax(annual):.2f} g/m²")

# ---- remap to ne32pg2 ----------------------------------------------------
print("Remapping to ne32pg2 ...")
src_flat = annual.ravel()                  # (64800,)  lat varies slowest
valid    = np.isfinite(src_flat)
src_masked = np.where(valid, src_flat, 0.0)

dst = W.dot(src_masked)                    # (24576,)
# Correct for missing source cells: re-weight by sum of weights hitting valid cells
w_valid = W.dot(valid.astype(float))
dst = np.where(w_valid > 0.5, dst / w_valid, np.nan)

print(f"  Remapped range: {np.nanmin(dst):.2f} – {np.nanmax(dst):.2f} g/m²")
print(f"  NaN cells: {np.isnan(dst).sum()} / {len(dst)}")

# ---- write output --------------------------------------------------------
print(f"Writing {OUT_FILE} ...")
with nc.Dataset(OUT_FILE, "w") as out:
    out.createDimension("ncol", n_b)
    out.createDimension("time", 1)

    t = out.createVariable("time", "f4", ("time",))
    t.units    = "days since 2019-08-01"
    t.long_name = "time"
    t[:] = [182.0]   # midpoint of Aug 2019–Jul 2020

    v = out.createVariable("totallwp", "f4", ("time", "ncol"),
                           fill_value=1e20)
    v.units     = "g/m^2"
    v.long_name = "Annual mean Total Cloud+Rain Liquid Water Path (Aug 2019–Jul 2020)"
    v.source    = "MACLWP v2, remapped to ne32pg2 via map_180x360_to_ne32pg2.nc"
    v[0, :]     = dst

    # Copy lat/lon from weights file for reference
    with nc.Dataset(MAP_FILE) as wf:
        lat_b = wf.variables["yc_b"][:]
        lon_b = wf.variables["xc_b"][:]
    la = out.createVariable("lat", "f4", ("ncol",))
    la.units = "degrees_north"; la[:] = lat_b
    lo = out.createVariable("lon", "f4", ("ncol",))
    lo.units = "degrees_east";  lo[:] = lon_b

    out.description = ("MACLWP Total LWP obs remapped to ne32pg2, "
                       "period Aug 2019–Jul 2020")

print(f"Done → {OUT_FILE}")
print()
print("Config entry:")
print("  TLWP:")
print("    sim_field: LiqWaterPath")
print("    sim_components: [LiqWaterPath, RainWaterPath]")
print(f"    obs_files:")
print(f"      ANN: {OUT_FILE}")
print("    obs_nc_var: totallwp")
print("    obs_scale: 0.001   # g/m² → kg/m²")
