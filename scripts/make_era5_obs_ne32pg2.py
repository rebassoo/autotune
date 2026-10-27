"""
Annual-mean (Aug 2019 - Jul 2020) ERA5 observation files on ne32pg2 for the
five ERA5-based observables, one file per variable, single time step, in the
units of the matching SCREAM output field:

  TAS_LAND  t2m (K), NaN where the ERA5 land-sea mask is < 0.5   vs  T_2m
  SLP       msl (Pa)                                              vs  SeaLevelPressure
  U200      u at 200 hPa (m/s)                                    vs  U interpolated to 200 hPa
  U850      u at 850 hPa (m/s), NaN where 850 hPa is near or      vs  U interpolated to 850 hPa
            below ground in any model resolution (see below)
  Z500      z at 500 hPa / g0 (m)                                 vs  z_mid interpolated to 500 hPa

Net TOA needs no new file: CERES_EBAF_ne32pg2_201908_202007.nc already holds
toa_net_all_mon for the same 12 months.

The preprocessing masks the simulations wherever the obs are NaN, so the land
mask and the below-ground mask carry over to every member automatically.

Below-ground mask for U850: ERA5 extrapolates pressure levels under the
terrain, the model interpolation returns NaN there. To compare like with
like, cells whose annual-mean surface pressure is below PS_MIN in any of the
three ensembles' first members (ne32, ne128, ne256 -- topography differs by
resolution) are NaN in the obs.

Months are weighted by their length (Feb 2020 has 29 days).

    python scripts/make_era5_obs_ne32pg2.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import xarray as xr

UTIL = Path("/global/cfs/cdirs/e3sm/beydoun/autotune_utils")
OUT = Path("/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning")
G0 = 9.80665
PS_MIN = 87000.0          # Pa: 850 hPa plus a 20 hPa margin for month-to-month ps changes
WINDOW = ("2019-08-01", "2020-07-31")
PS_MEMBERS = [
    "/pscratch/sd/b/beydoun/e3sm_scratch/pm-gpu/ne32_ppe_prod_v2/"
    "PPEensemble_1node_full1024.ne32pg2_ne32pg2.F2010-SCREAMv1.20260608.m001/run",
    "/global/cfs/cdirs/e3sm/beydoun/ne128_ppe_prod/"
    "PPEensemble_64node_full256.ne128pg2_ne128pg2.F2010-SCREAMv1.20260608.m000/run",
    "/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/ne256_n85_members/"
    "PPEensemble_64node_full256.ne256pg2_ne256pg2.F2010-SCREAMv1.20260608.m001/run",
]


def annual(da):
    """Days-weighted mean over the 12 months of the window."""
    da = da.sel(time=slice(*WINDOW))
    if da.sizes["time"] != 12:
        raise SystemExit(f"{da.name}: expected 12 months in {WINDOW}, got {da.sizes['time']}")
    return da.weighted(da.time.dt.days_in_month.astype(float)).mean("time")


def min_surface_pressure():
    """Smallest annual-mean ps across the three resolutions' first members."""
    out = None
    for run in PS_MEMBERS:
        files = sorted(glob.glob(f"{run}/1ma_ne32pg2.AVERAGE.nmonths_x1.*.nc"))[:12]
        if len(files) != 12:
            raise SystemExit(f"{run}: need 12 monthly files, found {len(files)}")
        ps = xr.open_mfdataset(files, combine="by_coords")["ps"]
        ps = ps.weighted(ps.time.dt.days_in_month.astype(float)).mean("time").values
        print(f"  ps from {Path(run).parent.name[-40:]}: min {ps.min() / 100:.0f} hPa")
        out = ps if out is None else np.minimum(out, ps)
    return out


def write(name, da, units, long_name, template):
    ds = xr.Dataset({name: (("time", "ncol"), da.values[None, :].astype("float64"))},
                    coords={"time": [np.datetime64("2020-01-31")]})
    ds[name].attrs.update(units=units, long_name=long_name)
    ds = ds.assign_coords({c: ("ncol", template[c].values) for c in ("lat", "lon")})
    for v in ("area", "lat_vertices", "lon_vertices"):
        if v in template:
            ds[v] = (template[v].dims, template[v].values)
    ds.attrs["description"] = ("ERA5 annual mean Aug 2019 - Jul 2020 (days-weighted), ne32pg2; "
                               "made by scripts/make_era5_obs_ne32pg2.py")
    path = OUT / f"{name}_ERA5_ne32pg2_201908_202007.nc"
    ds.to_netcdf(path)
    v = da.values
    print(f"  wrote {path.name}: valid {np.isfinite(v).mean():.1%} of columns, "
          f"range {np.nanmin(v):.4g} .. {np.nanmax(v):.4g} {units}")


def main():
    sfc = xr.open_dataset(UTIL / "ERA5_sfc_201908_202008_ne32pg2.nc")
    u = xr.open_dataset(UTIL / "ERA5_u_201908_202008_ne32pg2.nc")
    pl = xr.open_dataset(UTIL / "ERA5_plev_201908_202007_ne32pg2.nc")

    lsm = sfc["lsm"]
    land = (annual(lsm) if "time" in lsm.dims else lsm) >= 0.5
    write("TAS_LAND", annual(sfc["t2m"]).where(land), "K",
          "2 m air temperature over land (ERA5 lsm >= 0.5)", sfc)
    write("SLP", annual(sfc["msl"]), "Pa", "mean sea-level pressure", sfc)
    write("U200", annual(u["u"].sel(pressure_level=200.0)), "m/s", "zonal wind at 200 hPa", sfc)

    print("Below-ground mask for U850:")
    above = min_surface_pressure() >= PS_MIN
    print(f"  {(~above).mean():.1%} of columns masked (annual-mean ps < {PS_MIN / 100:.0f} hPa "
          "in at least one resolution)")
    write("U850", annual(u["u"].sel(pressure_level=850.0)).where(above), "m/s",
          f"zonal wind at 850 hPa; NaN where model ps < {PS_MIN / 100:.0f} hPa", sfc)
    write("Z500", annual(pl["z"].sel(pressure_level=500.0)) / G0, "m",
          "geopotential height at 500 hPa (z / 9.80665)", sfc)


if __name__ == "__main__":
    main()
