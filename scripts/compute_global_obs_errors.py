"""
Numbers behind the observation-based global-mean errors used in model-error
calibration experiment E2 (docs/global_obs_error_estimates.md).

Prints, for the Aug 2019 - Jul 2020 window:
  PCP  : IMERG global mean on its valid footprint, GPCP v3.2 on the same
         footprint, their annual and monthly differences;
  TLWP : MAC-LWP global mean and the global-mean error implied by its 1-sigma
         field under the fully-correlated and fully-random limits;
  and the PPE spread of each global feature (to express the errors as a
  fraction of it).
OSR/OLR come from the literature (Loeb et al. 2018) and are not computed here.

    python scripts/compute_global_obs_errors.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# scalers.pkl holds autotune_gp objects
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

IMERG = "/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/PRECT_IMERG_ne32pg2_201908_202007.nc"
GPCP = ("/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/GPCP_v3.2/"
        "sat_gauge_precip_198301_202112.nc")
MAC = "/global/cfs/cdirs/e3sm/beydoun/autotune_utils/MAC_2020_ne32pg2.nc"
TLWP_OBS = "/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/TLWP_MACLWP_ne32pg2_201908_202007.nc"
SCALERS = ("/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/"
           "preprocess_output_ne128_prod_v2/scalers.pkl")


def amean(x, a, ok):
    return float(np.sum(x[ok] * a[ok]) / np.sum(a[ok]))


def precip():
    im = xr.open_dataset(IMERG)
    a = im.area.values
    pr = im.PRECT.values * 86400e3                      # m/s -> mm/day, (day, ncol)
    clat = im.lat_vertices.values.mean(1)
    clon = np.mod(im.lon_vertices.values.mean(1), 360)
    ok = np.isfinite(pr).all(0)
    print(f"IMERG: {pr.shape[0]} days, valid columns {ok.mean():.1%} "
          f"(missing |lat| {np.abs(clat[~ok]).min():.0f}-{np.abs(clat[~ok]).max():.0f})")
    g_im = amean(np.nanmean(pr, 0), a, ok)

    G = xr.open_dataset(GPCP)["sat_gauge_precip"].sel(time=slice("2019-08-01", "2020-07-31"))
    G = G.assign_coords(lon=np.mod(G.lon, 360)).sortby("lon")

    def on_footprint(field):
        return field.sel(lat=xr.DataArray(clat), lon=xr.DataArray(clon), method="nearest").values

    g_gp_full = float(G.mean("time").weighted(np.cos(np.radians(G.lat))).mean())
    g_gp = amean(on_footprint(G.mean("time")), a, ok)
    print(f"  IMERG annual mean (valid footprint)   {g_im:.4f} mm/day")
    print(f"  GPCP v3.2, same footprint              {g_gp:.4f} mm/day  (full globe {g_gp_full:.4f})")
    print(f"  IMERG - GPCP                           {g_im - g_gp:+.4f} mm/day ({(g_im - g_gp) / g_gp:+.1%})")
    im_m = im.PRECT.resample(time="MS").mean().values * 86400e3
    d = [amean(im_m[k], a, ok) - amean(on_footprint(G.isel(time=k)), a, ok) for k in range(12)]
    print("  monthly IMERG - GPCP (mm/day):", np.round(d, 3))
    print("  9.5% bias error (Adler et al.) on the IMERG mean = "
          f"{0.095 * g_im:.3f} mm/day = {0.095 * g_im / 86400e3:.2e} m/s")


def lwp():
    m = xr.open_dataset(MAC, decode_times=False)
    a, L, E = m.area.values, m.totallwp.values, m.totallwp_error.values
    ok = np.isfinite(L) & np.isfinite(E)
    gm = [amean(L[k], a, ok[k]) for k in range(12)]
    corr = [amean(E[k], a, ok[k]) for k in range(12)]
    rand = [np.sqrt(np.sum((a[ok[k]] * E[k][ok[k]]) ** 2)) / np.sum(a[ok[k]]) for k in range(12)]
    print(f"MAC-LWP (calendar 2020, {m.sizes['time']} months), valid {ok.mean():.0%} of columns")
    print(f"  global mean LWP                        {np.mean(gm):.1f} g/m^2")
    print(f"  global error, fully correlated         {np.mean(corr):.1f} g/m^2 ({np.mean(corr) / np.mean(gm):.1%})")
    print(f"  global error, fully random (annual)    {np.sqrt(np.sum(np.square(rand))) / 12:.3f} g/m^2")
    t = xr.open_dataset(TLWP_OBS, decode_times=False).totallwp.values[0]
    okt = np.isfinite(t)
    g = amean(t, a, okt)
    print(f"  our TLWP obs (Aug19-Jul20) global mean {g:.1f} g/m^2 -> 10% = {0.1 * g / 1000:.4f} kg/m^2")


def ppe_spread():
    with open(SCALERS, "rb") as f:
        s = pickle.load(f)
    print("PPE spread of the global feature (ne128 scaler, physical units):")
    for v in ["PCP", "OSR", "OLR", "TLWP"]:
        sc = s[f"Y_pipeline_{v}"]
        print(f"  {v:5s} mean {sc.mean_[-1]:.4g}  sd {sc.scale_[-1]:.4g}")


if __name__ == "__main__":
    precip()
    print()
    lwp()
    print()
    ppe_spread()
