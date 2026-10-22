# Observation-based global-mean errors (calibration experiment E2)

These are the fixed global-mean error values passed as `--global` to
`scripts/calibrate_model_error.py` in experiment **E2** of the n=85 model-error
calibration (job 60114920, output `/pscratch/sd/r/rebassoo/autotune/calib_n85_E2_obsglobal`).
The procedure itself is in [model_error_residual_calibration.md](model_error_residual_calibration.md).
All numbers below are reproduced by `scripts/compute_global_obs_errors.py`,
except the CERES values, which come from the literature.

## Why the global block needs an external value

Each variable enters the cost as 15 zonal bands, 6 regions and 1 global mean.
Zonal and regional σ are estimated from the residuals at the optimum. The
global block has one residual per variable — no variance can be estimated from
it, and it is the residual the optimizer drives hardest to zero. So its σ must
be specified. Three treatments are being compared:

| Experiment | Global-mean term | Job |
|---|---|---|
| E1 | removed from the cost | 60114918 |
| **E2** | **fixed observation-based σ (this document)** | 60114920 |
| E3 | σ = 0 (GP variance only) | 60114922 |

All three start at σ = 0 everywhere (round 0) and estimate zonal/regional σ
from residuals from round 1 on.

## Values used

| Variable | σ_global (physical) | ≈ | Fraction of PPE spread* | Basis |
|---|---|---|---|---|
| OSR | 2.5 W/m² | — | 0.17 | CERES EBAF uncertainty (literature) |
| OLR | 2.5 W/m² | — | 0.39 | CERES EBAF uncertainty (literature) |
| PCP | 3.2e-9 m/s | 0.28 mm/day, 9.5 % | 1.02 | GPCP bias-error estimate (literature) |
| TLWP | 0.012 kg/m² | 12 g/m², 10 % | 0.11 | nominal; MAC-LWP brackets it (computed) |

\* PPE standard deviation of the global feature from the ne128 Y scaler
(PCP 3.13e-9 m/s, OSR 14.9 W/m², OLR 6.36 W/m², TLWP 0.113 kg/m²). This is the
σ in the normalized units the cost uses.

Window for everything: Aug 2019 – Jul 2020 (the 12-month calibration window),
except MAC-LWP's error field, which exists only for calendar 2020.

These are **observational** errors only. Model error in the global mean is not
included, so they are lower bounds on the total global error. That is
inconsistent with the zonal/regional σ, which are estimated from residuals and
therefore contain model + observation error together; E1 and E3 bracket the
effect of the choice.

## OSR and OLR: 2.5 W/m²

Loeb et al. (2018, J. Climate, CERES EBAF Ed4.0) give a 1σ uncertainty in
1°×1° regional monthly all-sky TOA flux of ~3 W/m² (Terra-only period) and
**2.5 W/m²** (Terra–Aqua period, which covers 2019–2020), for both SW and LW.
Ed4.1 all-sky TOA fluxes are identical to Ed4.0 (Ed4.1 data-quality summary),
so this carries over to the CERES file used here.

That number is regional-monthly, not global-annual. A global-annual mean
averages away the random part, but the dominant term is absolute calibration,
which does not average away, so a separate global number would be somewhat
smaller but of the same order. 2.5 W/m² is therefore taken as a conservative
global value. I did not find a published global-mean-only SW/LW uncertainty
for Ed4.1; if one is found it should replace this.

## PCP: 9.5 % → 3.2e-9 m/s

**Independent-product comparison (computed, not used).** IMERG (the PCP obs
file, daily on ne32pg2) vs GPCP v3.2 (`sat_gauge_precip`, monthly 0.5°),
both averaged over IMERG's valid footprint (IMERG lacks 8.5 % of columns, all
poleward of 61°):

| | mm/day |
|---|---|
| IMERG, annual mean | 3.0245 |
| GPCP v3.2, same footprint | 3.0111 |
| IMERG − GPCP | +0.013 (+0.4 %) |
| monthly differences | −0.12 … +0.19 |

The 0.4 % agreement is **not** a usable error estimate: GPCP v3.2 uses IMERG as
its satellite input between 60°S and 60°N, so the two products share most of
their error. (No independent GPCP version covers 2019–2020: v2.3 in the e3sm_diags
archive ends in 2017.)

**Literature value (used).** Adler et al.'s GPCP bias-error analysis (taking
the spread of independent products within 50 % of GPCP as the systematic error)
gives about 0.25 mm/day on a 2.64 mm/day global mean, i.e. **≈ 9.5 %**,
with ~10 % in the deep tropics and ~15 % in mid-latitudes. The global figure
was read from Adler's GPM/PMM presentation, not from the paper's text; it
should be confirmed against Adler, Gu & Huffman (2012, J. Appl. Meteor.
Climatol. 51, 84–99) before being cited.

Applied to the global PCP value the cost actually uses (3.352e-8 m/s in
`obs.pkl`, i.e. 2.90 mm/day): 0.095 × 3.352e-8 = **3.2e-9 m/s**. (Relative to
the raw IMERG footprint mean it would be 3.33e-9; the difference is how
preprocessing forms the global mean.)

Note: this is ≈ 1.0 PPE standard deviations of global PCP. In E2 the global
PCP term therefore carries almost no weight — the published uncertainty of
global-mean precipitation is as large as the spread the parameters produce.

## TLWP: nominal 10 % → 0.012 kg/m²

The TLWP observations are MAC-LWP total (cloud + rain) liquid water path,
ocean only (~69 % of columns valid). The source file
`MAC_2020_ne32pg2.nc` carries `totallwp_error`, a 1σ error on each monthly
grid-box mean. The product documentation states that this field is the
**statistical (random) error only; systematic errors are not included because
they are not fully characterized.**

From that field (calendar 2020):

| | g/m² |
|---|---|
| global-mean LWP | 113.4 |
| area-mean monthly 1σ — global error if errors are **fully correlated** | 15.1 (13.3 %) |
| global annual error if errors are **independent** in space and time | 0.04 |

The independent case is what the field actually describes, and it is
negligible for a global mean. The fully-correlated case treats the random
error as if it were systematic and is an upper-end guess. The systematic error
— the relevant one for a global mean — is not quantified by the product, and I
found no published percentage for MAC-LWP or its UWisc predecessor (O'Dell et
al. 2008).

**Choice:** a nominal **10 %** of the observed global mean used in the cost
(118.7 g/m² for Aug 2019 – Jul 2020) = 11.9 g/m² ≈ **0.012 kg/m²**, between
the two bounds and comparable to the precipitation value. This is a judgement,
not a measurement.

## Caveats

- OSR/OLR and PCP rely on literature values that should be confirmed against
  the primary papers before publication (see above).
- TLWP is nominal.
- Observational error only; global model error is not included.
- If E2's optimum differs materially from E1 and E3, these choices matter and
  deserve more care; if all three agree, they do not.

## Sources

- Loeb, N. G., et al. (2018), Clouds and the Earth's Radiant Energy System
  (CERES) Energy Balanced and Filled (EBAF) TOA Edition-4.0 data product,
  *J. Climate* 31, 895–918.
  [pdf](https://www.cen.uni-hamburg.de/en/icdc/data/atmosphere/docs-atmo/loebetal-jcli-2018.pdf)
- CERES EBAF Ed4.1 Data Quality Summary.
  [pdf](https://www.cen.uni-hamburg.de/en/icdc/data/atmosphere/docs-atmo/ceres-ebaf-ed4-1-dqs.pdf)
- Adler, R. F., GPCP bias-error presentation, NASA PMM Science Team Meeting 2011.
  [pdf](https://gpm.nasa.gov/sites/default/files/meeting_files/2011_PMM_Science_Team_Meeting_Files/Presentations/Session%205/AdlerPMM11.11.pdf);
  paper: Adler, Gu & Huffman (2012), *J. Appl. Meteor. Climatol.* 51, 84–99.
- GPCP Monthly, NCAR Climate Data Guide.
  [link](https://climatedataguide.ucar.edu/climate-data/gpcp-monthly-global-precipitation-climatology-project)
- MAC-LWP total LWP L3 monthly product metadata (error-field description).
  [link](https://catalog.data.gov/dataset/multisensor-advanced-climatology-total-liquid-water-path-l3-monthly-1-degree-x-1-degree-v1-1136c);
  Elsaesser, G. S., et al. (2017), *J. Climate* 30, 10193–10210.

Data files: IMERG
`/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/PRECT_IMERG_ne32pg2_201908_202007.nc`;
GPCP `/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/GPCP_v3.2/sat_gauge_precip_198301_202112.nc`;
MAC-LWP `/global/cfs/cdirs/e3sm/beydoun/autotune_utils/MAC_2020_ne32pg2.nc`;
TLWP obs `/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/TLWP_MACLWP_ne32pg2_201908_202007.nc`.
