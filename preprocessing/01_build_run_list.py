"""
Step 1: Build filtered run lists for DY1 and DY2.

- Load parameter JSON
- Enumerate m**** and opt**** folders for each simulation day
- Remove known-bad / incomplete runs
- Filter by p3_ice_sed_knob >= 1.0
- Compute intersection of DY1 and DY2 valid run names

Output: run_list.pkl
  {
    'sim_names': list[str],           # intersection, ordered by DY1
    'DY1_sim_names': np.ndarray[str], # all DY1 runs passing ice_sed filter
    'DY2_sim_names': np.ndarray[str], # all DY2 runs passing ice_sed filter
    'DY1_filename_list': np.ndarray[str],
    'DY2_filename_list': np.ndarray[str],
    'ppe_params': pd.DataFrame,       # params filtered to sim_names
  }
"""

import os
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATH_TO_JSON = (
    "/global/cfs/cdirs/e3sm/jpaige3/ESEm/"
    "SCREAM.2024-autocal-00.ne1024pg2-params.json"
)

DY1_PATH = "/pscratch/sd/j/jpaige3/dy1ne1024/"
DY2_PATH = (
    "/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/"
    "SCREAM.2024-autocal-00.ne1024pg2/"
)

DY1_NC_SUFFIX = (
    "SCREAM.2024-autocal-00.ne1024pg2/run/"
    "output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24."
    "2016-08-07-00000.nc"
)
DY2_NC_SUFFIX = (
    "SCREAM.2024-autocal-00.ne1024pg2/run/"
    "output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24."
    "2020-01-26-00000.nc"
)

OUTPUT_PKL = "run_list.pkl"

# ---------------------------------------------------------------------------
# Load parameter JSON
# ---------------------------------------------------------------------------
ppe_params_all = pd.read_json(PATH_TO_JSON)

# ---------------------------------------------------------------------------
# Build DY2 folder list
# ---------------------------------------------------------------------------
DY2_folders = []
for m in range(0, 301):
    folder = "m{:04}".format(m)
    if os.path.exists(DY2_PATH + folder):
        DY2_folders.append(m)
DY2_folders = ["m{:04}".format(m) for m in DY2_folders]

for file in os.listdir(DY2_PATH):
    if file.startswith("opt"):
        DY2_folders.append(file)

# Remove known incomplete / invalid DY2 runs
for bad in ["m0230", "optmar20day5", "m0024", "m0025", "m0061", "optmar22hd"]:
    if bad in DY2_folders:
        DY2_folders.remove(bad)

# ---------------------------------------------------------------------------
# Build DY1 folder list
# ---------------------------------------------------------------------------
DY1_folders = []
for m in range(0, 301):
    folder = "m{:04}".format(m)
    if os.path.exists(DY1_PATH + folder):
        DY1_folders.append(m)
DY1_folders = ["m{:04}".format(m) for m in DY1_folders]

for file in os.listdir(DY1_PATH):
    if file.startswith("opt"):
        DY1_folders.append(file)

# Remove known invalid / incomplete DY1 runs
DY1_bad = [
    "m0024", "m0025", "m0061", "optmar22hd",
    "m0262", "m0263", "m0264", "m0266", "m0267", "m0270",
    "m0272", "m0274", "m0275", "m0279", "m0289", "m0290",
    "m0292", "m0293", "m0294", "m0295", "m0296", "m0299",
    "m0300",
    "optmar15seed0", "optmar27a", "optmar20dayAll",
    "optmar20day2-fail", "optmar15b", "optmar20day2-ltend",
    "m0230", "optmar20day5",
]
for bad in DY1_bad:
    if bad in DY1_folders:
        DY1_folders.remove(bad)

# ---------------------------------------------------------------------------
# Filter by p3_ice_sed_knob >= 1.0
# ---------------------------------------------------------------------------
def _filter_ice_sed(folders, base_path, nc_suffix):
    file_check = np.zeros(len(folders), dtype=bool)
    for i, member in enumerate(folders):
        if float(ppe_params_all["p3_ice_sed_knob"][member]) >= 1.0:
            file_check[i] = True
    filtered_names = np.array(folders)[file_check]
    filtered_files = np.array(
        [base_path + f + "/" + nc_suffix for f in folders]
    )[file_check]
    return filtered_names, filtered_files


DY1_sim_names, DY1_filename_list = _filter_ice_sed(
    DY1_folders, DY1_PATH, DY1_NC_SUFFIX
)
DY2_sim_names, DY2_filename_list = _filter_ice_sed(
    DY2_folders, DY2_PATH, DY2_NC_SUFFIX
)

print("DY1 runs after filter:", len(DY1_sim_names))
print("DY2 runs after filter:", len(DY2_sim_names))
print("In DY1 but not DY2:", list(set(DY1_sim_names) - set(DY2_sim_names)))
print("In DY2 but not DY1:", list(set(DY2_sim_names) - set(DY1_sim_names)))

# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------
sim_names = [sim for sim in DY1_sim_names if sim in DY2_sim_names]
ppe_params = ppe_params_all[ppe_params_all.index.isin(sim_names)]
print("Runs in intersection:", len(sim_names))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(
        {
            "sim_names": sim_names,
            "DY1_sim_names": DY1_sim_names,
            "DY2_sim_names": DY2_sim_names,
            "DY1_filename_list": DY1_filename_list,
            "DY2_filename_list": DY2_filename_list,
            "ppe_params": ppe_params,
        },
        f,
    )
print(f"Saved {OUTPUT_PKL}")
