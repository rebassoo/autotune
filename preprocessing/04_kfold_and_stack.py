"""
Step 4: K-fold cross-validation splits and stack ZRG DataFrames into
numpy arrays ready for the transform steps in src/autotune_gp/preprocess.py.

Inputs:
  run_list.pkl  (from 01_build_run_list.py)
  zrg_data.pkl  (from 03_compute_zrg.py)

Outputs: kfold_data.pkl
  {
    'folds': list of dicts, one per fold:
      {
        'k': int,
        'train_run_labels': list[str],
        'test_run_labels':  list[str],
        'X_train': np.ndarray,  # (n_train, n_params)  — raw, not normalised
        'X_test':  np.ndarray,  # (n_test,  n_params)
        'Y_train_ZRG': np.ndarray,  # (n_train, n_features, 4)
        'Y_test_ZRG':  np.ndarray,  # (n_test,  n_features, 4)
        # Per-variable DataFrames (needed by fit_transform_Y / transform_obs)
        'PCP_train':  pd.DataFrame,
        'TLWP_train': pd.DataFrame,
        'OSR_train':  pd.DataFrame,
        'OLR_train':  pd.DataFrame,
        'PCP_test':   pd.DataFrame,
        'TLWP_test':  pd.DataFrame,
        'OSR_test':   pd.DataFrame,
        'OLR_test':   pd.DataFrame,
      }
  }

Next step:
  from autotune_gp.transforms import fit_transform_X, fit_transform_Y
  X_sc, X_train_norm = fit_transform_X(fold['X_train'])
  Y_scs, Y_train_norm = fit_transform_Y(fold['Y_train_ZRG'])
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

INPUT_RUN_LIST = "run_list.pkl"
INPUT_ZRG_DATA = "zrg_data.pkl"
OUTPUT_PKL     = "kfold_data.pkl"

N_FOLDS = 5
RANDOM_STATE = 2

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
with open(INPUT_RUN_LIST, "rb") as f:
    run_list = pickle.load(f)
ppe_params = run_list["ppe_params"]

with open(INPUT_ZRG_DATA, "rb") as f:
    zrg_data = pickle.load(f)

PCP_zrg_ppedataset  = zrg_data["PCP_zrg_ppedataset"]
TLWP_zrg_ppedataset = zrg_data["TLWP_zrg_ppedataset"]
OSR_zrg_ppedataset  = zrg_data["OSR_zrg_ppedataset"]
OLR_zrg_ppedataset  = zrg_data["OLR_zrg_ppedataset"]

# ---------------------------------------------------------------------------
# K-fold splits
# ---------------------------------------------------------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

folds = []
for k, (train_idx, test_idx) in enumerate(kf.split(ppe_params)):
    X_train_df = ppe_params.iloc[train_idx]
    X_test_df  = ppe_params.iloc[test_idx]
    train_run_labels = X_train_df.index.to_list()
    test_run_labels  = X_test_df.index.to_list()

    print(f"Fold {k}: train={len(train_run_labels)}, test={len(test_run_labels)}")

    # Per-variable ZRG slices
    PCP_train  = PCP_zrg_ppedataset.loc[train_run_labels].copy()
    TLWP_train = TLWP_zrg_ppedataset.loc[train_run_labels].copy()
    OSR_train  = OSR_zrg_ppedataset.loc[train_run_labels].copy()
    OLR_train  = OLR_zrg_ppedataset.loc[train_run_labels].copy()

    PCP_test   = PCP_zrg_ppedataset.loc[test_run_labels].copy()
    TLWP_test  = TLWP_zrg_ppedataset.loc[test_run_labels].copy()
    OSR_test   = OSR_zrg_ppedataset.loc[test_run_labels].copy()
    OLR_test   = OLR_zrg_ppedataset.loc[test_run_labels].copy()

    # Ensure columns are strings (required by downstream transform steps)
    for df in [PCP_train, TLWP_train, OSR_train, OLR_train,
               PCP_test,  TLWP_test,  OSR_test,  OLR_test]:
        df.columns = df.columns.astype(str)

    # Stack into (n_samples, n_features, 4) arrays
    Y_train_ZRG = np.stack(
        (PCP_train, TLWP_train, OSR_train, OLR_train), axis=0
    )
    Y_train_ZRG = np.transpose(Y_train_ZRG, (1, 2, 0))

    Y_test_ZRG = np.stack(
        (PCP_test, TLWP_test, OSR_test, OLR_test), axis=0
    )
    Y_test_ZRG = np.transpose(Y_test_ZRG, (1, 2, 0))

    print(f"  X_train: {X_train_df.to_numpy().shape}, Y_train_ZRG: {Y_train_ZRG.shape}")
    print(f"  X_test:  {X_test_df.to_numpy().shape},  Y_test_ZRG:  {Y_test_ZRG.shape}")

    folds.append(
        {
            "k":                 k,
            "train_run_labels":  train_run_labels,
            "test_run_labels":   test_run_labels,
            "X_train":           X_train_df.to_numpy(),
            "X_test":            X_test_df.to_numpy(),
            "Y_train_ZRG":       Y_train_ZRG,
            "Y_test_ZRG":        Y_test_ZRG,
            # Keep DataFrames for fit_transform_Y column alignment
            "PCP_train":  PCP_train,  "TLWP_train": TLWP_train,
            "OSR_train":  OSR_train,  "OLR_train":  OLR_train,
            "PCP_test":   PCP_test,   "TLWP_test":  TLWP_test,
            "OSR_test":   OSR_test,   "OLR_test":   OLR_test,
        }
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump({"folds": folds}, f)
print(f"Saved {OUTPUT_PKL}  ({N_FOLDS} folds)")
