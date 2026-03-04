from __future__ import annotations
import pickle

def load_obs(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_gp_proj(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def split_zrg_obs(zrg_obs_df, n_vars: int = 4):
    # Split columns into n_vars equal blocks (one per variable in cfg.data.variables)
    n_cols_per = zrg_obs_df.shape[1] // n_vars
    parts = [zrg_obs_df.iloc[:, i*n_cols_per:(i+1)*n_cols_per] for i in range(n_vars)]
    for p in parts:
        p.columns = p.columns.astype(str)
    return parts
