from __future__ import annotations
import numpy as np
from sklearn import preprocessing

def fit_transform_X(X_train, param_bounds=None):
    """Fit MinMaxScaler on param_bounds if provided (shape (2, n_params): [lows, highs]),
    otherwise fall back to fitting on X_train data range."""
    sc = preprocessing.MinMaxScaler()
    if param_bounds is not None:
        sc.fit(param_bounds)  # shape (2, n_params) -> maps theoretical [low,high] to [0,1]
    else:
        sc.fit(X_train)
    return sc, sc.transform(X_train)

def fit_transform_Y(Y_train_ZRG: np.ndarray):
    # Y_train_ZRG: (n_train, n_feat, 4)
    scalers = []
    Ys = []
    for j in range(Y_train_ZRG.shape[-1]):
        sc = preprocessing.StandardScaler()
        sc.fit(Y_train_ZRG[:, :, j])
        scalers.append(sc)
        Ys.append(sc.transform(Y_train_ZRG[:, :, j]))
    Y_norm = np.stack(Ys, axis=0).transpose(1, 2, 0)
    return scalers, Y_norm

def transform_obs(obs_parts, scalers):
    Ys = []
    for part, sc in zip(obs_parts, scalers):
        Ys.append(sc.transform(part))
    obs_norm = np.stack(Ys, axis=0).transpose(1, 2, 0)
    return obs_norm
