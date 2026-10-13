from __future__ import annotations
import numpy as np
from sklearn import preprocessing


class LogMinMaxScaler:
    """MinMax scaling to [0, 1], applied in log space for selected dimensions.

    Six of the 19 EAMxx parameters are sampled log-uniformly in the design
    (c_diag_3rd_mom, coeff_kh, coeff_km, accretion_prefactor, max_total_ni,
    autoconversion_prefactor -- confirmed against the design marginals with a
    KS test). Scaling those linearly leaves the training inputs bunched against
    the low edge of [0, 1], which is a poor input space for an ARD RBF: one
    lengthscale has to cover both the dense low end and the sparse high end.
    Taking the log first makes the design uniform in the GP's input space.

    Deliberately duck-types sklearn's MinMaxScaler (fit / transform /
    inverse_transform, data_min_ / data_max_) because this object is pickled
    into scalers.pkl under 'X_pipeline' and many scripts call those directly.
    With log_mask all-False it is numerically identical to MinMaxScaler fitted
    on the same bounds, so existing runs are unaffected.
    """

    def __init__(self, log_mask=None):
        self.log_mask = None if log_mask is None else np.asarray(log_mask, dtype=bool)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        if self.log_mask is None:
            self.log_mask = np.zeros(n, dtype=bool)
        if self.log_mask.shape[0] != n:
            raise ValueError(f"log_mask has {self.log_mask.shape[0]} entries, X has {n} columns.")
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        bad = self.log_mask & (self.data_min_ <= 0)
        if bad.any():
            raise ValueError(f"log-scaled dimensions {np.where(bad)[0].tolist()} have a "
                             "non-positive lower bound; log scaling needs strictly positive bounds.")
        # Only the masked columns are logged. np.where would evaluate np.log on
        # every column, so a linear parameter whose bound is legitimately 0
        # would raise a divide-by-zero warning for a value that is then thrown
        # away -- and exp() on a large linear bound would overflow likewise.
        self._lo = self.data_min_.astype(float).copy()
        self._hi = self.data_max_.astype(float).copy()
        m = self.log_mask
        self._lo[m] = np.log(self.data_min_[m])
        self._hi[m] = np.log(self.data_max_[m])
        self._span = np.where(self._hi > self._lo, self._hi - self._lo, 1.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = X.astype(float).copy()
        m = self.log_mask
        if m.any():
            Z[:, m] = np.log(np.clip(X[:, m], 1e-300, None))
        return (Z - self._lo) / self._span

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, U):
        U = np.asarray(U, dtype=float)
        Z = self._lo + U * self._span
        m = self.log_mask
        if m.any():
            Z = Z.copy()
            Z[:, m] = np.exp(Z[:, m])
        return Z


def fit_transform_X(X_train, param_bounds=None, log_mask=None):
    """Fit the X scaler on param_bounds if provided (shape (2, n_params):
    [lows, highs]), otherwise on the X_train data range.

    log_mask : boolean array over the n_params columns, or None. True entries
    are scaled in log space (see LogMinMaxScaler). None/all-False reproduces
    the previous MinMaxScaler behaviour exactly.
    """
    sc = LogMinMaxScaler(log_mask=log_mask)
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
