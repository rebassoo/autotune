from __future__ import annotations
import numpy as np
from esem import gp_model

class GPWrapper:
    def __init__(self, X_train_norm: np.ndarray, Y_train_norm: np.ndarray):
        self.model = gp_model(X_train_norm, Y_train_norm)

    def train(self, tf_determinism: bool = True):
        if tf_determinism:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
        self.model.train()

    def predict(self, x: np.ndarray):
        # returns (mean, var); x can be 1-D (single point) or 2-D (batch)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.model.predict(x)
