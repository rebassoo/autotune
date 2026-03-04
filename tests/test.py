import numpy as np
from sklearn.metrics import root_mean_squared_error

rng = np.random.RandomState(0)
a = rng.normal(size=(3, 18))
b = rng.normal(size=(3, 18))

rmse_sklearn = root_mean_squared_error(a, b)              # per-column RMSE then average
rmse_flat    = np.sqrt(np.mean((a-b)**2))                 # flattened RMSE

print(rmse_sklearn, rmse_flat)
