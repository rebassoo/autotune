import numpy as np
from autotune_gp.backend import get_backend
from autotune_gp.cost import zrg_cost_function_rmse_like_reference

def test_layout_ok():
    backend = get_backend("numpy", "cpu")
    n_z = 18; n_r = 6
    n_feat = 2*(n_z+n_r+1)
    preds = np.zeros((1,n_feat,4))
    obs   = np.zeros((1,n_feat,4))
    var_w = {"PCP":0.25,"TLWP":0.25,"OSR":0.25,"OLR":0.25}
    zrg_w = {"zonal":1/3,"regional":1/3,"global":1/3}
    dy_w  = {"DY1":0.5,"DY2":0.5}
    c = zrg_cost_function_rmse_like_reference(preds, obs, var_w, zrg_w, dy_w, n_z, n_r, backend)
    assert np.isfinite(c)
