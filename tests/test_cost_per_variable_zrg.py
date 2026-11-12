"""Per-variable zonal/regional/global weights (zrg_w['per_variable'])."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from autotune_gp.backend import get_backend
from autotune_gp.config import _zrg_weights
from autotune_gp.cost import (zrg_cost_function_gaussian_nll, zrg_cost_function_laplace,
                              zrg_cost_function_mae_weighted, zrg_cost_function_student_t)

N_Z, N_R = 4, 2
NAMES = ["PCP", "OSR", "OLR"]
VAR_W = {"PCP": 0.3, "OSR": 0.1, "OLR": 0.3}
BASE = {"zonal": 1 / 3, "regional": 1 / 3, "global": 1 / 3}
PCP_Z = {"zonal": 0.425, "regional": 0.425, "global": 0.15}


def _data():
    rng = np.random.default_rng(0)
    shape = (1, N_Z + N_R + 1, len(NAMES))
    return rng.normal(size=shape), rng.uniform(0.1, 1, size=shape), rng.normal(size=shape)


def _call(fn, var_w, zrg_w, preds, var, obs):
    kw = dict(dy_w={"ANN": 1.0}, n_zonal=N_Z, n_regions=N_R,
              backend=get_backend("numpy", "cpu"), var_names=NAMES)
    if fn is zrg_cost_function_mae_weighted:
        return fn(preds, obs, var_w, zrg_w, **kw)
    return fn(preds, var, obs, var_w, zrg_w, **kw)


FNS = [zrg_cost_function_gaussian_nll, zrg_cost_function_laplace,
       zrg_cost_function_mae_weighted, zrg_cost_function_student_t]


@pytest.mark.parametrize("fn", FNS)
def test_override_is_sum_of_groups(fn):
    p, v, o = _data()
    got = _call(fn, VAR_W, {**BASE, "per_variable": {"PCP": PCP_Z}}, p, v, o)
    want = (_call(fn, {"PCP": 0.3}, PCP_Z, p, v, o)
            + _call(fn, {"OSR": 0.1, "OLR": 0.3}, BASE, p, v, o))
    assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("fn", FNS)
def test_override_equal_to_base_changes_nothing(fn):
    p, v, o = _data()
    plain = _call(fn, VAR_W, BASE, p, v, o)
    assert _call(fn, VAR_W, {**BASE, "per_variable": {"PCP": dict(BASE)}}, p, v, o) == \
        pytest.approx(plain, rel=1e-12)


def test_override_changes_cost():
    p, v, o = _data()
    fn = zrg_cost_function_student_t
    assert _call(fn, VAR_W, {**BASE, "per_variable": {"PCP": PCP_Z}}, p, v, o) != \
        pytest.approx(_call(fn, VAR_W, BASE, p, v, o))


def test_config_folds_per_variable():
    z = _zrg_weights({"zrg": BASE, "zrg_per_variable": {"PCP": PCP_Z}})
    assert z["per_variable"] == {"PCP": PCP_Z} and z["global"] == BASE["global"]
    assert "per_variable" not in _zrg_weights({"zrg": BASE})
    with pytest.raises(ValueError):
        _zrg_weights({"zrg": BASE, "zrg_per_variable": {"PCP": {"global": 0.0}}})


def test_unknown_variable_rejected():
    p, v, o = _data()
    with pytest.raises(ValueError):
        _call(zrg_cost_function_student_t, VAR_W, {**BASE, "per_variable": {"TLWP": PCP_Z}}, p, v, o)


ZW = [0.0, 1.0, 1.0, 0.0]          # mask zonal bands 0 and 3
RW = [1.0, 0.0]                    # mask region 1


@pytest.mark.parametrize("fn", FNS)
def test_per_variable_feature_mask(fn):
    """A masked feature has no effect; an unmasked variable keeps the shared weights."""
    p, v, o = _data()
    zrg = {**BASE, "per_variable": {"OSR": {"zonal_weights": ZW, "regional_weights": RW}}}
    got = _call(fn, VAR_W, zrg, p, v, o)
    p2 = p.copy()
    j = NAMES.index("OSR")
    p2[0, [0, 3, N_Z + 1], j] += 50.0          # move only the masked OSR features
    assert _call(fn, VAR_W, zrg, p2, v, o) == pytest.approx(got, rel=1e-12)
    want = (_call(fn, {"PCP": 0.3, "OLR": 0.3}, BASE, p, v, o)
            + _call_fw(fn, {"OSR": 0.1}, BASE, p, v, o, ZW, RW))
    assert got == pytest.approx(want, rel=1e-12)


def _call_fw(fn, var_w, zrg_w, preds, var, obs, zw, rw):
    kw = dict(dy_w={"ANN": 1.0}, n_zonal=N_Z, n_regions=N_R,
              backend=get_backend("numpy", "cpu"), var_names=NAMES,
              zonal_weights=zw, regional_weights=rw)
    if fn is zrg_cost_function_mae_weighted:
        return fn(preds, obs, var_w, zrg_w, **kw)
    return fn(preds, var, obs, var_w, zrg_w, **kw)


def test_config_folds_feature_weights():
    z = _zrg_weights({"zrg": BASE, "zrg_per_variable": {
        "SLP": {"zonal": 1 / 3, "regional": 1 / 3, "global": 0.0},
        "OSR": {"zonal_weights": ZW}}})
    assert z["per_variable"]["OSR"] == {"zonal_weights": ZW}
    assert z["per_variable"]["SLP"]["global"] == 0.0
    with pytest.raises(ValueError):
        _zrg_weights({"zrg": BASE, "zrg_per_variable": {"OSR": {"zonal_weight": ZW}}})


def test_student_t_logvar_weight_drops_only_the_log_variance_term():
    # Equal variances everywhere: the 0.5*log(v) term is the same constant at every
    # location, so dropping it shifts the cost by exactly that constant (the block
    # weights average to 1); unequal variances: weight 0 no longer rewards low v.
    p, _, o = _data()
    kw = dict(var_w=VAR_W, zrg_w=BASE, dy_w={"ANN": 1.0}, n_zonal=N_Z, n_regions=N_R,
              backend=get_backend("numpy", "cpu"), var_names=NAMES)
    fn = zrg_cost_function_student_t
    v = np.full_like(p, 0.25)
    full, nolog = fn(p, v, o, **kw), fn(p, v, o, logvar_weight=0.0, **kw)
    assert full - nolog == pytest.approx(0.5 * np.log(0.25) * sum(VAR_W.values()), rel=1e-12)
    # with the residual held at zero, weight 0 makes the cost independent of v
    assert fn(o, np.full_like(p, 0.1), o, logvar_weight=0.0, **kw) == \
        pytest.approx(fn(o, np.full_like(p, 0.9), o, logvar_weight=0.0, **kw), rel=1e-12)
