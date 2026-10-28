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
