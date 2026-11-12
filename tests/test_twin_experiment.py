"""Twin-experiment setup: the target is laid out as obs, and hold-out drops one member."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from make_twin_experiment import hold_out, pseudo_obs   # noqa: E402
from autotune_gp.io import split_zrg_obs                # noqa: E402


def test_pseudo_obs_splits_back_to_each_variable():
    y = np.arange(12.0).reshape(4, 3)                    # 4 features, 3 variables
    cols = [f"c{i}" for i in range(12)]
    obs = pd.DataFrame(np.zeros((1, 12)), columns=cols)
    out = pseudo_obs(obs, y)
    assert list(out.columns) == cols
    for j, part in enumerate(split_zrg_obs(out.copy(), n_vars=3)):
        np.testing.assert_array_equal(part.to_numpy().ravel(), y[:, j])


def test_pseudo_obs_rejects_wrong_shape():
    with pytest.raises(ValueError):
        pseudo_obs(pd.DataFrame(np.zeros((1, 10))), np.zeros((4, 3)))


def test_hold_out_removes_only_that_member():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=["m1", "m2", "m3"])
    Y = np.arange(3.0)[:, None, None]
    out = hold_out({"X_train": X, "Y_train": Y}, "m2")
    assert list(out["X_train"].index) == ["m1", "m3"]
    np.testing.assert_array_equal(out["Y_train"].ravel(), [0.0, 2.0])
    with pytest.raises(ValueError):
        hold_out({"X_train": X, "Y_train": Y}, "absent")
