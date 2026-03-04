from pathlib import Path

import numpy as np
from sklearn.metrics import root_mean_squared_error

from autotune_gp.backend import get_backend
from autotune_gp.cost import zrg_cost_function_rmse_like_reference


def _extract_function_source(py_path: Path, fn_name: str) -> str:
    """
    Extract a top-level function's source by text scanning (no AST parsing),
    so it works even if the file contains non-Python notebook artifacts.
    """
    lines = py_path.read_text(encoding="utf-8").splitlines(True)

    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {fn_name}"):
            start = i
            break
    if start is None:
        raise AssertionError(f"Function {fn_name!r} not found in {py_path}")

    def_indent = len(lines[start]) - len(lines[start].lstrip(" \t"))

    out = [lines[start]]
    for j in range(start + 1, len(lines)):
        line = lines[j]
        stripped = line.strip()

        if stripped == "":
            out.append(line)
            continue

        indent = len(line) - len(line.lstrip(" \t"))

        # function ends when indentation returns to the def indentation (top-level)
        # and it isn't just a decorator/comment.
        if indent <= def_indent and not line.lstrip().startswith(("#", "@")):
            break

        out.append(line)

    return "".join(out)


def test_new_cost_matches_reference_using_sklearn_rmse():
    """
    Verify that the refactored repo cost function gives the SAME numerical
    result as the original script's ZRG_cost_function_rmse for identical inputs,
    using sklearn.metrics.root_mean_squared_error directly.
    """
    # Reference script tracked verbatim
    ref = (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "Final_Surrogate_ToShare_Background.py"
    )
    fn_src = _extract_function_source(ref, "ZRG_cost_function_rmse")

    # Exec namespace for the extracted reference function
    ns = {}
    ns["np"] = np

    # Provide sklearn RMSE exactly (both names to cover either usage pattern)
    ns["root_mean_squared_error"] = root_mean_squared_error
    ns["rmse"] = root_mean_squared_error

    # Globals the reference function expects
    ns["regions_list"] = [
        "poles",
        "extratropical_land",
        "extratropical_ocean",
        "tropical_land",
        "ascending_tropical_ocean",
        "descending_tropical_ocean",
    ]
    ns["DY_weights_dict"] = {"DY1": 0.5, "DY2": 0.5}

    # IMPORTANT: must match the ZRG vector construction assumption for n_zonal.
    # If your real ZRG has a different zonal count, change this.
    ns["lat_bands"] = np.linspace(-90, 90, 18)

    # Define the reference function
    exec(fn_src, ns, ns)
    old_cost = ns["ZRG_cost_function_rmse"]

    # Deterministic synthetic inputs with expected layout:
    n_obs = 1
    n_zonal = len(ns["lat_bands"])
    n_regions = len(ns["regions_list"])
    n_feat = 2 * (n_zonal + n_regions + 1)

    rng = np.random.RandomState(0)
    preds = rng.normal(size=(n_obs, n_feat, 4)).astype(np.float64)
    obs = rng.normal(size=(n_obs, n_feat, 4)).astype(np.float64)

    var_w = {"PCP": 0.25, "TLWP": 0.25, "OSR": 0.25, "OLR": 0.25}
    zrg_w = {"zonal": 1 / 3, "regional": 1 / 3, "global": 1 / 3}

    # Old vs new
    c_old = float(old_cost(preds, obs, var_w, zrg_w))

    backend = get_backend("numpy", "cpu")
    c_new = float(
        zrg_cost_function_rmse_like_reference(
            preds,
            obs,
            var_w,
            zrg_w,
            ns["DY_weights_dict"],
            n_zonal=n_zonal,
            n_regions=n_regions,
            backend=backend,
        )
    )

    np.testing.assert_allclose(c_new, c_old, rtol=0.0, atol=1e-12)
