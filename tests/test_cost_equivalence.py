from pathlib import Path

import numpy as np

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

        if indent <= def_indent and not line.lstrip().startswith(("#", "@")):
            break

        out.append(line)

    return "".join(out)


def _sklearn_style_root_mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Match sklearn.metrics.root_mean_squared_error for 1D/2D inputs:
      - 1D: sqrt(mean(d^2))
      - 2D: per-column RMSE then average across columns
    """
    d = a - b
    if d.ndim == 1:
        return float(np.sqrt(np.mean(d * d)))
    if d.ndim == 2:
        return float(np.mean(np.sqrt(np.mean(d * d, axis=0))))
    return float(np.sqrt(np.mean(d * d)))


def test_new_cost_matches_reference_function_exactly():
    # Reference script is tracked verbatim in the repo.
    ref = (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "Final_Surrogate_ToShare_Background.py"
    )

    fn_src = _extract_function_source(ref, "ZRG_cost_function_rmse")

    # Namespace to exec ONLY the cost function (no heavy imports / IO).
    ns = {}
    ns["np"] = np

    # Provide what the reference function might call.
    ns["root_mean_squared_error"] = _sklearn_style_root_mean_squared_error
    ns["rmse"] = _sklearn_style_root_mean_squared_error  # alias, in case it uses rmse()

    # Provide globals the reference cost relied on.
    ns["regions_list"] = [
        "poles",
        "extratropical_land",
        "extratropical_ocean",
        "tropical_land",
        "ascending_tropical_ocean",
        "descending_tropical_ocean",
    ]
    ns["DY_weights_dict"] = {"DY1": 0.5, "DY2": 0.5}

    # Must match your ZRG construction assumption for n_zonal.
    ns["lat_bands"] = np.linspace(-90, 90, 18)

    exec(fn_src, ns, ns)
    old_cost = ns["ZRG_cost_function_rmse"]

    # Deterministic synthetic inputs with expected layout.
    n_obs = 1
    n_zonal = len(ns["lat_bands"])
    n_regions = len(ns["regions_list"])
    n_feat = 2 * (n_zonal + n_regions + 1)

    rng = np.random.RandomState(0)
    preds = rng.normal(size=(n_obs, n_feat, 4)).astype(np.float64)
    obs = rng.normal(size=(n_obs, n_feat, 4)).astype(np.float64)

    var_w = {"PCP": 0.25, "TLWP": 0.25, "OSR": 0.25, "OLR": 0.25}
    zrg_w = {"zonal": 1 / 3, "regional": 1 / 3, "global": 1 / 3}

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
