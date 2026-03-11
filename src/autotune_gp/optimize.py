from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import basinhopping
from pathlib import Path
import csv
from datetime import datetime

def optimize_parallel(cost_fn, n_params, bounds_low, bounds_high, seed, n_xstarts, niter,
                      method, out_dir, max_workers=None):
    rn = np.random.RandomState(seed)
    xstarts = rn.rand(n_xstarts, n_params)

    bounds = [(bounds_low, bounds_high)] * n_params
    minimizer_kwargs = {"method": method, "bounds": bounds}

    def run_one(x0):
        res = basinhopping(cost_fn, x0, minimizer_kwargs=minimizer_kwargs, niter=niter,
                           seed=seed)
        return np.hstack((res.x, res.fun))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(run_one, xstarts))
    results = np.vstack(results)

    top_rows = np.argsort(np.abs(results[:, -1]))[:10]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = out_dir / ("results_%d_%d_%s.csv" % (n_xstarts, seed, date_str))

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "params", "cost"])
        for rank, ridx in enumerate(top_rows, 1):
            w.writerow([rank, results[ridx][:-1].tolist(), float(results[ridx, -1])])

    return results, top_rows, str(csv_path)
