"""
Score a twin (perfect-model) experiment: how close the optimizer got to the parameters
that produced the target (truth.pkl from make_twin_experiment.py), per parameter and
per distinct optimum, and how well the surrogate's prediction at the optimum matches
the target. Reads results only (no GP), so it runs on a login node.

    python scripts/score_twin.py --twin-dir .../twin_preprocess/m060_3level \\
        --results-dir .../results_twin_m060_3level [--ppe .../gp_proj.pkl]

Parameter errors are in normalized units (0-1 over the scaler range). As a scale, the
script also prints the typical distance from the truth to the other PPE members.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

MMD = 86400e3


def clusters(rows, tol=1e-3):
    """Distinct optima: (params, cost) rows grouped by cost rounded to tol, lowest first."""
    out = {}
    for r in sorted(rows, key=lambda r: r[-1]):
        out.setdefault(round(r[-1] / tol), []).append(r)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--ppe", help="ne256 gp_proj.pkl, for the member-distance scale")
    ap.add_argument("--n-optima", type=int, default=3)
    args = ap.parse_args()
    L = lambda p: pickle.load(open(p, "rb"))
    tw, rd = Path(args.twin_dir), Path(args.results_dir)
    truth = L(tw / "truth.pkl")
    xsc = L(tw / "scalers.pkl")["X_pipeline"]
    names = truth["params"]
    xt = xsc.transform(truth["x_phys"][None, :])[0]

    # every start, from the checkpoints (the CSV keeps only 10)
    rows = [np.asarray(L(f), float) for f in sorted((rd / "optimize_checkpoints").glob("start_*.pkl"))]
    groups = clusters(rows)
    print(f"{truth['member']}  held out: {truth['held_out']}  starts: {len(rows)}")
    opt = []
    for g in groups[:args.n_optima]:
        opt.append((g[0][-1], len(g), g[0][:-1]))
    print(f"{'parameter':38}{'truth':>8}" + "".join(f"{'opt' + str(i + 1):>8}" for i in range(len(opt))))
    for j, n in enumerate(names):
        print(f"{n:38}{xt[j]:8.3f}" + "".join(f"{x[j]:8.3f}" for _, _, x in opt))
    for i, (c, k, x) in enumerate(opt):
        e = np.abs(x - xt)
        print(f"opt{i + 1}: cost {c:.4f} ({k} starts)  |error| RMS {np.sqrt(np.mean(e ** 2)):.3f}, "
              f"median {np.median(e):.3f}, max {e.max():.3f} ({names[int(np.argmax(e))]}); "
              f"within 0.1: {int(np.sum(e < 0.1))}/{len(e)}")

    if args.ppe:
        gp = L(args.ppe)
        Xn = xsc.transform(gp["X_train"].to_numpy(float))
        keep = np.asarray(gp["X_train"].index != truth["member"])
        d = np.sqrt(np.mean((Xn[keep] - xt) ** 2, axis=1))
        print(f"scale: RMS distance truth -> other ne256 members: median {np.median(d):.3f}, "
              f"min {d.min():.3f}")

    p = L(rd / "diagnostics" / "projection_data_seed50.pkl")
    if not np.isclose(float(p["best_cost"]), opt[0][0]):
        print(f"NOTE: diagnostics are for the start with cost {float(p['best_cost']):.4f}, not the best "
              f"({opt[0][0]:.4f}); rerun them before trusting the prediction check below.")
    y, obs = np.asarray(p["hf_optimal"], float), np.asarray(p["obs"], float)
    tgt = np.asarray(truth["y"], float)
    if not np.allclose(obs, tgt[:len(obs)], rtol=1e-6, atol=0):
        print("WARNING: projection obs differ from the truth member's features")
    print("prediction at the diagnostics point vs target (zonal-band RMSE; global error):")
    for j, v in enumerate(p["var_names"]):
        f = MMD if v == "PCP" else 1.0
        print(f"  {v:5} RMSE {f * np.sqrt(np.mean((y[:15, j] - obs[:15, j]) ** 2)):.4f}  "
              f"global {f * (y[-1, j] - obs[-1, j]):+.4f}")


if __name__ == "__main__":
    main()
