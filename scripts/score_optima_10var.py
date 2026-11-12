"""
Score several parameter settings on all 10 variables with the merged 10-variable
3-level GP: RMS misfit to obs (zonal, regional) and the global-mean error, in
physical units, plus the surrogate's own sd. Writes a CSV and prints a table.

    python scripts/score_optima_10var.py \\
        --gp-pkl .../results_mf3_ne256_n85_10var/mf_gp_trained.pkl \\
        --scalers-dir .../mf_prod_v2/preprocess_output_ne128_prod_v2_10var \\
        --ref .../results_mf3_ne256_n85_10var \\
        --opt 10var=.../results_mf3_ne256_n85_10var \\
        --opt 4var=.../results_mf3_ne256_n85_studentt_dof4 \\
        --out-csv docs/score_optima_10var.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

UNIT = {"PCP": (86400e3, "mm/d"), "SLP": (0.01, "hPa")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp-pkl", required=True)
    ap.add_argument("--scalers-dir", required=True)
    ap.add_argument("--ref", required=True, help="Results dir whose projection_data gives obs, default.")
    ap.add_argument("--opt", action="append", default=[], help="NAME=RESULTS_DIR (repeatable).")
    ap.add_argument("--opt-rank", action="append", default=[],
                    help="NAME=RESULTS_DIR:RANK, the start of that cost rank (1 = best) from "
                         "optimize_checkpoints (repeatable).")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    L = lambda p: pickle.load(open(p, "rb"))
    pd = lambda d: L(Path(d) / "diagnostics" / "projection_data_seed50.pkl")
    ref = pd(args.ref)
    names = [str(v) for v in ref["var_names"]]
    obs = np.asarray(ref["obs"], float)                        # (22, nv) physical
    sc = L(Path(args.scalers_dir) / "scalers.pkl")
    gp = L(args.gp_pkl)

    def predict(x):
        m, v = gp.predict(np.asarray(x, float)[None, :])
        m, v = np.asarray(m)[0], np.asarray(v)[0]              # (22, nv), normalized
        mu = np.zeros_like(m)
        sd = np.zeros_like(m)
        for j, n in enumerate(names):
            p = sc[f"Y_pipeline_{n}"]
            mu[:, j] = p.inverse_transform(m[None, :, j])[0]
            hi = p.inverse_transform((m[:, j] + np.sqrt(np.maximum(v[:, j], 0)))[None, :])[0]
            sd[:, j] = np.abs(hi - mu[:, j])
        return mu, sd

    settings = []
    for s in args.opt:
        nm, d = s.split("=", 1)
        settings.append((nm, np.asarray(pd(d)["best_params_norm"], float)))
    for s in args.opt_rank:
        nm, rest = s.split("=", 1)
        d, rank = rest.rsplit(":", 1)
        ck = sorted((Path(d) / "optimize_checkpoints").glob("start_*.pkl"))
        A = np.array([L(f) for f in ck], float)
        row = A[np.argsort(A[:, -1])[int(rank) - 1]]
        print(f"{nm}: rank {rank} of {len(A)}, cost {row[-1]:.6f}")
        settings.append((nm, row[:-1]))
    # check: the 10-var optimum's prediction must match what the optimizer saved
    mu_ref, _ = predict(np.asarray(ref["best_params_norm"], float))
    err = np.max(np.abs(mu_ref - np.asarray(ref["hf_optimal"], float)) / (np.abs(obs) + 1e-30))
    print(f"check vs saved hf_optimal: max rel diff {err:.2e}")

    rows = []
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    preds = {"default": (np.asarray(ref["hf_default"], float), None)}
    for nm, x in settings:
        preds[nm] = predict(x)
    for j, v in enumerate(names):
        f, u = UNIT.get(v, (1.0, ""))
        for nm, (mu, sd) in preds.items():
            e = (mu[:, j] - obs[:, j]) * f
            rows.append(dict(var=v, unit=u, setting=nm, rms_zonal=rms(e[:15]), rms_region=rms(e[15:21]),
                             global_err=float(e[21]),
                             mean_sd=float(np.mean(sd[:, j]) * f) if sd is not None else float("nan")))
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'var':9}{'unit':6}{'setting':9}{'zonal':>9}{'region':>9}{'global':>10}{'GP sd':>9}")
    for r in rows:
        print(f"{r['var']:9}{r['unit']:6}{r['setting']:9}{r['rms_zonal']:9.3g}{r['rms_region']:9.3g}"
              f"{r['global_err']:+10.3g}{r['mean_sd']:9.2g}")
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
