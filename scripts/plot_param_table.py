"""
Table (PNG and CSV) of the 19 parameter values at the 4-variable optimum, the
10-variable best optimum and the 10-variable second optimum (cost ranks 5-32), with
the changes between them in normalized units (fraction of the declared range;
log-scaled parameters are normalized in log space, as the optimizer sees them).

    python scripts/plot_param_table.py --out docs/param_table_4var_10var.png
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

S = Path("/pscratch/sd/r/rebassoo/autotune")
P = Path("/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2")
R4, R10 = S / "results_mf3_ne256_n85_studentt_dof4", S / "results_mf3_ne256_n85_10var"
CONFIG = ROOT / "configs/perlmutter_mf3_ne256_n85_10var_annual.yaml"
LOG = ROOT / "logs/stage2b_mf3_ne256_n85_10var_60764492.log"
INK, MUTED, RULE = "#1a1a1a", "#6b6f76", "#d0d4da"
DIV = LinearSegmentedColormap.from_list("div", ["#2a6fdb", "#f1f2f3", "#d9542b"])
TOL = 0.005          # within this of a sampled bound counts as "at the bound"


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1e5 or a < 1e-3:
        return f"{x:.2e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    return f"{x:.3g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=5, help="Cost rank standing in for the second optimum.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = lambda p: pickle.load(open(p, "rb"))
    d = L(P / "preprocess_output_ne256_n85_10var" / "default_params.pkl")
    names = list(d["names"])
    assert names == list(yaml.safe_load(open(CONFIG))["optimize"]["param_physical_bounds"])
    log_params = set(yaml.safe_load(open(CONFIG))["optimize"]["log_params"])
    sc = L(P / "preprocess_output_ne128_prod_v2_10var" / "scalers.pkl")["X_pipeline"]
    phys = lambda x: sc.inverse_transform(np.asarray(x, float)[None, :])[0]
    # the scaler must reproduce the stored default, so the physical columns are trustworthy
    assert np.allclose(phys(d["norm"]), d["phys"], rtol=1e-6)

    pd = lambda r: L(r / "diagnostics" / "projection_data_seed50.pkl")
    x4, c4 = np.asarray(pd(R4)["best_params_norm"], float), float(pd(R4)["best_cost"])
    A = np.array([L(f) for f in sorted((R10 / "optimize_checkpoints").glob("start_*.pkl"))], float)
    A = A[np.argsort(A[:, -1])]
    x10, c10 = A[0, :-1], A[0, -1]
    xr, cr = A[args.rank - 1, :-1], A[args.rank - 1, -1]
    assert np.allclose(x10, pd(R10)["best_params_norm"], atol=1e-8)

    bounds = {m[0]: (float(m[1]), float(m[2])) for m in
              re.findall(r"^\[\d\d:\d\d:\d\d\]\s+(\w+)\s+\[([\d.]+), ([\d.]+)\]", LOG.read_text(), re.M)}
    assert set(bounds) == set(names)
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    mark = lambda x: ["low" if v - l < TOL else "high" if h - v < TOL else ""
                      for v, l, h in zip(x, lo, hi)]

    cols = [("Default", d["norm"]), ("4-var", x4), ("10-var best", x10), (f"10-var rank 5–32", xr)]
    dv = [("10-var best\n− 4-var", x10 - x4), ("rank 5–32\n− 4-var", xr - x4),
          ("rank 5–32\n− 10-var best", xr - x10)]

    n = len(names)
    fig, ax = plt.subplots(figsize=(16, 9.4))
    ax.set_axis_off()
    ax.set_xlim(0, 16)
    ax.set_ylim(n + 3.3, -1.6)
    xn, xs, wc, xd, wd = 0.1, 3.75, 1.72, 10.75, 1.72
    ax.text(xs + 2 * wc, -0.82, "Physical value (normalized value in brackets)", ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(xd + 1.5 * wd, -0.82, "Change in normalized units", ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(xn, 0, "Parameter", va="center", fontsize=10.5, weight="bold", color=INK)
    for k, (t, _) in enumerate(cols):
        ax.text(xs + (k + 0.5) * wc, 0, t, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
    for k, (t, _) in enumerate(dv):
        ax.text(xd + (k + 0.5) * wd, 0, t, ha="center", va="center", fontsize=10, weight="bold", color=INK,
                linespacing=1.1)
    ax.plot([xn, xd + 3 * wd], [0.62, 0.62], color=INK, lw=1)

    for i, nm in enumerate(names):
        y = i + 1.15
        ax.text(xn, y, nm + (" (log)" if nm in log_params else ""), va="center", fontsize=10, color=INK)
        for k, (_, x) in enumerate(cols):
            mk = mark(x)[i] if k else ""
            sym = {"low": " ▼", "high": " ▲", "": ""}[mk]
            ax.text(xs + (k + 0.5) * wc, y, f"{fmt(phys(x)[i])} ({x[i]:.2f}){sym}", ha="center", va="center",
                    fontsize=9.6, color=INK, weight="bold" if mk else "normal")
        for k, (_, v) in enumerate(dv):
            ax.add_patch(plt.Rectangle((xd + k * wd + 0.04, y - 0.45), wd - 0.08, 0.9, lw=0,
                                       color=DIV(0.5 + 0.5 * np.clip(v[i] / 0.5, -1, 1))))
            ax.text(xd + (k + 0.5) * wd, y, f"{v[i]:+.2f}", ha="center", va="center", fontsize=9.8, color=INK)
        if i < n - 1:
            ax.plot([xn, xd + 3 * wd], [y + 0.5, y + 0.5], color=RULE, lw=0.5)

    y = n + 1.0
    ax.plot([xn, xd + 3 * wd], [y - 0.35, y - 0.35], color=INK, lw=1)
    ax.text(xn, y + 0.15, "Cost (each fit's own cost function)", va="center", fontsize=10, weight="bold", color=INK)
    for k, c in enumerate(["–", f"{c4:.3f}", f"{c10:.3f}", f"{cr:.3f}"]):
        ax.text(xs + (k + 0.5) * wc, y + 0.15, c, ha="center", va="center", fontsize=10, weight="bold", color=INK)
    rmsd = [np.sqrt(np.mean(v ** 2)) for _, v in dv]
    ax.text(xn, y + 1.05, "RMS change over the 19 parameters", va="center", fontsize=10, weight="bold", color=INK)
    for k, r in enumerate(rmsd):
        ax.text(xd + (k + 0.5) * wd, y + 1.05, f"{r:.2f}", ha="center", va="center", fontsize=10, weight="bold", color=INK)

    fig.suptitle("The 19 parameters at the 4-variable and 10-variable optima", x=0.012, y=0.975, ha="left",
                 fontsize=15, weight="bold", color=INK)
    fig.text(0.012, 0.012,
             "▼ ▲ bold: optimum within 0.005 of the lower / upper bound of the range the PPE sampled (a parameter can hit its "
             "bound well inside the declared range).  Normalized: 0–1 over the declared range, in log space for (log) "
             f"parameters.\nRank 5–32 is the second optimum: rank {args.rank} shown; those 28 starts agree to within 0.005. "
             "The 4-variable cost is not comparable with the 10-variable costs.  Colour: orange = rises, blue = falls, full at ±0.5.",
             fontsize=8.8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.07)
    fig.savefig(args.out, dpi=150, facecolor="white")

    csv_out = Path(args.out).with_suffix(".csv")
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "log_scaled", "sampled_low_norm", "sampled_high_norm"]
                   + [f"{t}_{u}" for t, _ in cols for u in ("phys", "norm")])
        for i, nm in enumerate(names):
            w.writerow([nm, nm in log_params, lo[i], hi[i]]
                       + [v for _, x in cols for v in (phys(x)[i], x[i])])
    print(f"costs 4var {c4:.4f} 10var {c10:.4f} rank{args.rank} {cr:.4f}; RMS changes {np.round(rmsd, 3)}")
    for t, x in cols[1:]:
        print(t, "at bounds:", [f"{nm}({m})" for nm, m in zip(names, mark(x)) if m])
    print(f"Saved {args.out} and {csv_out}")


if __name__ == "__main__":
    main()
