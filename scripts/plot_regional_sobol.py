"""
Distributions of the regional Sobol results (scripts/regional_sobol.py) over the
boxes, for all boxes and for three subsets near the minimum:

  lowest Student-t  the 5% of boxes whose centre has the lowest nominal Student-t cost
  lowest PCA        the 5% with the lowest PCA likelihood cost
  near optima       the same number of boxes whose centres lie closest to any of the
                    given optima (distance in units of each parameter's analysed range)

Figure: rows = total-order index ST and absolute total effect sqrt(VT) (PPE sd),
columns = variables; per parameter the median (marker), 5-95% range (bar) and maximum (tick) of
each subset. Parameters are ordered by the all-box mean ST over the variables.
Also prints a table of the medians. Needs only numpy/matplotlib (no surrogate).

    python scripts/plot_regional_sobol.py \\
        --npz .../regional_sobol_n85/regional_sobol.npz \\
        --optima-csv .../ne256_optima_n85_all.csv \\
        --optima nocalib E1_noglobal E2_obsglobal E3_zeroglobal \\
        --scalers .../preprocess_output_ne128_prod_v2/scalers.pkl \\
        --out docs/regional_sobol_n85_3level.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))   # scalers.pkl pickles autotune_gp classes

COLORS = {"all boxes": "#8a8f98", "lowest Student-t": "#2a6fdb",
          "lowest PCA": "#d9480f", "near optima": "#2b8a3e"}


def subsets(d, frac, optima_n):
    n = len(d["done"])
    k = max(1, int(round(frac * n)))
    out = {"all boxes": np.arange(n),
           "lowest Student-t": np.argsort(d["cost_student_t"])[:k]}
    if np.isfinite(d["cost_pca"]).all():
        out["lowest PCA"] = np.argsort(d["cost_pca"])[:k]
    dist = None
    if optima_n is not None:
        rng = np.maximum(d["hi"] - d["lo"], 1e-12)
        c = (d["centre"] - d["lo"]) / rng
        o = (optima_n - d["lo"]) / rng
        dist = np.sqrt(((c[:, None, :] - o[None]) ** 2).mean(axis=2)).min(axis=1)   # RMS over params
        out["near optima"] = np.argsort(dist)[:k]
    return out, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--optima-csv", default=None, help="CSV, rows = parameters, columns = optima (physical)")
    ap.add_argument("--optima", nargs="+", default=None, help="Columns of --optima-csv to use")
    ap.add_argument("--scalers", default=None, help="scalers.pkl with X_pipeline (physical -> normalized)")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--pct", type=float, nargs=2, default=[5, 95], help="Percentile range of the bars")
    ap.add_argument("--index-only", action="store_true", help="Plot only the ST row (no absolute effect)")
    ap.add_argument("--all-only", action="store_true", help="Plot only the all-boxes distribution")
    ap.add_argument("--order-npz", default=None,
                    help="Order parameters by this regional_sobol.npz's all-box mean ST (e.g. the "
                         "3-level one, so plots of different surrogates line up)")
    ap.add_argument("--title", default="Regional Sobol, 3-level surrogate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = dict(np.load(args.npz, allow_pickle=True))
    params = [str(p) for p in d["params"]]
    var_names = [str(v) for v in d["variables"]]
    assert d["done"].all(), f"only {d['done'].sum()} of {len(d['done'])} boxes done"

    optima_n = None
    if args.optima_csv:
        import pandas as pd
        df = pd.read_csv(args.optima_csv, index_col=0)
        cols = args.optima or list(df.columns)
        sc = pickle.load(open(args.scalers, "rb"))
        optima_n = sc["X_pipeline"].transform(df.loc[params, cols].T.to_numpy(float))
    sel, dist = subsets(d, args.frac, optima_n)
    if args.all_only:
        sel = {"all boxes": sel["all boxes"]}
        dist = None
    if dist is not None:
        k = len(sel["near optima"])
        print(f"near-optima boxes: RMS distance (fraction of range) {dist[sel['near optima']].min():.3f}"
              f"-{dist[sel['near optima']].max():.3f}; all boxes median {np.median(dist):.3f}")
    for name in ("lowest Student-t", "lowest PCA"):
        if name in sel and dist is not None:
            print(f"{name}: median distance to nearest optimum {np.median(dist[sel[name]]):.3f}")

    ST, EF = d["ST"], d["effect"]                       # (boxes, V, d)
    ref = np.load(args.order_npz)["ST"] if args.order_npz else ST
    order = np.argsort(-np.nanmean(ref, axis=(0, 1)))
    rows = [(ST, "total-order index ST")]
    if not args.index_only:
        rows.append((EF, "absolute total effect √VT (PPE sd)"))
    fig, axes = plt.subplots(len(rows), len(var_names), figsize=(4.1 * len(var_names), 6 * len(rows)),
                             sharey=True, squeeze=False)
    y = np.arange(len(params))
    nsel = len(sel)
    off = (np.arange(nsel) - (nsel - 1) / 2) * 0.18
    for row, (A, lab) in enumerate(rows):
        for vi, v in enumerate(var_names):
            ax = axes[row, vi]
            for j, (name, idx) in enumerate(sel.items()):
                a = A[idx][:, vi][:, order]
                q1, med, q3 = np.percentile(a, [args.pct[0], 50, args.pct[1]], axis=0)
                mx = a.max(axis=0)
                ax.hlines(y + off[j], q1, q3, color=COLORS[name], lw=2, alpha=0.85)
                ax.plot(mx, y + off[j], "|", ms=6, mew=1.4, color=COLORS[name])
                if name == "all boxes":
                    for yy, m_ in zip(y + off[j], mx):
                        ax.text(m_, yy, f" {m_:.2f}", va="center", ha="left", fontsize=6.5,
                                color="#4a4f57")
                ax.plot(med, y + off[j], "o", ms=4.5, color=COLORS[name], mec="white", mew=0.6,
                        label=f"{name} (n={len(idx)})" if (row == 0 and vi == 0) else None)
            ax.set_title(v if row == 0 else "", fontsize=11)
            ax.set_xlabel(lab, fontsize=9)
            ax.grid(axis="x", color="#e3e5e8", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_xlim(0, 1.16 * np.nanmax(A[:, vi]))
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels([params[i] for i in order])
    axes[0, 0].invert_yaxis()
    for ax in axes[:, 0]:
        ax.tick_params(axis="y", labelsize=8.5, labelleft=True)
    if nsel > 1:
        fig.legend(loc="upper center", ncol=nsel, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(args.title, y=0.995, fontsize=13)
    fig.text(0.01, 0.005, f"{len(ST)} boxes, each 1/{int(d['n_bins'])} of every parameter's range; "
             f"marker = median over boxes, bar = {args.pct[0]:g}-{args.pct[1]:g}%, tick = maximum "
             "(number = maximum over all boxes)."
             + ("" if nsel == 1 else " Subsets: lowest 5% of box-centre cost, "
                "or the same number of box centres nearest an optimum."), fontsize=8, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.03 if len(rows) == 1 else 0.015, 1, 0.96 if nsel > 1 else 0.98))
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")

    # median table
    for A, lab in [(ST, "median ST"), (EF, "median effect")][:len(rows)]:
        print(f"\n{lab}: " + "  ".join(f"{v:>22s}" for v in var_names))
        print(" " * 33 + "  ".join(" ".join(f"{n.split()[-1][:5]:>5s}" for n in sel) for _ in var_names))
        for p in order:
            cells = []
            for vi in range(len(var_names)):
                cells.append(" ".join(f"{np.median(A[idx, vi, p]):5.3f}" for idx in sel.values()))
            print(f"{params[p]:33s}" + "  ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
