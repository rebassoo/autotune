"""
Learning curve of the 3-level surrogate's ne256 holdout skill: R^2 (normalized
units, pooled over the 85 held-out predictions) per variable against the number of
ne256 runs used in training per fold (jobs/holdout_learning_curve.sh).

  N = 0   the 2-level (ne128+ne32) GP, which never sees ne256 data (rows_2level of
          the full run's nfidelity_holdout.pkl)
  N = 68  the full run (all of each fold's training members)
  others  <lc-dir>/train{N}_seed{s}/nfidelity_holdout.pkl; the line goes through the
          mean over subsets, each subset is a small dot

    python scripts/plot_holdout_learning_curve.py \\
        --full .../results_ne256_n85_3level_holdout_ISO/nfidelity_holdout.pkl \\
        --lc-dir .../holdout_lc_n85 --out docs/holdout_learning_curve_n85.png
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

COLORS = {"PCP": "#2a6fdb", "OSR": "#d9480f", "OLR": "#2b8a3e", "TLWP": "#7048e8"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True)
    ap.add_argument("--lc-dir", required=True)
    ap.add_argument("--alone", action="store_true",
                    help="ne256-alone surrogate (evaluate_ne256_alone_holdout.py results): --full is "
                         "its ne256_alone_holdout.pkl, no N=0 point")
    ap.add_argument("--mean-only", action="store_true", help="Plot only the mean over subsets (no dots)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = pickle.load(open(args.full, "rb"))
    n_full = 68
    key, fname = ("rows", "ne256_alone_holdout.pkl") if args.alone else ("rows_mf", "nfidelity_holdout.pkl")
    pts = {}                                       # N -> list of {var: r2}
    if not args.alone:
        pts[0] = [{r[0]: r[1] for r in full["rows_2level"]}]
    pts[n_full] = [{r[0]: r[1] for r in full[key]}]
    for p in sorted(Path(args.lc_dir).glob(f"train*_seed*/{fname}")):
        m = re.match(r"train(\d+)_seed(\d+)", p.parent.name)
        d = pickle.load(open(p, "rb"))
        pts.setdefault(int(m.group(1)), []).append({r[0]: r[1] for r in d[key]})
    Ns = sorted(pts)
    var_names = [r[0] for r in full[key]]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for v in var_names:
        mean = [np.mean([s[v] for s in pts[n]]) for n in Ns]
        ax.plot(Ns, mean, "-o", color=COLORS.get(v), lw=2, ms=6, mec="white", mew=0.8, label=v)
        for n in Ns:
            if len(pts[n]) > 1 and not args.mean_only:
                ax.plot([n] * len(pts[n]), [s[v] for s in pts[n]], ".", color=COLORS.get(v),
                        ms=4, alpha=0.6)
    ax.set_xlabel("ne256 runs used in training (per fold)")
    ax.set_ylabel("holdout R² at ne256 (normalized units)")
    ax.set_xticks(Ns)
    ax.set_xlim(-3 if not args.alone else min(Ns) - 3, n_full + 3)
    if args.alone:
        ax.axhline(0, color="#9aa0a6", lw=0.8)
    ax.grid(color="#e3e5e8", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    cs = {len(pts[n]) for n in Ns if 0 < n < n_full}
    counts = (f"{cs.pop()} per N" if len(cs) == 1
              else ", ".join(f"N={n}: {len(pts[n])}" for n in Ns if 0 < n < n_full))
    ax.set_title(args.title or ("ne256-alone surrogate" if args.alone else "3-level surrogate")
                 + ": ne256 holdout skill vs. ne256 training runs", fontsize=12)
    fig.text(0.01, 0.01, "5 folds over the 85 ne256 members, always scored on the same 85 held-out "
             "predictions. " + ("Single-fidelity GP on ne256 runs only; " if args.alone else
             "N=0: 2-level (ne128+ne32) GP; ") + "N=68: full run.\nOther N: random subsets "
             f"of each fold's training members, each a full 5-fold run; line = mean over the "
             f"random subsets ({counts})" + ("." if args.mean_only else ", dots = single subsets."), fontsize=7.5, color="#6b6f76")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")
    for n in Ns:
        print(f"N={n:3d} ({len(pts[n])} subset(s)): " +
              "  ".join(f"{v} {np.mean([s[v] for s in pts[n]]):.3f}" for v in var_names))


if __name__ == "__main__":
    main()
