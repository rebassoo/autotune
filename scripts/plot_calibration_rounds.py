"""
Best-fit parameters and their Hessian standard errors after every round of a
model-error calibration (scripts/calibrate_model_error.py), one panel per
experiment, so you can see how the optimum and its curvature move as sigma_delta
is re-estimated.

Each round's values come from <round dir>/diagnostics/parameter_uncertainty_seed*.pkl,
written by run_two_stage.py at that round's best start. x is the normalized
parameter (0-1 over the scaler range; with bounds_from_data the search box
is narrower, so "at a bound" can sit inside 0-1). Error bars are +-1 stderr from the inverse
Hessian, clipped to the box; parameters within 0.001 of a bound have no stderr
and are drawn hollow. The cost is a per-block weighted MEAN, so these stderrs
are relative indicators, not calibrated intervals (see run_two_stage's note).

    python scripts/plot_calibration_rounds.py \\
        --calib E1=/pscratch/sd/r/rebassoo/autotune/calib_n72iso_E1_noglobal \\
        --calib E2=/pscratch/sd/r/rebassoo/autotune/calib_n72iso_E2_obsglobal \\
        --calib E3=/pscratch/sd/r/rebassoo/autotune/calib_n72iso_E3_zeroglobal \\
        --out docs/calibration_rounds_n72iso.png
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, GREY = "#1a1a1a", "#9aa0a6"


def _round_dir(calib_dir: Path, rd: dict) -> Path:
    """The run dir logged for a round; falls back to <calib>/round_K if it was moved."""
    p = Path(rd["run_dir"])
    if p.exists():
        return p
    local = calib_dir / f"round_{rd['round']}"
    if local.exists():
        return local
    raise SystemExit(f"Cannot find the run dir of round {rd['round']} ({p})")


def _load(calib_dir: Path, aliases):
    L = json.load(open(calib_dir / "calibration_log.json"))
    out = []
    for rd in L["rounds"]:
        try:
            d = _round_dir(calib_dir, rd)
        except SystemExit:
            run_dir = rd["run_dir"]
            for a, b in aliases:
                run_dir = run_dir.replace(a, b)
            d = Path(run_dir)
        f = sorted(glob.glob(str(d / "diagnostics" / "parameter_uncertainty_seed*.pkl")))
        if not f:
            print(f"  no uncertainty pickle for round {rd['round']} in {d}; skipped")
            continue
        u = pickle.load(open(f[0], "rb"))
        out.append(dict(round=rd["round"], cost=rd["cost"], x=np.asarray(u["x"]),
                        se=np.asarray(u["stderr"]), at_bound=np.asarray(u["at_bound"], bool),
                        names=u["param_names"]))
    return out, L.get("converged")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", action="append", required=True, help="LABEL=calibration dir")
    ap.add_argument("--alias", action="append", default=[],
                    help="OLD=NEW substring replacement for run dirs that were renamed")
    ap.add_argument("--no-errorbars", action="store_true",
                    help="Points only, joined across rounds. The Hessian stderrs of the "
                         "weighted-mean cost are inflated (see above), often to the whole box.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    aliases = [a.split("=", 1) for a in args.alias]

    runs = []
    for c in args.calib:
        label, path = c.split("=", 1)
        rounds, conv = _load(Path(path), aliases)
        runs.append((label, rounds, conv))

    names = runs[0][1][0]["names"]
    n = len(names)
    y = np.arange(n)[::-1].astype(float)
    n_max = max(len(r) for _, r, _ in runs)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.35 + 0.65 * k / max(n_max - 1, 1)) for k in range(n_max)]

    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 0.42 * n + 2.2),
                             sharey=True, squeeze=False)
    for ax, (label, rounds, conv) in zip(axes[0], runs):
        k_n = len(rounds)
        off = np.linspace(-0.3, 0.3, k_n) if k_n > 1 else [0.0]
        if args.no_errorbars and k_n > 1:
            X = np.array([r["x"] for r in rounds])            # (rounds, params)
            for i in range(n):
                ax.plot(X[:, i], y[i] + np.asarray(off), color=GREY, lw=0.8, zorder=2)
        for k, r in enumerate(rounds):
            lo = np.clip(r["x"] - np.nan_to_num(r["se"]), 0, 1)
            hi = np.clip(r["x"] + np.nan_to_num(r["se"]), 0, 1)
            yy = y + off[k]
            if not args.no_errorbars:
                ax.hlines(yy, lo, hi, color=colors[k], lw=2)
            ok = ~r["at_bound"]
            ax.scatter(r["x"][ok], yy[ok], s=28, color=colors[k], edgecolor="white", lw=0.6,
                       zorder=3, label=f"round {r['round']} (cost {r['cost']:.3f})")
            ax.scatter(r["x"][~ok], yy[~ok], s=28, facecolor="white", edgecolor=colors[k],
                       lw=1.4, zorder=3)
        for yi in y[1::2]:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f3f3f3", zorder=0, lw=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel("normalized parameter")
        ax.set_title(f"{label}  ({'converged' if conv else 'not converged'})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), fontsize=8, frameon=False,
                  ncol=2)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels(names, fontsize=9)
    default_title = ("Best fit after each calibration round (hollow = at a bound)"
                     if args.no_errorbars else
                     "Best fit ±1 Hessian stderr after each calibration round "
                     "(hollow = at a bound, no stderr)")
    fig.suptitle(args.title or default_title, y=0.995)
    fig.text(0.01, 0.003, "x: 0-1 over the scaler range; the search box is narrower for some "
             "parameters, so a hollow 'at a bound' point can sit inside 0-1.",
             fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
