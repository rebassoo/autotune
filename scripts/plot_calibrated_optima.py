"""
Best-fit parameters without model-error calibration and after each calibration
experiment, on one axis, so the effect of the global-error treatment is direct.

The uncalibrated point is round 0 of --baseline (E2 and E3 share it: both start
from the plain optimization; E1's round 0 drops the global term, so it is not a
baseline); each experiment contributes its final round. x is the normalized parameter (0-1 over
the scaler range). Hollow markers are within 0.001 of a search-box bound.

    python scripts/plot_calibrated_optima.py \\
        --baseline /pscratch/sd/r/rebassoo/autotune/calib_n85_E3_zeroglobal \\
        --calib "E1 global term removed=/pscratch/sd/r/rebassoo/autotune/calib_n85_E1_noglobal" \\
        --calib "E2 observational global error=/pscratch/sd/r/rebassoo/autotune/calib_n85_E2_obsglobal" \\
        --calib "E3 zero global error=/pscratch/sd/r/rebassoo/autotune/calib_n85_E3_zeroglobal" \\
        --out docs/calibrated_optima_n85.png
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, GREY = "#1a1a1a", "#9aa0a6"
COLORS = ["#2a6fdb", "#e8833a", "#8e44ad", "#1b9e77"]
MARKERS = ["o", "s", "D", "^"]


def _point(calib_dir: Path, which: str, aliases=()):
    """theta, names and at-bound flags of round 0 ('first') or the final round ('last')."""
    L = json.load(open(calib_dir / "calibration_log.json"))
    rd = L["rounds"][0 if which == "first" else -1]
    # the uncertainty pickle holds the names and the at-bound flags; look in the
    # round dir as logged, then in <calib>/round_K (the logged dir may have moved)
    run_dir = rd["run_dir"]
    for a, b in aliases:
        run_dir = run_dir.replace(a, b)
    cands = [Path(run_dir), calib_dir / f"round_{rd['round']}"]
    for d in cands:
        f = sorted(glob.glob(str(d / "diagnostics" / "parameter_uncertainty_seed*.pkl")))
        if f:
            u = pickle.load(open(f[0], "rb"))
            return np.asarray(rd["theta"]), u["param_names"], np.asarray(u["at_bound"], bool)
    raise SystemExit(f"No parameter_uncertainty pickle for round {rd['round']} of {calib_dir}")


def _box_from_log(path, names):
    """Per-parameter search box as logged by run_two_stage (bounds_from_data)."""
    box = {}
    for line in open(path):
        m = re.search(r"^\S*\s+(\w+)\s+\[([-\d.]+), ([-\d.]+)\]\s+\(", line.split("]", 1)[-1])
        if m and m.group(1) in names:
            box[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    missing = [n for n in names if n not in box]
    if missing:
        raise SystemExit(f"{path}: no logged search box for {missing}")
    return np.array([box[n] for n in names])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True,
                    help="Calibration dir whose round 0 is the uncalibrated optimum.")
    ap.add_argument("--calib", action="append", required=True, help="LABEL=calibration dir")
    ap.add_argument("--alias", action="append", default=[],
                    help="OLD=NEW substring replacement for logged run dirs that were renamed")
    ap.add_argument("--bounds-log", default=None,
                    help="An optimize log with the 'Optimizer bounds restricted' table; "
                         "each parameter's search box is shaded.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    aliases = [a.split("=", 1) for a in args.alias]
    x0, names, b0 = _point(Path(args.baseline), "first", aliases)
    runs = [("no calibration", x0, b0, INK, "o")]
    for k, c in enumerate(args.calib):
        label, path = c.split("=", 1)
        x, _, b = _point(Path(path), "last", aliases)
        runs.append((label, x, b, COLORS[k], MARKERS[k + 1]))

    n = len(names)
    y = np.arange(n)[::-1].astype(float)
    off = np.linspace(-0.27, 0.27, len(runs))
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * n + 2.0))
    if args.bounds_log:
        box = _box_from_log(args.bounds_log, names)
        for i in range(n):
            ax.fill_betweenx([y[i] - 0.42, y[i] + 0.42], box[i, 0], box[i, 1],
                             color="#dbe7f7", lw=0, zorder=0,
                             label="search box (sampled range)" if i == 0 else None)
    else:
        for yi in y[1::2]:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f3f3f3", zorder=0, lw=0)
    for i in range(n):          # thin line from the uncalibrated point to each calibrated one
        for k in range(1, len(runs)):
            ax.plot([x0[i], runs[k][1][i]], [y[i] + off[0], y[i] + off[k]],
                    color=GREY, lw=0.6, zorder=1)
    for k, (label, x, b, col, mk) in enumerate(runs):
        yy = y + off[k]
        ax.scatter(x[~b], yy[~b], s=34, marker=mk, color=col, edgecolor="white", lw=0.6,
                   zorder=3, label=label)
        ax.scatter(x[b], yy[b], s=34, marker=mk, facecolor="white", edgecolor=col, lw=1.4,
                   zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("normalized parameter")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9, frameon=False)
    ax.set_title(args.title or "Best fit without and with model-error calibration", pad=66)
    note = ("Hollow = at a search-box bound. x: 0-1 over the scaler range; "
            + ("shading = the range the optimizer searches (sampled by the PPE)."
               if args.bounds_log else "the search box is narrower for some parameters."))
    fig.text(0.01, 0.003, note, fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
