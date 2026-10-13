"""
Compare best-fit parameters across optimization runs, with error bars where a
run has a meaningful parameter-uncertainty report.

All 19 parameters are plotted in the *normalised* [0, 1] space the optimizer
works in (MinMaxScaler over optimize.param_physical_bounds). That is the only
way to put them on one axis -- physical values span ~1e-5 to ~1e7 -- and it is
also the space the inverse-Hessian stderr is computed in, so values and error
bars are directly comparable. Physical values are written to an accompanying
CSV.

Error bars are +/-1 sigma from diagnostics/parameter_uncertainty_*.pkl. A run
has none when its cost is not a real log-likelihood (likelihood: mae), in
which case run_two_stage never produces that file and the run is plotted as
points only.

Within a run that does have the report, individual parameters can still lack a
stderr: autotune_gp.uncertainty flags parameters sitting within boundary_tol of
a bound, drops them from the free set, and leaves their stderr NaN by
construction. Those are drawn as open markers -- pinned at the edge of the
PPE's sampled range, not merely uncertain.

Usage:
    python scripts/plot_best_params_comparison.py \
        --preprocess-dir /path/to/preprocess_output_ne128_prod_v2 \
        --run "MF Laplace=/path/to/results_mf_ne128_ne32_prod_v2" \
        --run "MF Student-t dof=4=/path/to/results_..._studentt_dof4_40starts" \
        --run "SF ESEm MAE=/path/to/results_ne128_prod_v2_sf_esem_mae_40starts" \
        --out-dir docs
"""
from __future__ import annotations

import argparse
import csv
import glob
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _parse_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Expected 'Label=/path/to/results_dir', got: {spec!r}")
    # rsplit: labels legitimately contain '=' (e.g. "MF Student-t dof=4").
    label, path = spec.rsplit("=", 1)
    return label.strip(), path.strip()


def _load_run(results_dir: str) -> dict:
    """Best-fit params (normalised), plus stderr/at_bound when available."""
    d = Path(results_dir)
    projs = sorted(glob.glob(str(d / "diagnostics" / "projection_data*.pkl")))
    if not projs:
        raise FileNotFoundError(f"No projection_data*.pkl in {d}/diagnostics")
    with open(projs[-1], "rb") as f:
        proj = pickle.load(f)
    out = {
        "x": np.asarray(proj["best_params_norm"], dtype=float),
        "stderr": None,
        "at_bound": None,
        "param_names": None,
    }
    uncs = sorted(glob.glob(str(d / "diagnostics" / "parameter_uncertainty*.pkl")))
    if uncs:
        with open(uncs[-1], "rb") as f:
            u = pickle.load(f)
        # The uncertainty report is computed at the optimizer's best x, the same
        # vector projection_data stores -- assert rather than assume.
        if not np.allclose(np.asarray(u["x"], dtype=float), out["x"]):
            raise SystemExit(
                f"{d.name}: parameter_uncertainty['x'] disagrees with "
                "projection_data['best_params_norm']; refusing to pair them."
            )
        out["stderr"] = np.asarray(u["stderr"], dtype=float)
        out["at_bound"] = np.asarray(u["at_bound"], dtype=bool)
        out["param_names"] = list(u["param_names"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess-dir", required=True,
                    help="Holds scalers.pkl, used to report physical values.")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--bounds-dir", action="append", default=[], metavar="PREPROCESS_DIR",
                    help="Preprocess dir(s) whose gp_proj.pkl X_train defines the range the "
                         "PPE actually sampled. Pass HF and LF for a multi-fidelity run, "
                         "mirroring run_two_stage._resolve_opt_bounds. With "
                         "optimize.bounds_from_data=true this, not param_physical_bounds, "
                         "is what the optimizer was allowed to search -- so it is what "
                         "'at bound' actually refers to.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="best_params_comparison")
    args = ap.parse_args()

    specs = [_parse_spec(s) for s in args.run]
    runs = [(lbl, _load_run(d)) for lbl, d in specs]

    names = next((r["param_names"] for _, r in runs if r["param_names"]), None)
    if names is None:
        raise SystemExit("No parameter_uncertainty pickle found in any run; "
                         "cannot recover parameter names.")
    n = len(names)
    for lbl, r in runs:
        if len(r["x"]) != n:
            raise SystemExit(f"{lbl}: has {len(r['x'])} params, expected {n}")

    with open(Path(args.preprocess_dir) / "scalers.pkl", "rb") as f:
        x_sc = pickle.load(f)["X_pipeline"]

    sampled = None
    if args.bounds_dir:
        stacked = []
        for bd in args.bounds_dir:
            with open(Path(bd) / "gp_proj.pkl", "rb") as f:
                stacked.append(np.asarray(x_sc.transform(pickle.load(f)["X_train"]), dtype=float))
        allx = np.vstack(stacked)
        sampled = (allx.min(axis=0), allx.max(axis=0))
        narrowed = int(np.sum((sampled[1] - sampled[0]) < 0.99))
        print(f"Sampled range from {len(args.bounds_dir)} preprocess dir(s); "
              f"{narrowed}/{n} parameter(s) narrower than the declared bounds")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV in physical units -------------------------------------------
    csv_path = out_dir / f"{args.name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["parameter"]
        for lbl, _ in runs:
            head += [f"{lbl} (norm)", f"{lbl} stderr(norm)", f"{lbl} (physical)", f"{lbl} at_bound"]
        w.writerow(head)
        phys = {lbl: x_sc.inverse_transform(r["x"].reshape(1, -1))[0] for lbl, r in runs}
        for i, nm in enumerate(names):
            row = [nm]
            for lbl, r in runs:
                se = "" if r["stderr"] is None else (
                    "" if np.isnan(r["stderr"][i]) else f"{r['stderr'][i]:.6g}")
                ab = "" if r["at_bound"] is None else ("yes" if r["at_bound"][i] else "no")
                row += [f"{r['x'][i]:.6f}", se, f"{phys[lbl][i]:.6g}", ab]
            w.writerow(row)
    print(f"Saved {csv_path}")

    # ---- Plot -------------------------------------------------------------
    y = np.arange(n)[::-1] * 1.0                # first parameter at the top
    n_runs = len(runs)
    span = 0.46                                 # keep well clear of neighbouring rows
    offs = np.linspace(-span / 2, span / 2, n_runs) if n_runs > 1 else [0.0]
    XLO, XHI = -0.08, 1.08

    fig, ax = plt.subplots(figsize=(11.0, 0.60 * n + 2.6))
    ax.axvspan(0, 1, color="#f7f7f7", zorder=0)
    # Separators BETWEEN parameters, never through a series.
    for yy in y:
        ax.axhline(yy + 0.5, color="#e0e0e0", lw=0.7, zorder=0)

    # The range the PPE actually sampled -- with bounds_from_data=true this is
    # the box the optimizer searched, and the edge "at bound" refers to. It is
    # much narrower than the declared bounds for several parameters, so a point
    # can sit at its bound while looking mid-axis on the declared scale.
    if sampled is not None:
        slo, shi = sampled
        for i in range(n):
            yy = y[i]
            ax.add_patch(plt.Rectangle((slo[i], yy - 0.40), shi[i] - slo[i], 0.80,
                                       facecolor="#dce7f2", edgecolor="none",
                                       zorder=1, alpha=0.75))
            for xb in (slo[i], shi[i]):
                ax.plot([xb, xb], [yy - 0.40, yy + 0.40], color="#8aa8c4",
                        lw=1.1, zorder=1)

    handles = []
    for k, (lbl, r) in enumerate(runs):
        c = COLORS[k % len(COLORS)]
        yk = y + offs[k]
        has_se = r["stderr"] is not None
        se = r["stderr"] if has_se else np.full(n, np.nan)
        ab = r["at_bound"] if r["at_bound"] is not None else np.zeros(n, bool)

        finite = np.isfinite(se) if has_se else np.zeros(n, bool)
        if np.any(finite):
            xi, yi, si = r["x"][finite], yk[finite], se[finite]
            # Clip bars to the frame; mark truncated ends with arrows so a
            # sigma wider than the whole parameter range stays readable
            # instead of painting a line across the figure.
            lo = np.maximum(xi - si, XLO)
            hi = np.minimum(xi + si, XHI)
            ax.errorbar(xi, yi, xerr=[xi - lo, hi - xi], fmt="o", ms=5.5,
                        color=c, ecolor=c, elinewidth=1.4, capsize=3.0,
                        alpha=0.95, zorder=3)
            for xx, yy, ll, hh, ss in zip(xi, yi, lo, hi, si):
                if xx - ss < XLO:
                    ax.plot(XLO, yy, marker="<", ms=5, color=c, zorder=3)
                if xx + ss > XHI:
                    ax.plot(XHI, yy, marker=">", ms=5, color=c, zorder=3)
        rest = ~finite
        if np.any(rest):
            # Open markers: pinned at a bound, or a run with no uncertainty report.
            ax.plot(r["x"][rest], yk[rest], "o", ms=5.5, mfc="none", mec=c,
                    mew=1.6, zorder=3)
        # Ring at-bound points so they are distinguishable from "no report at all".
        if np.any(ab):
            ax.plot(r["x"][ab], yk[ab], "o", ms=10.5, mfc="none", mec=c,
                    mew=0.9, alpha=0.55, zorder=2)

        suffix = "±1σ" if has_se else "no error bars (cost is not a likelihood)"
        handles.append(plt.Line2D([], [], marker="o", ls="none", color=c,
                                  mfc=(c if has_se else "none"), mew=1.6,
                                  ms=6, label=f"{lbl}  ({suffix})"))

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("normalised parameter value   (0 / 1 = declared param_physical_bounds)", fontsize=10)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_title(
        "Best-fit parameters by optimization scenario\n"
        "blue band = range the PPE actually sampled (the box the optimizer searched); "
        "error bars = ±1σ (inverse-Hessian), arrow = σ extends beyond the axis;\n"
        "open marker = no stderr; large ring = pinned at a sampled-range bound",
        fontsize=11, pad=14,
    )
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.055 - 0.5 / n),
              ncol=min(3, n_runs), frameon=False, fontsize=9)
    fig.tight_layout()
    out = out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
