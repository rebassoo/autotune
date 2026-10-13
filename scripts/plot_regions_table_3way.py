"""
Three-way per-variable ZRG region tables: two multi-fidelity optimizations and
one single-fidelity optimization, all compared against observations.

Extends plot_regions_table_cross_surrogate.py (which is two-way) to an
arbitrary number of runs, and keeps its central idea: every run's best-fit
parameters are evaluated through the *same* surrogate -- the MF AR1 GP -- so
the columns differ only in the parameters each optimization chose, not in
which emulator graded them. A run whose own results came from a different
surrogate (e.g. the single-fidelity ESEm run) is re-evaluated through the MF
GP rather than having its own projection_data read directly.

Cell colouring per row: green = closest to obs, red = furthest from obs,
across the optimization columns only (the obs column is never coloured).

One PNG per variable.

Usage:
    python scripts/plot_regions_table_3way.py \
        --mf-gp        /path/to/results_mf.../mf_gp_trained.pkl \
        --preprocess-dir /path/to/preprocess_output_ne128_prod_v2 \
        --native  "MF Laplace=/path/to/results_mf_ne128_ne32_prod_v2" \
        --native  "MF Student-t=/path/to/results_..._studentt_dof4_40starts" \
        --cross   "SF ESEm MAE=/path/to/results_ne128_prod_v2_sf_esem_mae_40starts" \
        --out-dir docs
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN = "#90EE90"
RED = "#FF9999"
WHITE = "white"
HEADER = "#D9D9D9"

# Named regions, in the order they appear in the ZRG layout. Zonal bands are
# discovered from the run's own zrg_labels (they are numeric band centres, and
# which bands survive depends on preprocess.drop_zonal_bands), so they are not
# hard-coded here.
REGIONS = [
    "poles",
    "extratropical_land",
    "extratropical_ocean",
    "tropical_land",
    "ascending_tropical_ocean",
    "descending_tropical_ocean",
    "global",
]
SHORT = {
    "poles": "poles",
    "extratropical_land": "extratropical land",
    "extratropical_ocean": "extratropical ocean",
    "tropical_land": "tropical land",
    "ascending_tropical_ocean": "ascending trop. ocean",
    "descending_tropical_ocean": "descending trop. ocean",
    "global": "GLOBAL",
}


def _is_zonal(label: str) -> bool:
    try:
        float(label)
        return True
    except ValueError:
        return False


def _zonal_name(label: str) -> str:
    """'-65' -> '65°S', '5' -> '5°N'."""
    c = float(label)
    hemi = "S" if c < 0 else "N"
    return f"{abs(c):g}°{hemi}"

# Display units. PCP is carried internally as m/s; the rest are already in
# their natural units.
UNITS = {"PCP": "mm/day", "OSR": "W/m²", "OLR": "W/m²", "TLWP": "kg/m²"}
SCALE = {"PCP": 8.64e7, "OSR": 1.0, "OLR": 1.0, "TLWP": 1.0}
DEC = {"PCP": 3, "OSR": 2, "OLR": 2, "TLWP": 4}


def _load_projection(results_dir: str) -> dict:
    pkls = sorted(glob.glob(str(Path(results_dir) / "diagnostics" / "projection_data*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No projection_data*.pkl found in {results_dir}/diagnostics")
    with open(pkls[-1], "rb") as f:
        return pickle.load(f)


def _parse_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Expected 'Label=/path/to/results_dir', got: {spec!r}")
    # rsplit, not split: labels legitimately contain '=' (e.g. "MF Student-t dof=4"),
    # whereas results paths do not.
    label, path = spec.rsplit("=", 1)
    return label.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mf-gp", required=True,
                    help="Trained MF AR1 GP pickle; every column is evaluated through this.")
    ap.add_argument("--preprocess-dir", required=True,
                    help="Holds scalers.pkl, used to inverse-transform predictions.")
    ap.add_argument("--native", action="append", default=[], metavar="LABEL=DIR",
                    help="Run whose saved projection_data already came from --mf-gp.")
    ap.add_argument("--cross", action="append", default=[], metavar="LABEL=DIR",
                    help="Run from a different surrogate; re-evaluated through --mf-gp.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="regions_table_3way")
    args = ap.parse_args()

    native = [_parse_spec(s) for s in args.native]
    cross = [_parse_spec(s) for s in args.cross]
    if not native:
        raise SystemExit("At least one --native run is required (it supplies obs and labels).")

    ref = _load_projection(native[0][1])
    var_names = list(ref["var_names"])
    labels = list(ref["zrg_labels"])
    obs = np.asarray(ref["obs"], dtype=float)

    # Zonal bands (in ZRG order) first, then the named regions, then global.
    zonal = [l for l in labels if _is_zonal(l)]
    regions = [r for r in REGIONS if r in labels and r != "global"]
    row_labels = zonal + regions + (["global"] if "global" in labels else [])
    ridx = [labels.index(r) for r in row_labels]
    n_zonal, n_regions = len(zonal), len(regions)
    print(f"Rows: {n_zonal} zonal band(s) + {n_regions} region(s) + global")

    columns: list[tuple[str, np.ndarray]] = []

    for label, d in native:
        p = _load_projection(d)
        if list(p["zrg_labels"]) != labels or list(p["var_names"]) != var_names:
            raise SystemExit(f"{label}: ZRG label/variable layout differs from {native[0][0]}")
        if not np.allclose(np.asarray(p["obs"], dtype=float), obs, equal_nan=True):
            raise SystemExit(f"{label}: observations differ from {native[0][0]}")
        columns.append((label, np.asarray(p["hf_optimal"], dtype=float)))

    if cross:
        with open(Path(args.preprocess_dir) / "scalers.pkl", "rb") as f:
            saved = pickle.load(f)
        y_scalers = [saved[f"Y_pipeline_{v}"] for v in var_names]
        print(f"Loading MF GP from {args.mf_gp} ...", flush=True)
        with open(args.mf_gp, "rb") as f:
            gp = pickle.load(f)
        for label, d in cross:
            p = _load_projection(d)
            x = np.asarray(p["best_params_norm"], dtype=float).reshape(1, -1)
            m, _ = gp.predict(x)
            vals = np.stack(
                [y_scalers[j].inverse_transform(m[:, :, j])[0] for j in range(len(var_names))],
                axis=1,
            )
            columns.append((label, vals))
            print(f"  {label}: re-evaluated through the MF GP")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for j, var in enumerate(var_names):
        s, p = SCALE.get(var, 1.0), DEC.get(var, 3)
        o = obs[ridx, j] * s
        vals = [(lbl, arr[ridx, j] * s) for lbl, arr in columns]
        errs = [np.abs(v - o) for _, v in vals]

        header = ["zone / region", f"OBS\n[{UNITS.get(var, '')}]"] + [
            f"{lbl}\nvalue (|err|)" for lbl, _ in vals
        ]
        cell_text, cell_cols = [], []

        def _data_row(i: int, name: str) -> None:
            e = np.array([errs[c][i] for c in range(len(vals))])
            best, worst = int(np.argmin(e)), int(np.argmax(e))
            row = [name, f"{o[i]:.{p}f}"]
            colr = [WHITE, WHITE]
            for c, (_, v) in enumerate(vals):
                row.append(f"{v[i]:.{p}f}  ({errs[c][i]:.{p}f})")
                colr.append(GREEN if c == best else (RED if c == worst else WHITE))
            cell_text.append(row)
            cell_cols.append(colr)

        def _subtotal(name: str, lo: int, hi: int) -> None:
            tot = [float(errs[c][lo:hi].sum()) for c in range(len(vals))]
            bt, wt = int(np.argmin(tot)), int(np.argmax(tot))
            row = [name, ""]
            colr = [HEADER, HEADER]
            for c in range(len(vals)):
                row.append(f"{tot[c]:.{p}f}")
                colr.append(GREEN if c == bt else (RED if c == wt else HEADER))
            cell_text.append(row)
            cell_cols.append(colr)

        # Zonal bands, then their subtotal.
        for i in range(n_zonal):
            _data_row(i, _zonal_name(row_labels[i]))
        if n_zonal:
            _subtotal(f"SUM |err|  ({n_zonal} zonal bands)", 0, n_zonal)

        # Named regions, then their subtotal. Zonal bands and regions each
        # tile the globe, so they are summed separately rather than pooled
        # into one total that would double-count every grid cell.
        for i in range(n_zonal, n_zonal + n_regions):
            _data_row(i, SHORT.get(row_labels[i], row_labels[i]))
        if n_regions:
            _subtotal(f"SUM |err|  ({n_regions} regions)", n_zonal, n_zonal + n_regions)

        # Global, on its own.
        for i in range(n_zonal + n_regions, len(row_labels)):
            _data_row(i, SHORT.get(row_labels[i], row_labels[i]))

        fig, ax = plt.subplots(figsize=(3.4 + 2.5 * len(vals), 0.40 * len(cell_text) + 1.9))
        ax.axis("off")
        tbl = ax.table(cellText=cell_text, colLabels=header,
                       cellColours=cell_cols, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.9)
        for c in range(len(header)):
            hc = tbl[0, c]
            hc.set_facecolor(HEADER)
            hc.set_text_props(weight="bold")
        ax.set_title(
            f"{var}  [{UNITS.get(var, '')}]  —  optimized parameters vs observations\n"
            "all columns evaluated through the same MF AR1 surrogate; "
            "green = closest to obs, red = furthest",
            fontsize=11, pad=16,
        )
        fig.tight_layout()
        out = out_dir / f"{args.prefix}_{var}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
