"""
Per-variable ZRG value tables (all rows: zonal bands + regions + global)
comparing a native MF run against a second run's best-fit parameters
re-evaluated through the MF surrogate -- i.e. a fair, same-surrogate
comparison rather than each run grading its own homework.

MF's Obs/Default/Optimal columns are read straight from its saved
diagnostics/projection_data*.pkl (same source plot_zrg_table_combined.py
uses). The second run's "cross" column is computed fresh: its best-fit
params (from its own results_*.csv) are run through the *MF* GP and
inverse-transformed with the same Y_scalers convention diagnostics.py uses
for hf_optimal, so it's on the same physical-units footing as MF's column.

Cell colouring per row, MF vs cross column only: green = closer to obs,
red = further (same convention as plot_zrg_table_combined.py).

One PNG per variable.

Usage:
    python scripts/plot_regions_table_cross_surrogate.py \
        --mf-dir /path/to/mf_results_dir \
        --mf-gp /path/to/mf_results_dir/mf_gp_trained.pkl \
        --cross-dir /path/to/sf_results_dir \
        --cross-label "SF dof=4 (via MF GP)" \
        --preprocess-dir /path/to/preprocess_output_ne128_prod \
        --out-dir /path/to/out_dir
"""
from __future__ import annotations

import argparse
import ast
import csv
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


def _load_projection(results_dir: str):
    diag_dir = Path(results_dir) / "diagnostics"
    pkls = sorted(glob.glob(str(diag_dir / "projection_data*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No projection_data*.pkl found in {diag_dir}")
    with open(pkls[-1], "rb") as f:
        return pickle.load(f)


def _x_best(results_dir: str) -> np.ndarray:
    csvs = glob.glob(str(Path(results_dir) / "results_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results_*.csv found in {results_dir}")
    with open(sorted(csvs)[-1]) as f:
        rows = list(csv.DictReader(f))
    return np.array(ast.literal_eval(rows[0]["params"]))


def plot_tables(mf_proj: dict, cross_optimal: np.ndarray, cross_label: str,
                 out_dir: Path, out_prefix: str = "regions_table"):
    var_names = mf_proj["var_names"]
    zrg_labels = mf_proj["zrg_labels"]
    obs = mf_proj["obs"]              # (n_feat, n_vars) physical
    default = mf_proj["hf_default"]   # (n_feat, n_vars) physical, TRUE default run
    mf_optimal = mf_proj["hf_optimal"]  # (n_feat, n_vars) physical

    n_feat = len(zrg_labels)
    col_labels = ["Feature", "Obs", "Default (m0)", "MF", cross_label]

    for j, var in enumerate(var_names):
        obs_v, def_v, mf_v, cr_v = (obs[:, j], default[:, j], mf_optimal[:, j],
                                     cross_optimal[:, j])

        cell_text, cell_colors = [], []
        for i in range(n_feat):
            d_mf = abs(mf_v[i] - obs_v[i])
            d_cr = abs(cr_v[i] - obs_v[i])
            if d_mf <= d_cr:
                mf_color, cr_color = GREEN, RED
            else:
                mf_color, cr_color = RED, GREEN

            fmt = "{:.3e}" if var == "PCP" else "{:.4g}"
            cell_colors.append([WHITE, WHITE, WHITE, mf_color, cr_color])
            cell_text.append([zrg_labels[i], fmt.format(obs_v[i]), fmt.format(def_v[i]),
                               fmt.format(mf_v[i]), fmt.format(cr_v[i])])

        fig_h = n_feat * 0.30 + 0.55
        fig, ax = plt.subplots(figsize=(9, fig_h))
        ax.axis("off")
        tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                        colLabels=col_labels, colColours=[WHITE] * len(col_labels),
                        cellLoc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
        title = f"{var} — Obs / Default / MF / {cross_label}  (green=closer to obs, red=further)"
        if var == "PCP":
            title += "\n(raw stored units, ~1e-8 scale)"
        ax.set_title(title, fontsize=10, pad=6)
        fig.tight_layout(pad=0.4)

        path = out_dir / f"{out_prefix}_{var}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mf-dir", required=True, help="MF results dir (native, own GP).")
    p.add_argument("--mf-gp", required=True, help="Path to the trained MF GP pickle.")
    p.add_argument("--cross-dir", required=True,
                   help="Other run's results dir; its best params get re-scored by the MF GP.")
    p.add_argument("--cross-label", default="cross (via MF GP)")
    p.add_argument("--preprocess-dir", required=True,
                   help="Dir containing scalers.pkl (for Y_pipeline inverse-transform).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-prefix", default="regions_table")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MF projection data from {args.mf_dir} ...")
    mf_proj = _load_projection(args.mf_dir)
    var_names = mf_proj["var_names"]

    print(f"Loading Y_pipeline scalers from {args.preprocess_dir} ...")
    with open(Path(args.preprocess_dir) / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    Y_scalers = [scalers[f"Y_pipeline_{v}"] for v in var_names]

    print(f"Loading MF GP checkpoint from {args.mf_gp} (this is the big one, be patient) ...")
    with open(args.mf_gp, "rb") as f:
        gp_mf = pickle.load(f)

    print(f"Loading best-fit params from {args.cross_dir} and scoring with the MF GP ...")
    x_cross = _x_best(args.cross_dir)
    m_cross, _ = gp_mf.predict(x_cross)   # (1, n_feat, n_vars), normalized GP-space
    cross_optimal = np.stack(
        [Y_scalers[j].inverse_transform(m_cross[:, :, j])[0] for j in range(len(var_names))],
        axis=1,
    )   # (n_feat, n_vars), physical units -- same convention as diagnostics.py's hf_optimal

    plot_tables(mf_proj, cross_optimal, args.cross_label, out_dir, args.out_prefix)
    print("Done.")


if __name__ == "__main__":
    main()
