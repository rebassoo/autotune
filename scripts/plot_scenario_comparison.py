"""
Compare optimization results across three scenarios:
  - Multi-fidelity ne128+ne32 (HF prediction only)
  - Single-fidelity ne128
  - Single-fidelity ne32

For each variable produces a ZRG projection plot overlaying all three
GP-optimal predictions against obs and the HF default member.

Also produces a parameter comparison table (normalized and physical values).

Usage:
    python scripts/plot_scenario_comparison.py \
        --mf-dir     /pscratch/.../results_mf_ne128_ne32_prod_allparams \
        --mf-config  configs/perlmutter_mf_ne128_ne32_prod_annual.yaml \
        --hf-dir     /pscratch/.../results_ne128_prod \
        --hf-config  configs/perlmutter_ne128_prod_annual.yaml \
        --lf-dir     /pscratch/.../results_ne32_prod \
        --lf-config  configs/perlmutter_ne32_prod_annual.yaml \
        --out-dir    /pscratch/.../comparison_plots
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_projection(results_dir: str):
    """Load the projection_data pkl saved by run_diagnostics."""
    diag_dir = Path(results_dir) / "diagnostics"
    pkls = sorted(glob.glob(str(diag_dir / "projection_data*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No projection_data*.pkl found in {diag_dir}")
    with open(pkls[-1], "rb") as f:
        return pickle.load(f)


def _load_best_params_physical(results_dir: str, preprocess_dir: str):
    """Return (param_names, best_params_physical) for the top-ranked result."""
    # Load X scaler and param names from preprocess_dir
    scalers_path = Path(preprocess_dir) / "scalers.pkl"
    with open(scalers_path, "rb") as f:
        saved = pickle.load(f)
    X_sc = saved["X_pipeline"]

    # Get param names from gp_proj.pkl (X_train is a DataFrame with column names)
    gp_proj_path = Path(preprocess_dir) / "gp_proj.pkl"
    with open(gp_proj_path, "rb") as f:
        gp_proj = pickle.load(f)
    X_train = gp_proj["X_train"]
    param_names = list(X_train.columns) if hasattr(X_train, "columns") else \
                  [f"p{i}" for i in range(X_sc.n_features_in_)]

    # Load results CSV
    csvs = sorted(glob.glob(str(Path(results_dir) / "results_*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No results_*.csv found in {results_dir}")
    csv_path = csvs[-1]

    import ast
    best_norm = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["rank"]) == 1:
                best_norm = np.array(ast.literal_eval(row["params"]), dtype=float)
                break

    if best_norm is None:
        raise ValueError(f"No rank-1 row found in {csv_path}")

    # Inverse-transform to physical space
    best_phys = X_sc.inverse_transform(best_norm.reshape(1, -1))[0]
    return param_names, best_phys


# ---------------------------------------------------------------------------
# ZRG comparison plot
# ---------------------------------------------------------------------------

def plot_zrg_comparison(scenarios: list[dict], out_dir: Path):
    """
    scenarios: list of dicts with keys:
        label, color, proj (projection_data dict)
    """
    var_names  = scenarios[0]["proj"]["var_names"]
    zrg_labels = scenarios[0]["proj"]["zrg_labels"]
    obs        = scenarios[0]["proj"]["obs"]        # (n_feat, n_vars)
    hf_default = scenarios[0]["proj"]["hf_default"] # use first scenario's default
    n_feat     = obs.shape[0]
    x_range    = list(range(n_feat))
    point_size = 30

    for j, var in enumerate(var_names):
        fig, ax = plt.subplots(figsize=(max(14, n_feat * 0.6), 4))

        for sc in scenarios:
            proj = sc["proj"]
            opt  = proj["hf_optimal"][:, j]
            ax.scatter(x_range, opt, label=f'{sc["label"]} optimal',
                       marker='s', edgecolors=sc["color"], facecolors='none',
                       s=point_size, zorder=4)

        ax.scatter(x_range, hf_default[:, j], label='Default (m0)',
                   marker='x', color='gray', s=point_size, zorder=3)
        ax.scatter(x_range, obs[:, j], label='Obs',
                   marker='^', edgecolors='red', facecolors='none',
                   s=point_size, zorder=5)

        ax.set_xticks(x_range)
        ax.set_xticklabels(zrg_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(var)
        ax.set_title(f'{var} — GP optimal vs obs (ZRG bins): scenario comparison')
        ax.legend(fontsize=8)
        fig.tight_layout()

        path = out_dir / f"comparison_zrg_{var}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# ZRG value table with colour coding
# ---------------------------------------------------------------------------

def plot_zrg_table(scenarios: list[dict], out_dir: Path,
                   vars_to_plot: tuple = ("PCP", "TLWP")):
    """
    For each variable in vars_to_plot, produce a table:
      rows    = ZRG features (zonal bands + regions + global)
      columns = Obs | scenario 1 | scenario 2 | scenario 3

    Prediction cells are coloured per row:
      green  = closest to obs
      yellow = middle
      red    = furthest from obs
    """
    var_names  = scenarios[0]["proj"]["var_names"]
    zrg_labels = scenarios[0]["proj"]["zrg_labels"]
    obs        = scenarios[0]["proj"]["obs"]   # (n_feat, n_vars)

    col_labels = ["Feature", "Obs", "Default (m0)"] + [sc["label"] for sc in scenarios]
    n_sc = len(scenarios)

    GREEN  = "#90EE90"
    YELLOW = "#FFD700"
    RED    = "#FF9999"
    WHITE  = "white"

    for var in vars_to_plot:
        if var not in var_names:
            print(f"  Warning: {var} not in var_names, skipping table")
            continue
        j        = var_names.index(var)
        obs_vals  = obs[:, j]
        default   = scenarios[0]["proj"]["hf_default"][:, j]
        preds     = [sc["proj"]["hf_optimal"][:, j] for sc in scenarios]
        n_feat    = len(zrg_labels)

        cell_text   = []
        cell_colors = []

        for i in range(n_feat):
            diffs      = np.array([abs(p[i] - obs_vals[i]) for p in preds])
            order      = np.argsort(diffs)   # best → worst

            pred_colors = [WHITE] * n_sc
            pred_colors[order[0]]  = GREEN
            pred_colors[order[-1]] = RED
            for k in range(1, n_sc - 1):
                pred_colors[order[k]] = YELLOW

            cell_colors.append([WHITE, WHITE, WHITE] + pred_colors)
            cell_text.append(
                [zrg_labels[i], f"{obs_vals[i]:.3g}", f"{default[i]:.3g}"]
                + [f"{preds[s][i]:.3g}" for s in range(n_sc)]
            )

        # Header row colours
        header_colors = [[WHITE] * len(col_labels)]

        n_cols  = len(col_labels)
        fig_w   = max(10, n_cols * 1.5)
        fig_h   = max(4,  n_feat * 0.32 + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=col_labels,
            colColours=header_colors[0],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(n_cols)))

        ax.set_title(f"{var} — ZRG values by scenario  "
                     f"(green=closest to obs, red=furthest)",
                     fontsize=10, pad=12)

        path = out_dir / f"comparison_table_{var}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Parameter comparison table
# ---------------------------------------------------------------------------

def plot_param_table(scenarios: list[dict], out_dir: Path):
    """Bar chart + CSV table of best physical parameter values per scenario."""
    # Use the first scenario that has param info as reference for names
    param_names  = scenarios[0]["param_names"]
    n_params     = len(param_names)
    n_scenarios  = len(scenarios)

    # --- CSV ---
    csv_path = out_dir / "comparison_params.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter"] + [sc["label"] for sc in scenarios])
        for i, name in enumerate(param_names):
            row = [name] + [
                f"{sc['best_phys'][i]:.6g}" if i < len(sc["best_phys"]) else "N/A"
                for sc in scenarios
            ]
            w.writerow(row)
    print(f"  Saved {csv_path}")

    # --- Plot: normalized values side by side ---
    # Normalize each scenario's params to [0,1] range for visual comparison
    # (they are already in physical space; normalize by the union min/max)
    all_phys = np.stack([sc["best_phys"][:n_params] for sc in scenarios], axis=0)  # (n_sc, n_params)

    label_len = max(len(p) for p in param_names)
    label_in  = label_len * 0.065
    plot_in   = max(3.0, 0.25 * n_params + 1.0)
    fig_h     = plot_in + label_in + 0.8
    fig, ax   = plt.subplots(figsize=(max(14, n_params * 0.7), fig_h))
    fig.subplots_adjust(bottom=label_in / fig_h)

    width  = 0.8 / n_scenarios
    colors = [sc["color"] for sc in scenarios]
    x      = np.arange(n_params)

    for s, sc in enumerate(scenarios):
        vals = sc["best_phys"][:n_params]
        ax.bar(x + s * width - 0.4 + width / 2, vals / (all_phys.max(axis=0) + 1e-30),
               width=width * 0.9, label=sc["label"], color=sc["color"], alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=90)
    ax.set_ylabel("Relative parameter value (normalized to max across scenarios)")
    ax.set_title("Best parameter values by scenario")
    ax.legend()

    path = out_dir / "comparison_params.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mf-dir",    required=True)
    p.add_argument("--mf-config", required=True)
    p.add_argument("--hf-dir",    required=True)
    p.add_argument("--hf-config", required=True)
    p.add_argument("--lf-dir",    required=True)
    p.add_argument("--lf-config", required=True)
    p.add_argument("--out-dir",   required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configs to get preprocess_dirs
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from autotune_gp.config import load_config

    cfg_mf = load_config(args.mf_config)
    cfg_hf = load_config(args.hf_config)
    cfg_lf = load_config(args.lf_config)

    print("Loading projection data ...")
    proj_mf = _load_projection(args.mf_dir)
    proj_hf = _load_projection(args.hf_dir)
    proj_lf = _load_projection(args.lf_dir)

    print("Loading best parameter values ...")
    param_names_hf, best_phys_mf = _load_best_params_physical(
        args.mf_dir, cfg_mf.paths.preprocess_dir)
    param_names_hf, best_phys_hf = _load_best_params_physical(
        args.hf_dir, cfg_hf.paths.preprocess_dir)
    param_names_lf, best_phys_lf = _load_best_params_physical(
        args.lf_dir, cfg_lf.paths.preprocess_dir)

    scenarios = [
        dict(label="Multi-res ne128+ne32", color="steelblue",
             proj=proj_mf, param_names=param_names_hf, best_phys=best_phys_mf),
        dict(label="SF ne128",             color="seagreen",
             proj=proj_hf, param_names=param_names_hf, best_phys=best_phys_hf),
        dict(label="SF ne32",              color="darkorange",
             proj=proj_lf, param_names=param_names_lf, best_phys=best_phys_lf),
    ]

    print("Plotting ZRG comparisons ...")
    plot_zrg_comparison(scenarios, out_dir)

    print("Plotting ZRG tables (PCP, TLWP) ...")
    plot_zrg_table(scenarios, out_dir)

    print("Plotting parameter table ...")
    plot_param_table(scenarios, out_dir)

    print(f"Done. All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
