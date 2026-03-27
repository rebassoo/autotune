"""
Combine results CSVs from multiple stage-2 sweep runs into a single ranked
CSV and barcode PNG.  Optionally re-trains the GP and generates ZRG
projection plots for the overall best parameter set.

Usage:
    python scripts/combine_sweep_results.py \
        --config configs/scream_autocal.yaml \
        --results-dir /path/to/results

    # Also generate ZRG projection plots for the best result:
    python scripts/combine_sweep_results.py \
        --config configs/scream_autocal.yaml \
        --results-dir /path/to/results \
        --plot-projections

All CSV files matching results_*.csv in --results-dir are loaded.
Outputs (in --results-dir unless --output is given):
    barcode_combined.png   — all results ranked by cost
    barcode_combined.csv   — same data as a ranked CSV
    zrg_projection_*.png   — per-variable ZRG plots (with --plot-projections)
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import load_config


def load_all_results(results_dir: str):
    """Load and combine all results_*.csv files in results_dir."""
    pattern = str(Path(results_dir) / "results_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No results_*.csv files found in {results_dir}")

    all_rows = []
    seeds    = []
    for path in csv_files:
        # Extract seed from filename: results_{n_xstarts}_{seed}_{datetime}.csv
        stem = Path(path).stem
        parts = stem.split("_")
        try:
            seed = int(parts[2])
        except (IndexError, ValueError):
            seed = -1

        with open(path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            bracket_start = line.index("[")
            bracket_end   = line.rindex("]")
            params = ast.literal_eval(line[bracket_start:bracket_end + 1])
            cost   = float(line[bracket_end + 2:])
            all_rows.append(params + [cost])
            seeds.append(seed)

    print(f"Loaded {len(all_rows)} results from {len(csv_files)} CSV files")
    return np.array(all_rows), np.array(seeds), csv_files


def write_combined_csv(results: np.ndarray, seeds: np.ndarray, param_names, output_path: str):
    """Write all results sorted by cost (best first) to a CSV."""
    ranked_idx = np.argsort(np.abs(results[:, -1]))
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "seed"] + list(param_names) + ["cost"])
        for rank, idx in enumerate(ranked_idx, 1):
            row = results[idx]
            w.writerow([rank, int(seeds[idx])] + list(row[:-1]) + [row[-1]])
    print(f"Saved {output_path}")


def plot_barcode(results: np.ndarray, param_names, output_path: str):
    """Barcode heatmap of normalized param values + cost, ranked by cost."""
    costs = results[:, -1]
    ranked_idx = np.argsort(np.abs(costs))
    ranked = results[ranked_idx]

    params = ranked[:, :-1]
    cost   = ranked[:, -1:]

    n, n_params = params.shape
    labels = list(param_names) if param_names is not None else [str(i) for i in range(n_params)]

    fig, ax = plt.subplots(figsize=(12, max(3.0, 0.15 * n + 2.0)))

    im1 = ax.imshow(np.clip(params, 0, 1),
                    aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    im2 = ax.imshow(cost, aspect='auto', cmap='Greens',
                    vmin=cost.min(), vmax=cost.max(),
                    extent=[n_params - 0.5, n_params + 0.5, -0.5, n - 0.5])

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([])
    ax.set_xticks(list(range(n_params)) + [n_params])
    ax.set_xticklabels(labels + ['Cost'], rotation=90)
    ax.set_xlabel('Parameters')
    ax.set_ylabel(f'Rank (1=best, n={n})')
    ax.set_title(f'All {n} optimized parameter sets ranked by cost')

    ax.set_xticks(np.arange(-0.5, n_params + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5, alpha=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_ylim(n - 0.5, -0.5)

    fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.065, label='Normalized Parameter Value')
    fig.colorbar(im2, ax=ax, fraction=0.025, pad=0.01,  label='Cost')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_projections_for_best(results: np.ndarray, cfg, out_dir: Path):
    """Re-train the GP and generate ZRG projection plots for the best result."""
    from autotune_gp.io import load_obs, load_gp_proj, split_zrg_obs
    from autotune_gp.transforms import fit_transform_X, fit_transform_Y, transform_obs
    from autotune_gp.gp import GPWrapper
    from autotune_gp.diagnostics import _plot_zrg_projections

    preprocess_dir = Path(cfg.paths.preprocess_dir)

    # --- Load pkl files (same priority as run_stage2) ---
    if cfg.paths.obs_pkl and Path(cfg.paths.obs_pkl).exists():
        obs_pkl     = cfg.paths.obs_pkl
        gp_proj_pkl = cfg.paths.gp_proj_pkl
    else:
        obs_pkl     = str(preprocess_dir / "obs.pkl")
        gp_proj_pkl = str(preprocess_dir / "gp_proj.pkl")

    print(f"  obs:     {obs_pkl}")
    print(f"  gp_proj: {gp_proj_pkl}")

    obs_loaded  = load_obs(obs_pkl)
    obs_parts   = split_zrg_obs(obs_loaded["zrg_obs"], n_vars=len(cfg.data.variables))

    gp_loaded   = load_gp_proj(gp_proj_pkl)
    X_train     = gp_loaded["X_train"]
    Y_train_ZRG = gp_loaded["Y_train"]
    var_names   = cfg.data.variables

    # --- Normalise ---
    X_sc      = gp_loaded.get("X_pipeline")
    Y_scalers = [gp_loaded.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is None or not all(s is not None for s in Y_scalers):
        scalers_path = preprocess_dir / "scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, "rb") as f:
                saved = pickle.load(f)
            X_sc      = saved.get("X_pipeline")
            Y_scalers = [saved.get(f"Y_pipeline_{v}") for v in var_names]
    if X_sc is not None and all(s is not None for s in Y_scalers):
        X_train_norm = X_sc.transform(X_train)
        Y_train_norm = np.stack(
            [Y_scalers[j].transform(Y_train_ZRG[:, :, j]) for j in range(len(var_names))],
            axis=0).transpose(1, 2, 0)
        print("  Using saved scalers")
    else:
        print("  Warning: no saved scalers found — refitting from scratch")
        X_sc, X_train_norm = fit_transform_X(X_train)
        Y_scalers, Y_train_norm = fit_transform_Y(Y_train_ZRG)

    # --- Train GP ---
    print("  Training GP (this may take a few minutes) ...")
    gp = GPWrapper(X_train_norm, Y_train_norm)
    gp.train(tf_determinism=cfg.runtime.tf_determinism)

    # --- Get best params from combined results ---
    best_idx = int(np.argmin(np.abs(results[:, -1])))
    best_params_norm = results[best_idx, :-1].reshape(1, -1)
    best_cost        = float(results[best_idx, -1])
    print(f"  Best cost: {best_cost:.6f}")

    # --- ZRG layout ---
    n_regions = len(cfg.data.regions_list)
    n_zonal   = Y_train_ZRG.shape[1] // 2 - n_regions - 1

    # Wrap best result in the format _plot_zrg_projections expects
    best_results  = results[best_idx:best_idx + 1]
    best_top_rows = np.array([0])

    _plot_zrg_projections(
        results=best_results,
        top_rows=best_top_rows,
        gp=gp,
        Y_train_ZRG=Y_train_ZRG,
        Y_scalers=Y_scalers,
        obs_parts=obs_parts,
        var_names=var_names,
        n_zonal=n_zonal,
        n_regions=n_regions,
        regions_list=cfg.data.regions_list,
        out_dir=out_dir,
        suffix="_best",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",           required=True,
                   help="Path to YAML config.")
    p.add_argument("--results-dir",      required=True,
                   help="Directory containing results_*.csv files.")
    p.add_argument("--output",           default=None,
                   help="Output PNG path. Default: <results-dir>/barcode_combined.png")
    p.add_argument("--plot-projections", action="store_true",
                   help="Re-train GP and generate ZRG projection plots for the best result.")
    args = p.parse_args()

    cfg = load_config(args.config)
    param_names = list(cfg.optimize.bounds.keys())

    results, seeds, _ = load_all_results(args.results_dir)

    output  = args.output or str(Path(args.results_dir) / "barcode_combined.png")
    csv_out = str(Path(output).with_suffix(".csv"))

    write_combined_csv(results, seeds, param_names, csv_out)
    plot_barcode(results, param_names, output)

    if args.plot_projections:
        print("=== ZRG projection plots (best overall result) ===")
        out_dir = Path(output).parent
        plot_projections_for_best(results, cfg, out_dir)


if __name__ == "__main__":
    main()
