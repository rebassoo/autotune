"""
Combine results CSVs from multiple stage-2 runs and plot a single barcode
heatmap of all optimized parameter values ranked by cost.

Usage:
    python scripts/plot_combined_barcode.py \
        --config configs/scream_autocal.yaml \
        --results-dir /path/to/results \
        --output barcode_combined.png

All CSV files matching results_*.csv in --results-dir are loaded.
"""
from __future__ import annotations

import argparse
import ast
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autotune_gp.config import load_config


def load_all_results(results_dir: str):
    """Load and combine all results_*.csv files in results_dir."""
    pattern = str(Path(results_dir) / "results_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No results_*.csv files found in {results_dir}")

    all_rows = []
    for path in csv_files:
        with open(path) as f:
            lines = f.readlines()
        header = lines[0].strip().split(",")
        # columns: rank, params (python list as string), cost
        for line in lines[1:]:
            # params column may contain commas inside brackets — split carefully
            line = line.strip()
            # find the list boundaries
            bracket_start = line.index("[")
            bracket_end   = line.rindex("]")
            params = ast.literal_eval(line[bracket_start:bracket_end + 1])
            cost   = float(line[bracket_end + 2:])  # after '],'
            all_rows.append(params + [cost])

    print(f"Loaded {len(all_rows)} results from {len(csv_files)} CSV files")
    return np.array(all_rows), csv_files


def plot_barcode(results: np.ndarray, param_names, output_path: str):
    """Barcode heatmap of normalized param values + cost, ranked by cost."""
    costs = results[:, -1]
    ranked_idx = np.argsort(np.abs(costs))
    ranked = results[ranked_idx]

    params = ranked[:, :-1]   # (n_results, n_params) — already in [0,1]
    cost   = ranked[:, -1:]   # (n_results, 1)

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",      required=True,
                   help="Path to YAML config (used for parameter names).")
    p.add_argument("--results-dir", required=True,
                   help="Directory containing results_*.csv files.")
    p.add_argument("--output",      default=None,
                   help="Output PNG path. Default: <results-dir>/barcode_combined.png")
    args = p.parse_args()

    cfg = load_config(args.config)
    param_names = list(cfg.optimize.bounds.keys())

    results, csv_files = load_all_results(args.results_dir)

    output = args.output or str(Path(args.results_dir) / "barcode_combined.png")
    plot_barcode(results, param_names, output)


if __name__ == "__main__":
    main()
