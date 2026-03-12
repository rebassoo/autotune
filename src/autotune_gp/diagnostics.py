"""Post-optimization diagnostic plots saved as PNGs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch/HPC use
import matplotlib.pyplot as plt
import numpy as np


def run_diagnostics(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                    param_names, var_names, n_zonal, n_regions, regions_list, out_dir):
    """Generate and save all diagnostic plots to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_barcode(results, top_rows, param_names, out_dir)
    _plot_zrg_projections(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                          var_names, n_zonal, n_regions, regions_list, out_dir)


def _plot_barcode(results, top_rows, param_names, out_dir):
    """Heatmap of normalized parameter values + cost, ranked by cost."""
    ranked = results[top_rows]
    main_params = ranked[:, :-1]         # (n_results, n_params)
    cost = ranked[:, -1:].copy()         # (n_results, 1)
    cost_flipped = np.flipud(cost)

    n, n_params = len(top_rows), main_params.shape[1]
    labels = list(param_names) if param_names is not None else [str(i) for i in range(n_params)]

    fig, ax = plt.subplots(figsize=(11, max(2.0, 0.2 * n + 1.5)))

    im1 = ax.imshow(np.clip(main_params, 0, 1),
                    aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    im2 = ax.imshow(cost_flipped, aspect='auto', cmap='Greens',
                    vmin=cost.min(), vmax=cost.max(),
                    extent=[n_params - 0.5, n_params + 0.5, -0.5, n - 0.5])

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([])
    ax.set_xticks(list(range(n_params)) + [n_params])
    ax.set_xticklabels(labels + ['Cost'], rotation=90)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Rank')
    ax.set_title('Optimized parameter values ranked by cost')

    ax.set_xticks(np.arange(-0.5, n_params + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_ylim(n - 0.5, -0.5)

    fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.065, label='Normalized Parameter Value')
    fig.colorbar(im2, ax=ax, fraction=0.025, pad=0.01, label='Cost')

    plt.tight_layout()
    path = out_dir / "barcode_params.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_zrg_projections(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                           var_names, n_zonal, n_regions, regions_list, out_dir):
    """Scatter plot per variable: GP optimal projection vs default (m0000) vs obs."""
    best_params_norm = results[top_rows[0], :-1].reshape(1, -1)
    m_opt, _ = gp.predict(best_params_norm)  # (1, n_feat, n_vars)

    n_per_day = n_zonal + n_regions + 1
    lat_bands = np.linspace(-90, 90, n_zonal + 1)
    zonal_labels = [f"{(lat_bands[i] + lat_bands[i+1]) / 2:.0f}" for i in range(n_zonal)]
    zrg_labels = zonal_labels + list(regions_list) + ["global"]  # length = n_per_day

    x_range = list(range(n_per_day * 2))
    point_size = 30

    for j, var in enumerate(var_names):
        sc = Y_scalers[j]

        # GP-projected optimal (inverse transform from normalized space)
        opt_pred = sc.inverse_transform(m_opt[:, :, j])[0]

        # Default run (m0000) — Y_train_ZRG is in physical units
        default_vals = Y_train_ZRG[0, :, j]

        # Observations — obs_parts are already in physical units
        obs_vals = obs_parts[j].values[0]

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.scatter(x_range, opt_pred,    label='GP optimal projection', marker='s',
                   edgecolors='green', facecolors='none', s=point_size)
        ax.scatter(x_range, default_vals, label='Default (m0000)',       marker='x',
                   color='blue', s=point_size)
        ax.scatter(x_range, obs_vals,     label='Obs',                   marker='^',
                   edgecolors='red', facecolors='none', s=point_size)

        ax.axvline(x=n_per_day - 0.5, color='black', linewidth=1)

        ax.set_xticks(x_range)
        ax.set_xticklabels(zrg_labels * 2, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel("DY1: Zonal / Regions / Global          "
                      "DY2: Zonal / Regions / Global")
        ax.set_ylabel(var)
        ax.set_title(f'{var} — GP optimal projection vs default vs obs (ZRG bins)')
        ax.legend(fontsize=8)

        plt.tight_layout()
        path = out_dir / f"zrg_projection_{var}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")
