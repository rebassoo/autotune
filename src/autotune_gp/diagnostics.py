"""Post-optimization diagnostic plots and data files saved to out_dir."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch/HPC use
import matplotlib.pyplot as plt
import numpy as np


def run_diagnostics(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                    param_names, var_names, n_zonal, n_regions, regions_list, out_dir,
                    suffix="", n_snaps=1, Y_low_ZRG=None, zonal_center_lats=None):
    """Generate and save all diagnostic plots to out_dir.

    suffix            — appended to each filename before the extension.
    n_snaps           — number of temporal snapshots (1 for ANN, 2 for DY1+DY2, etc.).
    Y_low_ZRG         — (n_low, n_feat, n_vars) LF training data in physical units;
                        when provided (multi-fidelity), LF predictions and LF default
                        are added to the ZRG projection plots.
    zonal_center_lats — list of actual band-centre latitudes for the surviving zonal
                        bands (e.g. [-65., -55., ..., 75.]).  When None the labels are
                        recomputed from n_zonal, which is wrong if bands were dropped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_barcode(results, top_rows, param_names, out_dir, suffix)
    _plot_zrg_projections(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                          var_names, n_zonal, n_regions, regions_list, out_dir, suffix,
                          n_snaps=n_snaps, Y_low_ZRG=Y_low_ZRG,
                          zonal_center_lats=zonal_center_lats)
    _save_projection_data(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                          var_names, n_zonal, n_regions, regions_list, out_dir, suffix,
                          n_snaps=n_snaps, Y_low_ZRG=Y_low_ZRG,
                          zonal_center_lats=zonal_center_lats)


def _plot_barcode(results, top_rows, param_names, out_dir, suffix=""):
    """Heatmap of normalized parameter values + cost, ranked by cost."""
    ranked = results[top_rows]
    main_params = ranked[:, :-1]         # (n_results, n_params)
    cost = ranked[:, -1:].copy()         # (n_results, 1)
    cost_flipped = np.flipud(cost)

    n, n_params = len(top_rows), main_params.shape[1]
    labels = list(param_names) if param_names is not None else [str(i) for i in range(n_params)]

    label_len = max(len(lb) for lb in labels) if labels else 8
    bottom_margin = max(0.25, label_len * 0.055)
    fig, ax = plt.subplots(figsize=(max(14, n_params * 0.55), max(3.0, 0.2 * n + 1.5)))
    fig.subplots_adjust(bottom=bottom_margin)

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

    path = out_dir / f"barcode_params{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_zrg_projections(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                           var_names, n_zonal, n_regions, regions_list, out_dir, suffix="",
                           n_snaps=1, Y_low_ZRG=None, zonal_center_lats=None):
    """Scatter plot per variable: GP projection vs default vs obs.

    When Y_low_ZRG is provided (multi-fidelity), also shows LF GP prediction
    and LF default (first LF member).
    """
    is_multifidelity = Y_low_ZRG is not None
    best_params_norm = results[top_rows[0], :-1].reshape(1, -1)

    m_hf, _ = gp.predict(best_params_norm)     # (1, n_feat, n_vars)
    m_lf = None
    if is_multifidelity and hasattr(gp, "predict_lf"):
        m_lf, _ = gp.predict_lf(best_params_norm)

    n_per_snap = n_zonal + n_regions + 1
    n_feat     = n_per_snap * n_snaps
    x_range    = list(range(n_feat))

    if zonal_center_lats is not None:
        zonal_labels = [f"{c:.0f}" for c in zonal_center_lats]
    else:
        lat_bands    = np.linspace(-90, 90, n_zonal + 1)
        zonal_labels = [f"{(lat_bands[i] + lat_bands[i+1]) / 2:.0f}" for i in range(n_zonal)]

    snap_labels = zonal_labels + list(regions_list) + ["global"]
    zrg_labels  = snap_labels * n_snaps

    point_size = 30

    for j, var in enumerate(var_names):
        sc = Y_scalers[j]

        hf_opt   = sc.inverse_transform(m_hf[:, :, j])[0]     # (n_feat,)
        hf_def   = Y_train_ZRG[0, :, j]                       # first HF member
        obs_vals = obs_parts[j].values[0]                     # (n_feat,)

        fig, ax = plt.subplots(figsize=(max(12, n_feat * 0.6), 4))

        hf_label = 'High-res' if is_multifidelity else 'GP'
        ax.scatter(x_range, hf_opt,  label=f'{hf_label} optimal', marker='s',
                   edgecolors='steelblue', facecolors='none', s=point_size, zorder=4)
        ax.scatter(x_range, hf_def,  label=f'{hf_label} default (m0)', marker='x',
                   color='steelblue', s=point_size, zorder=3)

        if is_multifidelity:
            lf_opt = sc.inverse_transform(m_lf[:, :, j])[0]
            lf_def = Y_low_ZRG[0, :, j]
            ax.scatter(x_range, lf_opt, label='Low-res optimal', marker='s',
                       edgecolors='darkorange', facecolors='none', s=point_size, zorder=4)
            ax.scatter(x_range, lf_def, label='Low-res default (m0)', marker='x',
                       color='darkorange', s=point_size, zorder=3)

        ax.scatter(x_range, obs_vals, label='Obs', marker='^',
                   edgecolors='red', facecolors='none', s=point_size, zorder=5)

        # Vertical dividers between snapshots
        for s in range(1, n_snaps):
            ax.axvline(x=s * n_per_snap - 0.5, color='black', linewidth=1)

        ax.set_xticks(x_range)
        ax.set_xticklabels(zrg_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(var)
        title = f'{var} — GP optimal vs default vs obs (ZRG bins)'
        if is_multifidelity:
            title += '  [blue=HF ne256, orange=LF ne32]'
        ax.set_title(title)
        ax.legend(fontsize=8)

        plt.tight_layout()
        path = out_dir / f"zrg_projection_{var}{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def _save_projection_data(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                           var_names, n_zonal, n_regions, regions_list, out_dir, suffix="",
                           n_snaps=1, Y_low_ZRG=None, zonal_center_lats=None):
    """Save projection arrays as a pickle for cross-run comparison plots.

    Output file: projection_data{suffix}.pkl

    Keys
    ----
    fidelity_type   : 'multi' or 'single'
    var_names       : list of variable name strings
    zrg_labels      : list of x-axis label strings (zonal + regional + global)
    obs             : (n_feat, n_vars)  physical units
    hf_optimal      : (n_feat, n_vars)  GP prediction at best params, physical
    hf_default      : (n_feat, n_vars)  first HF training member, physical
    lf_optimal      : (n_feat, n_vars) or None  LF prediction at best params
    lf_default      : (n_feat, n_vars) or None  first LF training member
    best_params_norm: (n_params,)  normalised parameter vector
    best_cost       : float
    """
    is_multifidelity = Y_low_ZRG is not None
    best_params_norm = results[top_rows[0], :-1]
    best_cost        = float(results[top_rows[0], -1])

    m_hf, _ = gp.predict(best_params_norm.reshape(1, -1))
    m_lf     = None
    if is_multifidelity and hasattr(gp, "predict_lf"):
        m_lf, _ = gp.predict_lf(best_params_norm.reshape(1, -1))

    n_feat = Y_train_ZRG.shape[1]

    if zonal_center_lats is not None:
        zonal_labels = [f"{c:.0f}" for c in zonal_center_lats]
    else:
        lat_bands    = np.linspace(-90, 90, n_zonal + 1)
        zonal_labels = [f"{(lat_bands[i] + lat_bands[i+1]) / 2:.0f}" for i in range(n_zonal)]
    snap_labels = zonal_labels + list(regions_list) + ["global"]
    zrg_labels  = snap_labels * n_snaps

    n_vars = len(var_names)
    hf_optimal = np.stack(
        [Y_scalers[j].inverse_transform(m_hf[:, :, j])[0] for j in range(n_vars)], axis=1
    )   # (n_feat, n_vars)
    hf_default = Y_train_ZRG[0]   # (n_feat, n_vars)
    obs        = np.stack([obs_parts[j].values[0] for j in range(n_vars)], axis=1)

    lf_optimal = None
    lf_default = None
    if is_multifidelity and m_lf is not None:
        lf_optimal = np.stack(
            [Y_scalers[j].inverse_transform(m_lf[:, :, j])[0] for j in range(n_vars)], axis=1
        )
        lf_default = Y_low_ZRG[0]   # (n_feat, n_vars)

    data = dict(
        fidelity_type    = "multi" if is_multifidelity else "single",
        var_names        = list(var_names),
        zrg_labels       = zrg_labels,
        obs              = obs,
        hf_optimal       = hf_optimal,
        hf_default       = hf_default,
        lf_optimal       = lf_optimal,
        lf_default       = lf_default,
        best_params_norm = best_params_norm,
        best_cost        = best_cost,
    )

    path = out_dir / f"projection_data{suffix}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved {path}")
