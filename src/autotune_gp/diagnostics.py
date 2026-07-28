"""Post-optimization diagnostic plots and data files saved to out_dir."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch/HPC use
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


def run_diagnostics(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                    param_names, var_names, n_zonal, n_regions, regions_list, out_dir,
                    suffix="", n_snaps=1, Y_low_ZRG=None, zonal_center_lats=None,
                    hf_default_ZRG=None, lf_default_ZRG=None):
    """Generate and save all diagnostic plots to out_dir.

    suffix            — appended to each filename before the extension.
    n_snaps           — number of temporal snapshots (1 for ANN, 2 for DY1+DY2, etc.).
    Y_low_ZRG         — (n_low, n_feat, n_vars) LF training data in physical units;
                        when provided (multi-fidelity), LF predictions and LF default
                        are added to the ZRG projection plots.
    zonal_center_lats — list of actual band-centre latitudes for the surviving zonal
                        bands (e.g. [-65., -55., ..., 75.]).  When None the labels are
                        recomputed from n_zonal, which is wrong if bands were dropped.
    hf_default_ZRG    — (n_feat, n_vars) ZRG values for the true default/control run
                        (physical units). When None, falls back to Y_train_ZRG[0] —
                        which is just the first PPE ensemble member, NOT the true
                        default, and is only a fallback for older preprocess_dirs
                        that predate default_run_zrg.pkl.
    lf_default_ZRG    — same as hf_default_ZRG but for the LF default run
                        (multi-fidelity only).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_barcode(results, top_rows, param_names, out_dir, suffix)
    _plot_zrg_projections(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                          var_names, n_zonal, n_regions, regions_list, out_dir, suffix,
                          n_snaps=n_snaps, Y_low_ZRG=Y_low_ZRG,
                          zonal_center_lats=zonal_center_lats,
                          hf_default_ZRG=hf_default_ZRG, lf_default_ZRG=lf_default_ZRG)
    _save_projection_data(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                          var_names, n_zonal, n_regions, regions_list, out_dir, suffix,
                          n_snaps=n_snaps, Y_low_ZRG=Y_low_ZRG,
                          zonal_center_lats=zonal_center_lats,
                          hf_default_ZRG=hf_default_ZRG, lf_default_ZRG=lf_default_ZRG)


def _plot_barcode(results, top_rows, param_names, out_dir, suffix=""):
    """Heatmap of normalized parameter values + cost, ranked by cost."""
    ranked = results[top_rows]
    main_params = ranked[:, :-1]         # (n_results, n_params)
    cost = ranked[:, -1:].copy()         # (n_results, 1)
    cost_flipped = np.flipud(cost)

    n, n_params = len(top_rows), main_params.shape[1]
    labels = list(param_names) if param_names is not None else [str(i) for i in range(n_params)]

    label_len = max(len(lb) for lb in labels) if labels else 8
    label_in  = label_len * 0.065          # approx inches needed for rotated labels
    plot_in   = max(2.0, 0.2 * n + 1.0)   # inches for the heatmap itself
    fig_h     = plot_in + label_in + 0.8   # 0.8 for title + colorbars
    fig, ax = plt.subplots(figsize=(max(14, n_params * 0.55), fig_h))
    fig.subplots_adjust(bottom=label_in / fig_h)

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
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.025, pad=0.01, label='Cost')
    # Without this, matplotlib's default ScalarFormatter silently factors out a
    # shared offset (e.g. "1e-12+1.104184216") whenever all shown costs are
    # nearly identical (common here — many random starts often converge to the
    # same optimum), leaving tick labels like "8.3" that look like the cost
    # itself but are actually a tiny offset from it — easy to misread.
    cbar2.formatter = ScalarFormatter(useOffset=False)
    cbar2.formatter.set_scientific(False)
    cbar2.update_ticks()

    path = out_dir / f"barcode_params{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_parameter_uncertainty(x, stderr, param_names, out_dir, suffix="",
                               at_bound=None, bounds_low=None, bounds_high=None):
    """
    Error-bar plot of the optimizer's best-found (normalized) parameter values
    +/- 1 stderr, from the inverse-Hessian covariance computed by
    autotune_gp.uncertainty.parameter_uncertainty(). Only meaningful when that
    was computed against a real likelihood (optimize.likelihood='gaussian_gp_variance'),
    and even then is a relative/order-of-magnitude indicator, not a calibrated
    absolute confidence interval (see that function's docstring for why).

    at_bound, bounds_low, bounds_high — the same arrays parameter_uncertainty()
    used/returned. When given:
      - each parameter's own feasible range [bounds_low[i], bounds_high[i]] is
        shaded individually, instead of one flat [0,1] band for every column.
        With optimize.bounds_from_data=true this per-parameter range can be far
        narrower than [0,1] (e.g. a parameter only ever sampled in [0, 0.19]),
        and a value that looks safely interior against a [0,1] backdrop can
        actually be sitting right on its own real ceiling — this makes that
        visible instead of hiding it.
      - NaN parameters are split by *why* they're NaN: pinned at a (possibly
        narrowed) bound is the deliberate, by-design exclusion documented in
        parameter_uncertainty(); a free (non-boundary) parameter whose Hessian
        still failed to invert is a genuine numerical failure. Conflating the
        two under one "non-invertible Hessian" label (the previous behavior)
        misreports the common case, since most NaNs come from the former.
    Without these three arrays, falls back to the previous flat-[0,1]-band,
    single-reason-label behavior (e.g. when called from older pickles that
    didn't save at_bound/bounds).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x      = np.asarray(x, dtype=float)
    stderr = np.asarray(stderr, dtype=float)
    n_params = len(x)
    labels = list(param_names) if param_names is not None else [str(i) for i in range(n_params)]

    is_nan      = np.isnan(stderr)
    n_nan       = int(np.sum(is_nan))
    stderr_plot = np.nan_to_num(stderr, nan=0.0)

    if at_bound is not None:
        at_bound   = np.asarray(at_bound, dtype=bool)
        n_pinned   = int(np.sum(is_nan & at_bound))
        n_singular = int(np.sum(is_nan & ~at_bound))
    else:
        n_pinned, n_singular = 0, n_nan   # fall back to the old undifferentiated count

    fig, ax = plt.subplots(figsize=(max(10, n_params * 0.5), 5))
    xpos = np.arange(n_params)

    if bounds_low is not None and bounds_high is not None:
        bl = np.asarray(bounds_low, dtype=float)
        bh = np.asarray(bounds_high, dtype=float)
        for i in range(n_params):
            ax.fill_between([i - 0.4, i + 0.4], bl[i], bh[i], color='0.9', zorder=0)
    else:
        ax.axhspan(0, 1, color='0.9', zorder=0)   # fallback: flat normalized parameter range

    if at_bound is not None:
        pinned_mask = is_nan & at_bound
        singular_mask = is_nan & ~at_bound
        free_mask = ~is_nan
        for mask, color, label in [
            (free_mask,    'tab:blue',   'free (real stderr)'),
            (pinned_mask,  'tab:gray',   'pinned at bound (excluded, NaN by design)'),
            (singular_mask, 'tab:red',   'free but non-invertible (NaN — numerical failure)'),
        ]:
            if not np.any(mask):
                continue
            ax.errorbar(xpos[mask], x[mask], yerr=stderr_plot[mask], fmt='o', color=color,
                       ecolor=color, elinewidth=1.5, capsize=4, markersize=5, zorder=1,
                       label=label)
        ax.legend(fontsize=7, loc='best')
    else:
        ax.errorbar(xpos, x, yerr=stderr_plot, fmt='o', color='tab:blue',
                   ecolor='tab:blue', elinewidth=1.5, capsize=4, markersize=5, zorder=1)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('Normalized parameter value')
    title = 'Best-fit parameters ± 1 stderr (inverse-Hessian — relative uncertainty, not a calibrated CI)'
    detail = []
    if n_pinned:
        detail.append(f'{n_pinned} pinned at bound (excluded by design)')
    if n_singular:
        detail.append(f'{n_singular} free but non-invertible Hessian')
    if detail:
        title += '\n' + '; '.join(detail)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    path = out_dir / f"parameter_uncertainty{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _plot_zrg_projections(results, top_rows, gp, Y_train_ZRG, Y_scalers, obs_parts,
                           var_names, n_zonal, n_regions, regions_list, out_dir, suffix="",
                           n_snaps=1, Y_low_ZRG=None, zonal_center_lats=None,
                           hf_default_ZRG=None, lf_default_ZRG=None):
    """Scatter plot per variable: GP projection vs default vs obs.

    When Y_low_ZRG is provided (multi-fidelity), also shows LF GP prediction
    and LF default.
    """
    is_multifidelity = Y_low_ZRG is not None
    best_params_norm = results[top_rows[0], :-1].reshape(1, -1)

    if hf_default_ZRG is None:
        print("  Warning: no true default-run ZRG provided — falling back to "
              "the first PPE ensemble member, which is NOT the true default. "
              "Run scripts/compute_default_run_zrg.py to fix this.")
        hf_default_ZRG = Y_train_ZRG[0]
    if is_multifidelity and lf_default_ZRG is None:
        lf_default_ZRG = Y_low_ZRG[0]

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
        hf_def   = hf_default_ZRG[:, j]                       # true default run
        obs_vals = obs_parts[j].values[0]                     # (n_feat,)

        fig, ax = plt.subplots(figsize=(max(12, n_feat * 0.6), 4))

        hf_label = 'High-res' if is_multifidelity else 'GP'
        ax.scatter(x_range, hf_opt,  label=f'{hf_label} optimal', marker='s',
                   edgecolors='steelblue', facecolors='none', s=point_size, zorder=4)
        ax.scatter(x_range, hf_def,  label=f'{hf_label} default', marker='x',
                   color='steelblue', s=point_size, zorder=3)

        if is_multifidelity:
            lf_opt = sc.inverse_transform(m_lf[:, :, j])[0]
            lf_def = lf_default_ZRG[:, j]
            ax.scatter(x_range, lf_opt, label='Low-res optimal', marker='s',
                       edgecolors='darkorange', facecolors='none', s=point_size, zorder=4)
            ax.scatter(x_range, lf_def, label='Low-res default', marker='x',
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
                           n_snaps=1, Y_low_ZRG=None, zonal_center_lats=None,
                           hf_default_ZRG=None, lf_default_ZRG=None):
    """Save projection arrays as a pickle for cross-run comparison plots.

    Output file: projection_data{suffix}.pkl

    Keys
    ----
    fidelity_type   : 'multi' or 'single'
    var_names       : list of variable name strings
    zrg_labels      : list of x-axis label strings (zonal + regional + global)
    obs             : (n_feat, n_vars)  physical units
    hf_optimal      : (n_feat, n_vars)  GP prediction at best params, physical
    hf_default      : (n_feat, n_vars)  true HF default/control run, physical
    lf_optimal      : (n_feat, n_vars) or None  LF prediction at best params
    lf_default      : (n_feat, n_vars) or None  true LF default/control run
    best_params_norm: (n_params,)  normalised parameter vector
    best_cost       : float
    """
    is_multifidelity = Y_low_ZRG is not None
    best_params_norm = results[top_rows[0], :-1]
    best_cost        = float(results[top_rows[0], -1])

    if hf_default_ZRG is None:
        print("  Warning: no true default-run ZRG provided — falling back to "
              "the first PPE ensemble member, which is NOT the true default. "
              "Run scripts/compute_default_run_zrg.py to fix this.")
        hf_default_ZRG = Y_train_ZRG[0]
    if is_multifidelity and lf_default_ZRG is None:
        lf_default_ZRG = Y_low_ZRG[0]

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
    hf_default = hf_default_ZRG   # (n_feat, n_vars)
    obs        = np.stack([obs_parts[j].values[0] for j in range(n_vars)], axis=1)

    lf_optimal = None
    lf_default = None
    if is_multifidelity and m_lf is not None:
        lf_optimal = np.stack(
            [Y_scalers[j].inverse_transform(m_lf[:, :, j])[0] for j in range(n_vars)], axis=1
        )
        lf_default = lf_default_ZRG   # (n_feat, n_vars)

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
