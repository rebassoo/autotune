"""PPE ZRG diagnostic plots."""
from __future__ import annotations

import os

import numpy as np

_SNAP_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
_OBS_COLOR   = "tab:red"


def plot_ppe_zrg(
    zrg_result: dict,
    var_names: list,
    n_regions: int,
    regions_list: list,
    snapshots,          # List[SnapshotCfg] — only .label is used
    out_dir: str,
    suffix: str = "",
) -> str:
    """
    One figure with one subplot per variable.  Each subplot shows all PPE
    member values (scatter + median line) and, if present, the obs (red star)
    across the ZRG positions for every snapshot.

    n_zonal is inferred from the data shape so the plot is automatically correct
    after column-dropping.

    Obs overlay is skipped silently for any snapshot/variable where obs is all-NaN
    (e.g. when called after PPE-only preprocessing).

    Returns the path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    n_snaps = len(snapshots)
    n_vars  = len(var_names)
    rng     = np.random.default_rng(0)

    # Infer n_zonal from data shape (correct even after column-dropping)
    sample_df  = zrg_result[f"{var_names[0]}_zrg_ppedataset"]
    n_feat     = sample_df.shape[1]
    n_per_snap = n_feat // n_snaps
    n_zonal    = n_per_snap - n_regions - 1

    # X-tick labels for one snapshot block
    lat_bands  = np.linspace(-90, 90, n_zonal + 1)
    zonal_lbl  = [f"{(lat_bands[i] + lat_bands[i+1]) / 2:.0f}°" for i in range(n_zonal)]
    pos_labels = zonal_lbl + list(regions_list) + ["Global"]
    x          = np.arange(n_per_snap)

    fig, axes = plt.subplots(
        n_vars, 1,
        figsize=(max(14, n_per_snap * 0.75), 4.5 * n_vars),
        squeeze=False,
    )

    for v_idx, var_name in enumerate(var_names):
        ax     = axes[v_idx, 0]
        ppe_df = zrg_result[f"{var_name}_zrg_ppedataset"]

        # Light background shading for Zonal / Regional / Global sections
        ax.axvspan(-0.5,                         n_zonal - 0.5,              alpha=0.04, color="steelblue",  zorder=0)
        ax.axvspan(n_zonal - 0.5,                n_zonal + n_regions - 0.5,  alpha=0.04, color="seagreen",   zorder=0)
        ax.axvspan(n_zonal + n_regions - 0.5,    n_per_snap - 0.5,           alpha=0.08, color="darkorange",  zorder=0)

        for s_idx, snap in enumerate(snapshots):
            color       = _SNAP_COLORS[s_idx % len(_SNAP_COLORS)]
            snap_offset = (s_idx - (n_snaps - 1) / 2) * 0.25

            # PPE scatter — small random jitter reveals point density
            for xi in range(n_per_snap):
                vals = ppe_df.iloc[:, s_idx * n_per_snap + xi].dropna().values
                if not len(vals):
                    continue
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(
                    xi + snap_offset + jitter, vals,
                    color=color, alpha=0.2, s=5, linewidths=0,
                )

            # Median line connecting positions
            medians  = [ppe_df.iloc[:, s_idx * n_per_snap + xi].median()
                        for xi in range(n_per_snap)]
            snap_lbl = snap.label if n_snaps > 1 else "PPE"
            ax.plot(x + snap_offset, medians, color=color, lw=1.5, zorder=3,
                    label=f"{snap_lbl} median")

        # Obs overlay — silently skipped when all-NaN (PPE-only mode)
        if "zrg_obs" in zrg_result:
            obs_row = zrg_result["zrg_obs"]
            for s_idx, snap in enumerate(snapshots):
                snap_offset = (s_idx - (n_snaps - 1) / 2) * 0.25
                col0     = v_idx * n_feat + s_idx * n_per_snap
                obs_vals = obs_row.iloc[0, col0:col0 + n_per_snap].values.astype(float)
                if np.all(np.isnan(obs_vals)):
                    continue
                obs_lbl = f"{snap.label} obs" if n_snaps > 1 else "Obs"
                ax.scatter(x + snap_offset, obs_vals,
                           color=_OBS_COLOR, s=40, marker="*", zorder=5, label=obs_lbl)

        # Axis decoration
        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.7, n_per_snap - 0.3)
        ax.axvline(n_zonal - 0.5,            color="gray", lw=0.8, ls="--", zorder=1)
        ax.axvline(n_zonal + n_regions - 0.5, color="gray", lw=0.8, ls="--", zorder=1)
        ax.set_title(var_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")

        # Section labels just above the plot area
        xform = ax.get_xaxis_transform()
        for mid, lbl in [
            ((n_zonal - 1) / 2,             "Zonal"),
            (n_zonal + (n_regions - 1) / 2, "Regional"),
            (n_zonal + n_regions,           "Global"),
        ]:
            ax.text(mid, 1.01, lbl, ha="center", va="bottom", fontsize=8,
                    color="gray", fontstyle="italic", transform=xform)

    fig.suptitle("PPE ZRG diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"ppe_zrg_diagnostics{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PPE ZRG plot: {out_path}")
    return out_path
