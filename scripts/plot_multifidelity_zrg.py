"""
Multi-fidelity ZRG comparison plot.

Loads ZRG data from two preprocess directories (high-fidelity ne256 remapped
and low-fidelity ne32 native) and overlays both PPE distributions plus
observations on a single figure.

Obs are read from whichever dataset has them (expected to be the LF/ne32 run
preprocessed with --preprocess-mode both on Perlmutter).  If both have obs,
the LF obs are shown (they are the same grid, so results are identical).

Usage:
    python scripts/plot_multifidelity_zrg.py \\
        --config-hf configs/aurora_ne256_remapped_ne32pg2_annual.yaml \\
        --config-lf configs/perlmutter_ne32_annual.yaml

    # Override preprocess dirs directly:
    python scripts/plot_multifidelity_zrg.py \\
        --config-hf configs/aurora_ne256_remapped_ne32pg2_annual.yaml \\
        --config-lf configs/perlmutter_ne32_annual.yaml \\
        --preprocess-dir-hf /path/to/hf_preprocess \\
        --preprocess-dir-lf /path/to/lf_preprocess \\
        --out-dir /path/to/output --suffix _v2
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import load_config

_HF_COLOR  = "tab:blue"
_LF_COLOR  = "tab:orange"
_OBS_COLOR = "tab:red"


def _load_zrg(preprocess_dir: Path) -> dict:
    clean_pkl = preprocess_dir / "zrg_data_clean.pkl"
    raw_pkl   = preprocess_dir / "zrg_data.pkl"
    if clean_pkl.exists():
        path = clean_pkl
    elif raw_pkl.exists():
        path = raw_pkl
        print(f"  Warning: using {raw_pkl} — zrg_data_clean.pkl not found")
    else:
        raise FileNotFoundError(
            f"No ZRG data found in {preprocess_dir}. "
            "Run 'python scripts/run_two_stage.py --stage 1' first."
        )
    print(f"  Loading {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _pos_labels(zrg_result: dict, var_names: list, n_regions: int,
                regions_list: list, n_snaps: int) -> tuple[np.ndarray, list, int, int]:
    """Return (x, pos_labels, n_zonal, n_per_snap) for one snapshot block."""
    sample_df  = zrg_result[f"{var_names[0]}_zrg_ppedataset"]
    n_feat     = sample_df.shape[1]
    n_per_snap = n_feat // n_snaps
    n_zonal    = n_per_snap - n_regions - 1

    if "_n_zonal_original" in zrg_result:
        n_zonal_orig    = zrg_result["_n_zonal_original"]
        n_per_snap_orig = zrg_result["_n_per_snap_original"]
        valid_indices   = zrg_result["_valid_feat_indices"]
        lb_orig         = np.linspace(-90, 90, n_zonal_orig + 1)
        orig_labels     = (
            [f"{(lb_orig[i] + lb_orig[i+1]) / 2:.0f}°" for i in range(n_zonal_orig)]
            + list(regions_list)
            + ["Global"]
        )
        valid_in_snap0 = sorted(i for i in valid_indices if i < n_per_snap_orig)
        labels = [orig_labels[i] for i in valid_in_snap0]
    else:
        lat_bands = np.linspace(-90, 90, n_zonal + 1)
        labels = (
            [f"{(lat_bands[i] + lat_bands[i+1]) / 2:.0f}°" for i in range(n_zonal)]
            + list(regions_list)
            + ["Global"]
        )

    return np.arange(n_per_snap), labels, n_zonal, n_per_snap


def plot_multifidelity_zrg(
    zrg_hf: dict,
    zrg_lf: dict,
    var_names: list,
    n_regions: int,
    regions_list: list,
    snapshots,
    out_dir: str,
    suffix: str = "",
) -> str:
    """
    One figure, one subplot per variable, showing HF PPE (blue), LF PPE (orange),
    and obs (red stars).  Obs are taken from zrg_lf if available, else from zrg_hf.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    n_snaps = len(snapshots)
    n_vars  = len(var_names)
    rng     = np.random.default_rng(0)

    x_hf, labels_hf, n_zonal_hf, n_per_snap_hf = _pos_labels(
        zrg_hf, var_names, n_regions, regions_list, n_snaps)
    x_lf, labels_lf, n_zonal_lf, n_per_snap_lf = _pos_labels(
        zrg_lf, var_names, n_regions, regions_list, n_snaps)

    # Warn if the two grids have different numbers of positions after band-dropping
    if n_per_snap_hf != n_per_snap_lf:
        print(
            f"Warning: HF has {n_per_snap_hf} positions/snapshot and "
            f"LF has {n_per_snap_lf}. Plotting separately on shared axes."
        )

    # Use LF labels for the x-axis (LF is the reference grid)
    x          = x_lf
    pos_labels = labels_lf
    n_zonal    = n_zonal_lf
    n_per_snap = n_per_snap_lf

    # Pick obs source: prefer LF (it was preprocessed with obs)
    obs_source = zrg_lf if "zrg_obs" in zrg_lf else (
        zrg_hf if "zrg_obs" in zrg_hf else None
    )

    fig, axes = plt.subplots(
        n_vars, 1,
        figsize=(max(14, n_per_snap * 0.75), 4.5 * n_vars),
        squeeze=False,
    )

    for v_idx, var_name in enumerate(var_names):
        ax = axes[v_idx, 0]

        # Background shading
        ax.axvspan(-0.5,                         n_zonal - 0.5,              alpha=0.04, color="steelblue", zorder=0)
        ax.axvspan(n_zonal - 0.5,                n_zonal + n_regions - 0.5,  alpha=0.04, color="seagreen",  zorder=0)
        ax.axvspan(n_zonal + n_regions - 0.5,    n_per_snap - 0.5,           alpha=0.08, color="darkorange", zorder=0)

        def _scatter_ppe(ppe_df, color, label_prefix, n_per_snap_local, x_local,
                         s_idx_, snap_offset_, snap_lbl_):
            for xi in range(n_per_snap_local):
                vals = ppe_df.iloc[:, s_idx_ * n_per_snap_local + xi].dropna().values
                if not len(vals):
                    continue
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(
                    xi + snap_offset_ + jitter, vals,
                    color=color, alpha=0.15, s=5, linewidths=0,
                )
            medians = [
                ppe_df.iloc[:, s_idx_ * n_per_snap_local + xi].median()
                for xi in range(n_per_snap_local)
            ]
            ax.plot(
                x_local + snap_offset_, medians,
                color=color, lw=1.8, zorder=3,
                label=f"{label_prefix} {snap_lbl_} median",
            )

        for s_idx, snap in enumerate(snapshots):
            snap_lbl    = snap.label if n_snaps > 1 else "PPE"
            snap_offset = (s_idx - (n_snaps - 1) / 2) * 0.25

            _scatter_ppe(zrg_hf[f"{var_name}_zrg_ppedataset"],
                         _HF_COLOR, "ne256 (HF)", n_per_snap_hf, x_hf,
                         s_idx, snap_offset, snap_lbl)
            _scatter_ppe(zrg_lf[f"{var_name}_zrg_ppedataset"],
                         _LF_COLOR, "ne32 (LF)", n_per_snap_lf, x_lf,
                         s_idx, snap_offset, snap_lbl)

        # Obs overlay
        if obs_source is not None and "zrg_obs" in obs_source:
            obs_row = obs_source["zrg_obs"]
            n_feat_obs = obs_row.shape[1] // n_vars
            n_per_snap_obs = n_feat_obs // n_snaps
            for s_idx, snap in enumerate(snapshots):
                snap_offset = (s_idx - (n_snaps - 1) / 2) * 0.25
                col0     = v_idx * n_feat_obs + s_idx * n_per_snap_obs
                obs_vals = obs_row.iloc[0, col0:col0 + n_per_snap_obs].values.astype(float)
                if np.all(np.isnan(obs_vals)):
                    continue
                obs_lbl = f"{snap.label} obs" if n_snaps > 1 else "Obs"
                ax.scatter(x + snap_offset, obs_vals,
                           color=_OBS_COLOR, s=50, marker="*", zorder=5, label=obs_lbl)

        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.7, n_per_snap - 0.3)
        ax.axvline(n_zonal - 0.5,              color="gray", lw=0.8, ls="--", zorder=1)
        ax.axvline(n_zonal + n_regions - 0.5,  color="gray", lw=0.8, ls="--", zorder=1)
        ax.set_title(var_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")

        xform = ax.get_xaxis_transform()
        for mid, lbl in [
            ((n_zonal - 1) / 2,              "Zonal"),
            (n_zonal + (n_regions - 1) / 2,  "Regional"),
            (n_zonal + n_regions,            "Global"),
        ]:
            ax.text(mid, 1.01, lbl, ha="center", va="bottom", fontsize=8,
                    color="gray", fontstyle="italic", transform=xform)

    fig.suptitle("Multi-fidelity ZRG diagnostics  (blue = ne256 HF,  orange = ne32 LF)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"mf_zrg_diagnostics{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved multi-fidelity ZRG plot: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Overlay HF (ne256) and LF (ne32) PPE ZRG distributions on one plot."
    )
    p.add_argument("--config-hf", required=True,
                   help="YAML config for high-fidelity (ne256 remapped) preprocessing.")
    p.add_argument("--config-lf", required=True,
                   help="YAML config for low-fidelity (ne32) preprocessing.")
    p.add_argument("--preprocess-dir-hf", default=None,
                   help="Override HF preprocess_dir from config.")
    p.add_argument("--preprocess-dir-lf", default=None,
                   help="Override LF preprocess_dir from config.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for the PNG (default: HF preprocess_dir).")
    p.add_argument("--suffix", default="",
                   help="Optional suffix appended to the PNG filename.")
    args = p.parse_args()

    cfg_hf = load_config(args.config_hf)
    cfg_lf = load_config(args.config_lf)

    pp_hf = cfg_hf.preprocess
    pp_lf = cfg_lf.preprocess
    if pp_hf is None or pp_lf is None:
        raise ValueError("Both configs must include a [preprocess] section.")

    preprocess_dir_hf = Path(args.preprocess_dir_hf or cfg_hf.paths.preprocess_dir)
    preprocess_dir_lf = Path(args.preprocess_dir_lf or cfg_lf.paths.preprocess_dir)
    out_dir           = args.out_dir or str(preprocess_dir_hf)

    var_names   = list(pp_hf.variables.keys())
    regions     = cfg_hf.data.regions_list
    n_regions   = len(regions)
    snapshots   = pp_hf.snapshots

    print(f"HF preprocess dir: {preprocess_dir_hf}")
    print(f"LF preprocess dir: {preprocess_dir_lf}")

    zrg_hf = _load_zrg(preprocess_dir_hf)
    zrg_lf = _load_zrg(preprocess_dir_lf)

    plot_multifidelity_zrg(
        zrg_hf=zrg_hf,
        zrg_lf=zrg_lf,
        var_names=var_names,
        n_regions=n_regions,
        regions_list=regions,
        snapshots=snapshots,
        out_dir=out_dir,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
