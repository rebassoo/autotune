"""
Plot LF vs HF data comparison to assess multi-fidelity GP suitability.

Four plot types:
  1. ZRG profiles  — ensemble mean ± 1 std for LF and HF per variable
  2. LF vs HF scatter — HF mean vs LF mean per ZRG feature; slope ≈ expected ρ
  3. Sensitivity heatmaps — Pearson r(param, ZRG feature) for LF and HF side by side
  4. Spread ratio — HF_std / LF_std per feature; shows where variability differs

Usage:
    python scripts/plot_lf_hf_comparison.py \
        --hf  <hf_preprocess_dir> \
        --lf  <lf_preprocess_dir> \
        --out lf_hf_comparison.pdf \
        [--vars PCP OSR OLR] \
        [--n-zonal 15] \
        [--regions poles extratropical_land ...]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

# ── defaults matching the Perlmutter MF config ───────────────────────────────
DEFAULT_HF  = ("/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning"
               "/preprocess_output_ne256_remapped_ne32pg2")
DEFAULT_LF  = ("/global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning"
               "/preprocess_output_ne32")
DEFAULT_OUT = "lf_hf_comparison.pdf"
DEFAULT_VARS = ["PCP", "OSR", "OLR"]
DEFAULT_REGIONS = [
    "poles", "extratropical_land", "extratropical_ocean",
    "tropical_land", "ascending_tropical_ocean", "descending_tropical_ocean",
]
N_ZONAL_ORIG = 18   # bands before dropping
DROPPED_BANDS = [-85, -75, 85]   # dropped in preprocessing

C_HF = "#d6604d"
C_LF = "#2166ac"


# ── helpers ───────────────────────────────────────────────────────────────────
def load(preprocess_dir: str):
    d = Path(preprocess_dir)
    with open(d / "gp_proj.pkl", "rb") as f:
        proj = pickle.load(f)
    with open(d / "column_mask.pkl", "rb") as f:
        mask = pickle.load(f)
    return proj["X_train"], proj["Y_train"], mask


def feat_labels(n_zonal_orig, dropped, regions):
    """Return labels for surviving ZRG features."""
    edges  = np.linspace(-90, 90, n_zonal_orig + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    surviving = [c for c in centers if not any(abs(c - d) < 1 for d in dropped)]
    return [f"{c:.0f}" for c in surviving] + list(regions) + ["global"]


def param_labels():
    return [
        "thl2tune", "qw2tune", "length_fac", "c_diag_3rd_mom",
        "coeff_kh", "coeff_km", "lambda_low", "lambda_high",
        "spa_ccn_to_nc_factor", "cldliq_to_ice", "rain_to_ice",
        "accretion", "dep_nuc_exp", "max_total_ni", "ice_sed_fac",
        "rain_sb_diam", "autoconv_prefac", "autoconv_qc_exp", "autoconv_rad",
    ]


# ── shared save helper ────────────────────────────────────────────────────────
def save_fig(fig, pdf, png_dir: Path | None, name: str):
    if png_dir is not None:
        fig.savefig(png_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── page 1: ZRG profiles ──────────────────────────────────────────────────────
def page_profiles(pdf, X_hf, Y_hf, X_lf, Y_lf, var_names, feat_lbs,
                  lf_label="LF", hf_label="HF", png_dir=None):
    n_feat = Y_hf.shape[1]
    n_vars = len(var_names)
    x      = np.arange(n_feat)

    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    fig.suptitle(f"ZRG Profiles: Ensemble Mean ± 1 Std  (LR={lf_label}, HR={hf_label})",
                 fontsize=13, fontweight="bold", y=1.01)

    for j, (var, ax) in enumerate(zip(var_names, axes)):
        hf_mean = Y_hf[:, :, j].mean(axis=0)
        hf_std  = Y_hf[:, :, j].std(axis=0)
        lf_mean = Y_lf[:, :, j].mean(axis=0)
        lf_std  = Y_lf[:, :, j].std(axis=0)

        ax.fill_between(x, hf_mean - hf_std, hf_mean + hf_std,
                        alpha=0.20, color=C_HF)
        ax.fill_between(x, lf_mean - lf_std, lf_mean + lf_std,
                        alpha=0.20, color=C_LF)
        ax.plot(x, hf_mean, color=C_HF, lw=1.8, label=f"High-res {hf_label} (n={Y_hf.shape[0]})")
        ax.plot(x, lf_mean, color=C_LF, lw=1.8, label=f"Low-res  {lf_label} (n={Y_lf.shape[0]})")

        ax.set_ylabel(var, fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Vertical separator before regional features
        n_zonal = n_feat - len(DEFAULT_REGIONS) - 1
        ax.axvline(n_zonal - 0.5, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(n_feat - 1.5, color="black", lw=0.8, ls="--", alpha=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feat_lbs, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    save_fig(fig, pdf, png_dir, "01_zrg_profiles")


# ── page 2: LF vs HF scatter per variable ────────────────────────────────────
def page_scatter(pdf, Y_hf, Y_lf, var_names, feat_lbs, png_dir=None):
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
    if n_vars == 1:
        axes = [axes]

    fig.suptitle("Low-res vs High-res Ensemble Means per ZRG Feature\n"
                 "(slope ≈ expected ρ;  R² shows linear predictability)",
                 fontsize=12, fontweight="bold")

    for j, (var, ax) in enumerate(zip(var_names, axes)):
        hf_mean = Y_hf[:, :, j].mean(axis=0)   # (n_feat,)
        lf_mean = Y_lf[:, :, j].mean(axis=0)

        # colour by feature type
        n_zonal = len(feat_lbs) - len(DEFAULT_REGIONS) - 1
        colors  = (["#4dac26"] * n_zonal +
                   ["#984ea3"] * len(DEFAULT_REGIONS) +
                   ["#ff7f00"])

        for fi, (lf_v, hf_v, c, lbl) in enumerate(
                zip(lf_mean, hf_mean, colors, feat_lbs)):
            ax.scatter(lf_v, hf_v, color=c, s=55, zorder=3)
            ax.annotate(lbl, (lf_v, hf_v), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", color="#444")

        # fit line through origin (ρ estimate)
        slope = np.dot(lf_mean, hf_mean) / np.dot(lf_mean, lf_mean)
        r, _  = pearsonr(lf_mean, hf_mean)
        xlim  = np.array([min(lf_mean.min(), hf_mean.min()) * 0.95,
                          max(lf_mean.max(), hf_mean.max()) * 1.05])
        ax.plot(xlim, slope * xlim, "k--", lw=1.2,
                label=f"fit through origin  slope={slope:.2f}")
        ax.plot(xlim, xlim, color="#aaa", lw=0.8, ls=":", label="y=x")
        ax.set_xlim(xlim)

        ax.set_xlabel("Low-res mean", fontsize=10)
        ax.set_ylabel("High-res mean", fontsize=10)
        ax.set_title(f"{var}   r={r:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # legend for feature colours
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4dac26",
               markersize=8, label="Zonal bands"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#984ea3",
               markersize=8, label="Regions"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f00",
               markersize=8, label="Global"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    save_fig(fig, pdf, png_dir, "02_lf_hf_scatter")


# ── page 3: sensitivity heatmaps (one page per variable) ─────────────────────
def page_sensitivity(pdf, X_hf, Y_hf, X_lf, Y_lf, var_names, feat_lbs, p_lbs,
                     lf_label="LF", hf_label="HF", png_dir=None):
    for j, var in enumerate(var_names):
        n_feat   = Y_hf.shape[1]
        n_params = X_hf.shape[1]

        def corr_matrix(X, Y_j):
            Xarr = np.asarray(X, dtype=float)
            C = np.zeros((n_params, n_feat))
            for pi in range(n_params):
                for fi in range(n_feat):
                    r, _ = pearsonr(Xarr[:, pi], Y_j[:, fi])
                    C[pi, fi] = r
            return C

        C_hf = corr_matrix(X_hf, Y_hf[:, :, j])
        C_lf = corr_matrix(X_lf, Y_lf[:, :, j])
        C_diff = C_hf - C_lf

        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)
        axes = [fig.add_subplot(gs[i]) for i in range(3)]

        kw = dict(aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                  interpolation="nearest")
        for ax, C, title in zip(axes[:2], [C_lf, C_hf],
                                 [f"Low-res {lf_label} ({var})", f"High-res {hf_label} ({var})"]):
            im = ax.imshow(C, **kw)
            ax.set_xticks(range(n_feat))
            ax.set_xticklabels(feat_lbs, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(n_params))
            ax.set_yticklabels(p_lbs, fontsize=7)
            ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=axes[1], fraction=0.03, label="Pearson r")

        kw_diff = dict(aspect="auto", cmap="PuOr", vmin=-1, vmax=1,
                       interpolation="nearest")
        im2 = axes[2].imshow(C_diff, **kw_diff)
        axes[2].set_xticks(range(n_feat))
        axes[2].set_xticklabels(feat_lbs, rotation=60, ha="right", fontsize=7)
        axes[2].set_yticks(range(n_params))
        axes[2].set_yticklabels(p_lbs, fontsize=7)
        axes[2].set_title(f"High-res − Low-res  ({var})", fontsize=11, fontweight="bold")
        fig.colorbar(im2, ax=axes[2], fraction=0.03, label="Δ Pearson r")

        fig.suptitle(
            f"Parameter Sensitivity: Pearson r(param, ZRG feature) — {var}\n"
            "Large differences mean low-res and high-res respond differently to the same parameter",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        save_fig(fig, pdf, png_dir, f"03_sensitivity_{var}")


# ── page 4: spread ratio ─────────────────────────────────────────────────────
def page_spread(pdf, Y_hf, Y_lf, var_names, feat_lbs, png_dir=None):
    n_feat = Y_hf.shape[1]
    n_vars = len(var_names)
    x      = np.arange(n_feat)

    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3.5 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    fig.suptitle("Ensemble Spread Ratio  High-res std / Low-res std per ZRG Feature\n"
                 "(ratio ≈ 1 means same variability; >> 1 or << 1 means different sensitivity)",
                 fontsize=12, fontweight="bold", y=1.01)

    for j, (var, ax) in enumerate(zip(var_names, axes)):
        hf_std = Y_hf[:, :, j].std(axis=0)
        lf_std = Y_lf[:, :, j].std(axis=0)
        ratio  = hf_std / (lf_std + 1e-30)

        ax.bar(x, ratio, color=np.where(ratio > 1, C_HF, C_LF), alpha=0.75)
        ax.axhline(1.0, color="black", lw=1.2, ls="--")
        ax.set_ylabel(f"{var}\nHigh-res/Low-res std ratio", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        n_zonal = n_feat - len(DEFAULT_REGIONS) - 1
        ax.axvline(n_zonal - 0.5, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(n_feat - 1.5, color="black", lw=0.8, ls="--", alpha=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feat_lbs, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    save_fig(fig, pdf, png_dir, "06_spread_ratio")


# ── page 5: per-member scatter (ensemble overlap) ────────────────────────────
def page_ensemble_overlap(pdf, Y_hf, Y_lf, var_names, feat_lbs, png_dir=None):
    """
    For a few representative features, plot all HF and LF member values to
    see if the ensemble ranges overlap — a prerequisite for AR1 to work.
    """
    n_feat  = Y_hf.shape[1]
    n_zonal = n_feat - len(DEFAULT_REGIONS) - 1

    # Pick representative features: tropical, midlat, polar, one region
    candidates = {
        "tropical (≈0°)": n_zonal // 2,
        "midlat (≈45°)":  int(n_zonal * 0.7),
        "polar (≈75°)":   n_zonal - 1,
        "tropical_land":  n_zonal + DEFAULT_REGIONS.index("tropical_land"),
        "global":         n_feat - 1,
    }
    sel = list(candidates.items())

    n_vars = len(var_names)
    fig, axes = plt.subplots(len(sel), n_vars,
                             figsize=(5 * n_vars, 3 * len(sel)),
                             squeeze=False)

    fig.suptitle("Ensemble Member Distributions: Low-res vs High-res\n"
                 "(overlap means AR1 can interpolate; no overlap means mismatch)",
                 fontsize=12, fontweight="bold", y=1.01)

    for ri, (feat_name, fi) in enumerate(sel):
        for j, var in enumerate(var_names):
            ax = axes[ri][j]
            hf_vals = Y_hf[:, fi, j]
            lf_vals = Y_lf[:, fi, j]

            ax.hist(lf_vals, bins=20, alpha=0.5, color=C_LF,
                    label=f"Low-res n={len(lf_vals)}", density=True)
            ax.hist(hf_vals, bins=15, alpha=0.5, color=C_HF,
                    label=f"High-res n={len(hf_vals)}", density=True)
            ax.axvline(lf_vals.mean(), color=C_LF, lw=1.5, ls="--")
            ax.axvline(hf_vals.mean(), color=C_HF, lw=1.5, ls="--")

            if ri == 0:
                ax.set_title(var, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(feat_name, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, pdf, png_dir, "07_ensemble_overlap")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LF vs HF comparison plots")
    parser.add_argument("--hf",  default=DEFAULT_HF,  help="HF preprocess dir")
    parser.add_argument("--lf",  default=DEFAULT_LF,  help="LF preprocess dir")
    parser.add_argument("--hf-label", default="HF", help="Label for HF in plot titles")
    parser.add_argument("--lf-label", default="LF", help="Label for LF in plot titles")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output PDF path")
    parser.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--n-zonal-orig", type=int, default=N_ZONAL_ORIG)
    parser.add_argument("--dropped", nargs="+", type=float, default=DROPPED_BANDS)
    args = parser.parse_args()

    print(f"Loading HF data from {args.hf} ...")
    X_hf, Y_hf, mask_hf = load(args.hf)
    print(f"  HF: X={X_hf.shape}, Y={Y_hf.shape}")

    print(f"Loading LF data from {args.lf} ...")
    X_lf, Y_lf, mask_lf = load(args.lf)
    print(f"  LF: X={X_lf.shape}, Y={Y_lf.shape}")

    f_lbs = feat_labels(args.n_zonal_orig, args.dropped, args.regions)
    p_lbs = param_labels()
    print(f"  {len(f_lbs)} ZRG features,  {len(p_lbs)} parameters")

    # PNG output directory alongside the PDF
    out_path = Path(args.out)
    png_dir  = out_path.parent / out_path.stem
    png_dir.mkdir(parents=True, exist_ok=True)
    print(f"PNG directory: {png_dir}")

    with PdfPages(args.out) as pdf:
        info = pdf.infodict()
        info["Title"] = "LF vs HF Data Comparison"

        print("Page 1: ZRG profiles ...")
        page_profiles(pdf, X_hf, Y_hf, X_lf, Y_lf, args.vars, f_lbs,
                      lf_label=args.lf_label, hf_label=args.hf_label,
                      png_dir=png_dir)

        print("Page 2: LF vs HF scatter ...")
        page_scatter(pdf, Y_hf, Y_lf, args.vars, f_lbs, png_dir=png_dir)

        print("Pages 3-5: Sensitivity heatmaps (one per variable) ...")
        page_sensitivity(pdf, X_hf, Y_hf, X_lf, Y_lf, args.vars, f_lbs, p_lbs,
                         lf_label=args.lf_label, hf_label=args.hf_label,
                         png_dir=png_dir)

        print("Page 6: Spread ratio ...")
        page_spread(pdf, Y_hf, Y_lf, args.vars, f_lbs, png_dir=png_dir)

        print("Page 7: Ensemble overlap histograms ...")
        page_ensemble_overlap(pdf, Y_hf, Y_lf, args.vars, f_lbs, png_dir=png_dir)

    print(f"\nSaved {args.out}")
    print(f"Saved PNGs in {png_dir}/")


if __name__ == "__main__":
    main()
