"""
Pearson correlation between the 19 input parameters and the simulated ZRG
outputs, computed independently at each resolution.

This is a property of the ensembles themselves, not of any surrogate: for each
level it correlates the parameter values across members against that member's
simulated output. Comparing the panels shows whether a parameter influences
the model the same way at ne32, ne128 and ne256 -- the assumption the
multi-fidelity model leans on.

Correlation is computed per (parameter, variable, ZRG location). The figure
shows the global-mean location; the CSV carries every location.

Sample size matters and differs enormously between levels (ne32 ~984 vs ne256
~16), so each panel is annotated with n and the |r| needed for p < 0.05, and
cells that do not reach it are drawn faded. With ~16 members only very strong
correlations are distinguishable from noise.

Usage:
    python scripts/plot_param_output_correlation.py \
        --level "ne32=.../preprocess_output_ne32_prod_v2" \
        --level "ne128=.../preprocess_output_ne128_prod_v2" \
        --level "ne256=.../preprocess_output_ne256_frontier" \
        --out-dir docs
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import stats

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
# Diverging blue <-> red with a neutral gray midpoint (dataviz reference palette).
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [
    "#184f95", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#9e2b2a",
])


def _parse(spec: str):
    if "=" not in spec:
        raise ValueError(f"Expected 'Label=/path/to/preprocess_dir', got {spec!r}")
    lab, path = spec.rsplit("=", 1)
    return lab.strip(), path.strip()


def _load(d: str):
    d = Path(d)
    with open(d / "gp_proj.pkl", "rb") as f:
        gp = pickle.load(f)
    X, Y = gp["X_train"], np.asarray(gp["Y_train"], dtype=float)
    names = list(X.columns) if hasattr(X, "columns") else None
    labels = None
    mp = d / "column_mask.pkl"
    if mp.exists():
        with open(mp, "rb") as f:
            labels = pickle.load(f).get("valid_feat_indices")
    return np.asarray(X, dtype=float), Y, names, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--variables", default="PCP,OSR,OLR,TLWP")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="param_output_correlation")
    args = ap.parse_args()

    var_names = [v.strip() for v in args.variables.split(",")]
    levels = []
    for spec in args.level:
        lab, d = _parse(spec)
        X, Y, pnames, _ = _load(d)
        levels.append({"label": lab, "X": X, "Y": Y, "pnames": pnames, "dir": d})
        print(f"{lab}: X{X.shape}  Y{Y.shape}")

    pnames = next((l["pnames"] for l in levels if l["pnames"]), None)
    if pnames is None:
        pnames = [f"p{i}" for i in range(levels[0]["X"].shape[1])]
    n_p = len(pnames)
    n_feat = levels[0]["Y"].shape[1]
    for l in levels:
        if l["Y"].shape[1] != n_feat:
            raise SystemExit(f"{l['label']} has {l['Y'].shape[1]} features, expected {n_feat}; "
                             "levels must share the ZRG layout.")

    # ---- correlations: (n_levels, n_params, n_feat, n_vars) ----------------
    R = np.full((len(levels), n_p, n_feat, len(var_names)), np.nan)
    P = np.full_like(R, np.nan)
    for li, l in enumerate(levels):
        n = l["X"].shape[0]
        for pi in range(n_p):
            x = l["X"][:, pi]
            # ptp, not np.allclose: its default atol=1e-8 exceeds the whole
            # spread of some variables in SI units (PCP ~3e-8 m/s), which
            # silently discards a real signal as "constant".
            if np.ptp(x[np.isfinite(x)]) == 0:
                continue                      # parameter never varied at this level
            for fi in range(n_feat):
                for vi in range(len(var_names)):
                    y = l["Y"][:, fi, vi]
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 3 or np.ptp(y[ok]) == 0 or np.ptp(x[ok]) == 0:
                        continue
                    r, p = stats.pearsonr(x[ok], y[ok])
                    R[li, pi, fi, vi], P[li, pi, fi, vi] = r, p

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "n_members", "parameter", "zrg_index", "variable", "pearson_r", "p_value"])
        for li, l in enumerate(levels):
            for pi in range(n_p):
                for fi in range(n_feat):
                    for vi, v in enumerate(var_names):
                        if np.isfinite(R[li, pi, fi, vi]):
                            w.writerow([l["label"], l["X"].shape[0], pnames[pi], fi, v,
                                        f"{R[li, pi, fi, vi]:.6f}", f"{P[li, pi, fi, vi]:.3e}"])
    print(f"Saved {csv_path}")

    # ---- figure: global-mean location (last ZRG column of the block) -------
    gi = n_feat - 1
    nl = len(levels)
    fig, axes = plt.subplots(1, nl, figsize=(2.6 + 3.7 * nl, 0.42 * n_p + 2.6),
                             facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(wspace=0.16)
    for li, (ax, l) in enumerate(zip(axes[0], levels)):
        n = l["X"].shape[0]
        r_crit = _r_crit(n)
        vals = R[li, :, gi, :]
        sig = P[li, :, gi, :] < 0.05
        ax.imshow(np.where(np.isfinite(vals), vals, 0.0), cmap=DIVERGING, vmin=-1, vmax=1,
                  aspect="auto")
        # Fade cells that do not reach p < 0.05 so noise does not read as signal.
        ax.imshow(np.where(sig, np.nan, 1.0), cmap=matplotlib.colors.ListedColormap([SURFACE]),
                  alpha=0.62, aspect="auto", vmin=0, vmax=1)
        for pi in range(n_p):
            for vi in range(len(var_names)):
                v = vals[pi, vi]
                if not np.isfinite(v):
                    continue
                ax.text(vi, pi, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color=TEXT if sig[pi, vi] else MUTED,
                        weight="bold" if sig[pi, vi] else "normal")
        ax.set_xticks(range(len(var_names)))
        ax.set_xticklabels(var_names, fontsize=10)
        ax.set_yticks(range(n_p))
        ax.set_yticklabels(pnames if li == 0 else [""] * n_p, fontsize=9)
        ax.set_title(f"{l['label']}\nn = {n} members\n|r| > {r_crit:.2f} for p < 0.05",
                     fontsize=11, color=TEXT, pad=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_color(MUTED)

    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cb.set_label("Pearson r  (parameter vs global-mean output)", fontsize=10, color=TEXT2)
    cb.outline.set_edgecolor(MUTED)
    fig.suptitle("Parameter–output correlation by resolution  (global mean; "
                 "faded = not significant at p < 0.05)", fontsize=13, y=0.99)
    out = out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


def _r_crit(n: int, alpha: float = 0.05) -> float:
    """|r| needed for a two-sided p < alpha at sample size n."""
    if n < 3:
        return float("nan")
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(t / np.sqrt(t ** 2 + n - 2))


if __name__ == "__main__":
    main()
