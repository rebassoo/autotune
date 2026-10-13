"""
Is ne256 systematically different from ne128 / ne32?

Compares the two resolutions member-by-member at IDENTICAL parameter settings
(the design is nested, so most ne256 members have a twin at the coarser
resolutions). Anything left is a resolution effect, not a parameter effect.

One scatter per variable per reference resolution: ne256 on y, the coarser run
on x, with the 1:1 line. Reported per panel:
    bias  = mean(ne256 - coarse)      systematic offset
    RMSD  = root mean square difference
    r     = Pearson correlation across members
    slope = ordinary least squares fit, to show a scale (not just offset) error

A per-location bias table is written alongside, since a global-mean bias can
hide compensating regional errors.

Usage:
    python scripts/plot_resolution_bias.py \
        --top   .../preprocess_output_ne256_all \
        --ref   "ne128=.../preprocess_output_ne128_prod_v2" \
        --ref   "ne32=.../preprocess_output_ne32_prod_v2" \
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
import numpy as np
from scipy import stats

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
POINT = "#2a78d6"

VARS = ["PCP", "OSR", "OLR", "TLWP"]
# PCP is carried in m/s; everything else is already in its natural unit.
SCALE = {"PCP": 8.64e7, "OSR": 1.0, "OLR": 1.0, "TLWP": 1.0}
UNITS = {"PCP": "mm/day", "OSR": "W/m²", "OLR": "W/m²", "TLWP": "kg/m²"}


def _load(d: str):
    d = Path(d)
    with open(d / "gp_proj.pkl", "rb") as f:
        g = pickle.load(f)
    return np.asarray(g["X_train"], dtype=float), np.asarray(g["Y_train"], dtype=float)


def _parse(spec: str):
    lab, path = spec.rsplit("=", 1)
    return lab.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--ref", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--labels-from", default=None,
                    help="projection_data*.pkl for ZRG location names (for the CSV).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="resolution_bias")
    ap.add_argument("--mode", choices=["global", "regions"], default="global",
                    help="'global': one panel per variable, global mean only. "
                         "'regions': one figure per variable, a panel per named "
                         "region (plus GLOBAL), rows = reference resolution.")
    args = ap.parse_args()

    Xt, Yt = _load(args.top)
    refs = [(lab, *_load(d)) for lab, d in (_parse(s) for s in args.ref)]
    n_feat = Yt.shape[1]
    gi = n_feat - 1                      # global mean is the last ZRG column

    labels = [str(i) for i in range(n_feat)]
    if args.labels_from:
        import glob
        with open(sorted(glob.glob(args.labels_from))[-1], "rb") as f:
            labels = list(pickle.load(f)["zrg_labels"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = []

    def _is_named(lb):
        try:
            float(lb); return False
        except ValueError:
            return True

    SHORT = {"poles": "poles", "extratropical_land": "extratrop land",
             "extratropical_ocean": "extratrop ocean", "tropical_land": "trop land",
             "ascending_tropical_ocean": "asc trop ocean",
             "descending_tropical_ocean": "desc trop ocean", "global": "GLOBAL"}

    def _panel(ax, x, y, xlab, ylab, title):
        """One matched-member scatter with the 1:1 line and a fit."""
        bias = float(np.mean(y - x)); rmsd = float(np.sqrt(np.mean((y - x) ** 2)))
        r = float(stats.pearsonr(x, y)[0]) if len(x) > 2 else np.nan
        slope, intercept = np.polyfit(x, y, 1)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo) if hi > lo else 0.1
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--", zorder=1)
        ax.plot(np.array(lim), slope * np.array(lim) + intercept, color="#eb6834",
                lw=1.5, zorder=2)
        ax.plot(x, y, "o", ms=4.5, mfc=POINT, mec="white", mew=0.7, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        pct = 100 * bias / np.mean(x) if np.mean(x) != 0 else np.nan
        ax.text(0.04, 0.96, f"bias {bias:+.3g} ({pct:+.1f}%)\nr {r:.3f}\nslope {slope:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color=TEXT2, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                       edgecolor=GRID))

    if args.mode == "regions":
        ridx = [i for i, lb in enumerate(labels) if _is_named(lb)]
        matched = []
        for lab, Xr, Yr in refs:
            d = np.abs(Xt[:, None, :] - Xr[None, :, :]).max(axis=2)
            pr = [(i, int(np.argmin(d[i]))) for i in range(len(Xt)) if d[i].min() <= 1e-9]
            matched.append((lab, Yr, [p[0] for p in pr], [p[1] for p in pr]))
            print(f"{lab}: {len(pr)} of {len(Xt)} ne256 members have an exact twin")
        for vi, v in enumerate(VARS):
            sc = SCALE[v]
            fig, axes = plt.subplots(len(matched), len(ridx),
                                     figsize=(3.05 * len(ridx), 3.5 * len(matched)),
                                     facecolor=SURFACE, squeeze=False)
            for ri, (lab, Yr, ti, rj) in enumerate(matched):
                for ci, fi in enumerate(ridx):
                    _panel(axes[ri][ci], Yr[rj, fi, vi] * sc, Yt[ti, fi, vi] * sc,
                           f"{lab}  [{UNITS[v]}]", f"ne256  [{UNITS[v]}]",
                           f"{SHORT.get(labels[fi], labels[fi])}   (n={len(ti)})")
            fig.suptitle(f"{v}: ne256 vs coarser resolution at identical parameter settings, "
                         "by region\nrow 1 = vs ne128, row 2 = vs ne32   |   "
                         "dashed = 1:1, orange = least-squares fit", fontsize=13, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            out = out_dir / f"{args.name}_{v}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
            plt.close(fig)
            print(f"Saved {out}")
        return

    fig, axes = plt.subplots(len(refs), len(VARS),
                             figsize=(4.3 * len(VARS), 4.5 * len(refs)),
                             facecolor=SURFACE, squeeze=False)

    for ri, (lab, Xr, Yr) in enumerate(refs):
        # match members by exact parameter vector
        d = np.abs(Xt[:, None, :] - Xr[None, :, :]).max(axis=2)
        pairs = [(i, int(np.argmin(d[i]))) for i in range(len(Xt)) if d[i].min() <= 1e-9]
        ti = [p[0] for p in pairs]
        rj = [p[1] for p in pairs]
        print(f"{lab}: {len(pairs)} of {len(Xt)} ne256 members have an exact twin")

        for vi, v in enumerate(VARS):
            ax = axes[ri][vi]
            s = SCALE[v]
            x = Yr[rj, gi, vi] * s
            y = Yt[ti, gi, vi] * s
            bias = float(np.mean(y - x))
            rmsd = float(np.sqrt(np.mean((y - x) ** 2)))
            r = float(stats.pearsonr(x, y)[0]) if len(x) > 2 else np.nan
            slope, intercept = np.polyfit(x, y, 1)

            lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
            pad = 0.06 * (hi - lo)
            lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, color=MUTED, lw=1.3, ls="--", zorder=1)
            ax.plot(np.array(lim), slope * np.array(lim) + intercept,
                    color="#eb6834", lw=1.6, zorder=2)
            ax.plot(x, y, "o", ms=5.5, mfc=POINT, mec="white", mew=0.8, zorder=3)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel(f"{lab}  [{UNITS[v]}]", fontsize=10)
            ax.set_ylabel(f"ne256  [{UNITS[v]}]", fontsize=10)
            ax.set_title(f"{v}   ne256 vs {lab}", fontsize=12, pad=8)
            ax.grid(True, color=GRID, lw=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            pct = 100 * bias / np.mean(x) if np.mean(x) != 0 else np.nan
            ax.text(0.04, 0.96,
                    f"bias {bias:+.4g} ({pct:+.1f}%)\nRMSD {rmsd:.4g}\n"
                    f"r {r:.3f}\nslope {slope:.3f}\nn {len(x)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                    color=TEXT2,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=GRID))

            # per-location bias for the CSV
            for fi in range(n_feat):
                xx = Yr[rj, fi, vi] * s
                yy = Yt[ti, fi, vi] * s
                rows_csv.append([lab, v, labels[fi], len(xx),
                                 f"{np.mean(yy - xx):.6g}",
                                 f"{np.sqrt(np.mean((yy - xx) ** 2)):.6g}",
                                 f"{stats.pearsonr(xx, yy)[0]:.4f}" if len(xx) > 2 else ""])

    fig.suptitle("Resolution bias at identical parameter settings — "
                 "global mean, one point per matched member\n"
                 "dashed = 1:1,  orange = least-squares fit",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")

    csv_path = out_dir / f"{args.name}_by_location.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference", "variable", "zrg_location", "n_matched",
                    "bias_ne256_minus_ref", "rmsd", "pearson_r"])
        w.writerows(rows_csv)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
