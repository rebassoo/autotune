"""
Standalone script to plot PPE ZRG diagnostic figures from stage-1 output.

Loads ZRG data saved by run_two_stage.py --stage 1 and produces one PNG
with subplots for each observational variable.  Prefers zrg_data_clean.pkl
(post-column-drop) if it exists, falling back to zrg_data.pkl.

Obs overlay is shown automatically if obs data is present; silently omitted
if preprocessing was run in PPE-only mode.

Usage:
    python scripts/plot_ppe_zrg.py --config configs/aurora_ne256_annual.yaml
    python scripts/plot_ppe_zrg.py --config configs/aurora_ne256_annual.yaml \\
        --preprocess-dir /custom/path --suffix _v2
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import load_config
from preprocessing.plots import plot_ppe_zrg


def main():
    p = argparse.ArgumentParser(
        description="Plot PPE ZRG diagnostics from stage-1 preprocessing output."
    )
    p.add_argument("--config", required=True,
                   help="Path to YAML config file.")
    p.add_argument("--preprocess-dir", default=None,
                   help="Override cfg.paths.preprocess_dir.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for the PNG (default: preprocess_dir).")
    p.add_argument("--suffix", default="",
                   help="Optional suffix appended to the PNG filename.")
    args = p.parse_args()

    cfg = load_config(args.config)
    pp  = cfg.preprocess
    if pp is None:
        raise ValueError("Config must include a [preprocess] section.")
    if pp.snapshots is None:
        raise ValueError("plot_ppe_zrg requires the generic (snapshots) pipeline.")

    preprocess_dir = Path(args.preprocess_dir or cfg.paths.preprocess_dir)
    out_dir        = args.out_dir or str(preprocess_dir)
    var_names      = list(pp.variables.keys())

    # Prefer post-drop data; fall back to raw
    clean_pkl = preprocess_dir / "zrg_data_clean.pkl"
    raw_pkl   = preprocess_dir / "zrg_data.pkl"
    if clean_pkl.exists():
        pkl_path = clean_pkl
        print(f"Loading {clean_pkl}")
    elif raw_pkl.exists():
        pkl_path = raw_pkl
        print(f"Loading {raw_pkl}  (zrg_data_clean.pkl not found — may include dropped columns)")
    else:
        raise FileNotFoundError(
            f"No ZRG data found in {preprocess_dir}. "
            "Run 'python scripts/run_two_stage.py --stage 1' first."
        )

    with open(pkl_path, "rb") as f:
        zrg_result = pickle.load(f)

    plot_ppe_zrg(
        zrg_result=zrg_result,
        var_names=var_names,
        n_regions=len(cfg.data.regions_list),
        regions_list=cfg.data.regions_list,
        snapshots=pp.snapshots,
        out_dir=out_dir,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
