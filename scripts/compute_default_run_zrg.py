"""
Retroactively compute default_run_zrg.pkl for a preprocess_dir that was
built before default-run ZRG computation existed in run_stage1.

Loads the saved run_list.pkl + column_mask.pkl (already contain everything
needed — no need to re-scan the PPE ensemble) and runs the single default
directory through the same ZRG machinery as the training ensemble.

Usage:
    python scripts/compute_default_run_zrg.py --config configs/perlmutter_ne128_prod_annual.yaml
    python scripts/compute_default_run_zrg.py --config configs/perlmutter_ne32_prod_annual.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import load_config
from preprocessing.pipeline import compute_default_run_zrg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    pp = cfg.preprocess
    var_names = list(pp.variables.keys())
    out_dir = Path(cfg.paths.preprocess_dir)

    with open(out_dir / "run_list.pkl", "rb") as f:
        run_list = pickle.load(f)
    with open(out_dir / "column_mask.pkl", "rb") as f:
        column_mask = pickle.load(f)

    print(f"=== Computing default-run ZRG for {out_dir} ===")
    result = compute_default_run_zrg(run_list, column_mask, pp, var_names)
    if result is None:
        print("Failed to identify a unique default run — see warning above.")
        sys.exit(1)

    with open(out_dir / "default_run_zrg.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"Saved default_run_zrg.pkl  (default_name={result['default_name']}, "
          f"Y_default_ZRG shape={result['Y_default_ZRG'].shape})")


if __name__ == "__main__":
    main()
