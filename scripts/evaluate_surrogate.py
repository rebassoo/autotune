"""
Standalone surrogate skill evaluation: R² and RMSE per variable, all folds.

Requires kfold_data.pkl in cfg.paths.preprocess_dir (produced by
preprocessing/04_kfold_and_stack.py or run_two_stage.py --stage 1).

Usage:
    python scripts/evaluate_surrogate.py --config configs/scream_autocal.yaml
    python scripts/evaluate_surrogate.py --config configs/scream_autocal.yaml --fold 1
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autotune_gp.config import load_config
from autotune_gp.evaluate import run_kfold_evaluation, evaluate_fold, print_fold_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Evaluate a single fold (0-indexed). Default: all folds.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    kfold_pkl = Path(cfg.paths.preprocess_dir) / "kfold_data.pkl"

    if not kfold_pkl.exists():
        raise FileNotFoundError(
            f"{kfold_pkl} not found. Run preprocessing/04_kfold_and_stack.py "
            "or run_two_stage.py --stage 1 first."
        )

    with open(kfold_pkl, "rb") as f:
        kfold_data = pickle.load(f)

    folds = kfold_data["folds"]
    if args.fold is not None:
        folds = [folds[args.fold]]

    run_kfold_evaluation(folds, train_gp=cfg.runtime.train_gp)


if __name__ == "__main__":
    main()
