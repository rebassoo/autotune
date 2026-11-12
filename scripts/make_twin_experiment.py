"""
Set up a perfect-model (twin) experiment: one ne256 PPE member stands in for the
observations, optionally held out of the top level, so the optimizer's answer can be
compared with the parameters that really produced the target.

Writes, leaving the original preprocess directory untouched:
  <twin-dir>/          symlinks to the base preprocess_dir, with obs.pkl replaced by the
                       member's features and, with --hold-out, gp_proj.pkl without it
  <twin-dir>/truth.pkl member, parameters and target, for scoring
  <out-config>         the base config pointing at <twin-dir> and <output-dir>

    python scripts/make_twin_experiment.py \\
        --base-config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \\
        --truth-dir .../mf_prod_v2/preprocess_output_ne256_n85 \\
        --member PPEensemble_...20260910.m060 --hold-out \\
        --twin-dir .../twin_preprocess/m060_3level \\
        --output-dir .../results_twin_m060_3level \\
        --out-config .../twin_preprocess/m060_3level/config.yaml
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def pseudo_obs(obs_df: pd.DataFrame, y_member: np.ndarray) -> pd.DataFrame:
    """obs table holding a member's (n_feat, n_vars) features.

    Obs columns are one block per variable, each in feature order
    (autotune_gp.io.split_zrg_obs); columns and index are kept.
    """
    y = np.asarray(y_member, float)
    if obs_df.shape != (1, y.size):
        raise ValueError(f"obs shape {obs_df.shape} does not fit member features {y.shape}")
    return pd.DataFrame(y.T.ravel()[None, :], columns=obs_df.columns, index=obs_df.index)


def hold_out(gp_proj: dict, member: str) -> dict:
    """gp_proj without one member; every other member and key unchanged."""
    keep = np.asarray(gp_proj["X_train"].index != member)
    if keep.sum() != len(keep) - 1:
        raise ValueError(f"{member}: expected exactly one match, found {len(keep) - keep.sum()}")
    return {**gp_proj, "X_train": gp_proj["X_train"].loc[keep], "Y_train": np.asarray(gp_proj["Y_train"])[keep]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--truth-dir", required=True, help="Preprocess dir holding the member (ne256).")
    ap.add_argument("--member", required=True, help="Full member name, as in gp_proj X_train's index.")
    ap.add_argument("--hold-out", action="store_true",
                    help="Remove the member from the base config's top level (3-level runs).")
    ap.add_argument("--twin-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out-config", required=True)
    args = ap.parse_args()

    L = lambda p: pickle.load(open(p, "rb"))
    cfg = yaml.safe_load(open(args.base_config))
    src = Path(cfg["paths"]["preprocess_dir"])
    twin = Path(args.twin_dir)
    twin.mkdir(parents=True, exist_ok=True)

    truth = L(Path(args.truth_dir) / "gp_proj.pkl")
    i = list(truth["X_train"].index).index(args.member)
    y_member = np.asarray(truth["Y_train"], float)[i]
    base_gp = L(src / "gp_proj.pkl")
    if y_member.shape[1] != len(cfg["data"]["variables"]):
        raise ValueError(f"member has {y_member.shape[1]} variables, config {cfg['data']['variables']}")

    replaced = {"obs.pkl", "truth.pkl", "config.yaml"} | ({"gp_proj.pkl"} if args.hold_out else set())
    for f in src.iterdir():
        if f.name not in replaced:
            link = twin / f.name
            link.unlink(missing_ok=True)
            link.symlink_to(f)
    obs = pseudo_obs(L(src / "obs.pkl")["zrg_obs"], y_member)
    with open(twin / "obs.pkl", "wb") as fh:
        pickle.dump({"zrg_obs": obs}, fh)
    if args.hold_out:
        with open(twin / "gp_proj.pkl", "wb") as fh:
            pickle.dump(hold_out(base_gp, args.member), fh)
    with open(twin / "truth.pkl", "wb") as fh:
        pickle.dump({"member": args.member, "params": list(truth["X_train"].columns),
                     "x_phys": truth["X_train"].iloc[i].to_numpy(float), "y": y_member,
                     "variables": cfg["data"]["variables"], "held_out": args.hold_out}, fh)

    cfg["paths"]["preprocess_dir"] = str(twin)
    cfg["paths"]["output_dir"] = args.output_dir
    with open(args.out_config, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    n = len(L(twin / "gp_proj.pkl")["X_train"])
    print(f"{args.member}: top level {n} members (held out: {args.hold_out}); wrote {twin}, {args.out_config}")


if __name__ == "__main__":
    main()
