"""
Reconstruct normranked_LH_sampling_base10.json from scream_input.yaml files.

Each ensemble member stores its parameter values in:
    <case_dir>/run/data/scream_input.yaml

The output format matches the existing JSON used by the autotune pipeline:
    list-of-lists, indexed by member number (0-based),
    each inner list has 19 floats in the same order as param_physical_bounds
    in the yaml configs.

Usage:
    python scripts/build_params_json.py \\
        --ppe-dir /pscratch/sd/b/beydoun/e3sm_scratch/pm-gpu/ne32_ppe_prod \\
        --output normranked_ne32_prod.json

    python scripts/build_params_json.py \\
        --ppe-dir /pscratch/sd/b/beydoun/e3sm_scratch/pm-gpu/ne128_ppe_prod \\
        --output normranked_ne128_prod.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

# Parameter names in the exact order used by the autotune configs
# (matches param_physical_bounds order in perlmutter_ne32_annual.yaml)
PARAM_ORDER = [
    "thl2tune",
    "qw2tune",
    "length_fac",
    "c_diag_3rd_mom",
    "coeff_kh",
    "coeff_km",
    "lambda_low",
    "lambda_high",
    "spa_ccn_to_nc_factor",
    "cldliq_to_ice_collection_factor",
    "rain_to_ice_collection_factor",
    "accretion_prefactor",
    "deposition_nucleation_exponent",
    "max_total_ni",
    "ice_sedimentation_factor",
    "rain_selfcollection_breakup_diameter",
    "autoconversion_prefactor",
    "autoconversion_qc_exponent",
    "autoconversion_radius",
]


def _member_index(dirname: str) -> int:
    """Extract 0-based integer index from a case directory name.

    Handles names ending in '.m042' or '.m1002'.
    """
    last = dirname.rsplit(".", 1)[-1]
    if last.startswith("m"):
        return int(last[1:])
    raise ValueError(f"Cannot parse member index from: {dirname!r}")


def _extract_params(yaml_path: str) -> list[float]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    mac  = cfg["eamxx"]["physics"]["mac_aero_mic"]
    shoc = mac["shoc"]
    p3   = mac["p3"]
    merged = {**shoc, **p3}
    missing = [p for p in PARAM_ORDER if p not in merged]
    if missing:
        raise KeyError(f"Parameters missing from {yaml_path}: {missing}")
    return [float(merged[p]) for p in PARAM_ORDER]


def build(ppe_dir: str, output: str) -> None:
    entries = [
        e for e in os.listdir(ppe_dir)
        if os.path.isdir(os.path.join(ppe_dir, e)) and ".m" in e
    ]

    params: dict[int, list[float]] = {}
    failed: list[str] = []

    for entry in sorted(entries):
        try:
            idx = _member_index(entry)
        except ValueError:
            continue

        yaml_path = os.path.join(ppe_dir, entry, "run", "data", "scream_input.yaml")
        if not os.path.exists(yaml_path):
            print(f"  SKIP {entry}: no scream_input.yaml", file=sys.stderr)
            failed.append(entry)
            continue

        try:
            params[idx] = _extract_params(yaml_path)
        except Exception as e:
            print(f"  SKIP {entry}: {e}", file=sys.stderr)
            failed.append(entry)

    if not params:
        raise RuntimeError("No valid members found.")

    max_idx = max(params)
    n_members = max_idx + 1
    print(f"Found {len(params)} members, max index {max_idx} → array size {n_members}")
    if failed:
        print(f"  {len(failed)} members skipped (see stderr)")

    # Build list-of-lists; fill gaps (if any) with None so indices stay aligned
    result: list = [None] * n_members
    for idx, vals in params.items():
        result[idx] = vals

    n_gaps = sum(1 for v in result if v is None)
    if n_gaps:
        print(f"  WARNING: {n_gaps} index gap(s) in member numbering (None entries in JSON)")

    with open(output, "w") as f:
        json.dump(result, f, indent=None, separators=(",", ":"))
    print(f"Saved {len(params)} entries → {output}")

    # Sanity check: verify index 0 matches first valid member
    print(f"\nFirst entry (index 0): {result[0]}")
    print(f"Param names:           {PARAM_ORDER}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ppe-dir", required=True,
                   help="Path to PPE root directory (contains case dirs)")
    p.add_argument("--output", required=True,
                   help="Output JSON file path")
    args = p.parse_args()
    build(args.ppe_dir, args.output)


if __name__ == "__main__":
    main()
