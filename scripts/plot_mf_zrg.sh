#!/bin/bash
set -euo pipefail

WORK=~/work/2026_05_11_MultiFidelityAutotuning
AUTOTUNE=$(dirname "$0")/..

python "${AUTOTUNE}/scripts/plot_multifidelity_zrg.py" \
    --config-hf "${AUTOTUNE}/configs/aurora_ne256_remapped_ne32pg2_annual.yaml" \
    --config-lf "${AUTOTUNE}/configs/perlmutter_ne32_annual.yaml" \
    --preprocess-dir-hf "${WORK}/preprocess_output_ne256_remapped_ne32pg2" \
    --preprocess-dir-lf "${WORK}/preprocess_output_ne32" \
    --out-dir "${WORK}/plots"
