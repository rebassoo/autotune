#!/bin/bash

cd /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/autotune

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/plot_lf_hf_comparison.py \
    --hf /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/preprocess_output_ne128_prod \
    --lf /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/preprocess_output_ne32_prod \
    --hf-label ne128 --lf-label ne32 \
    --out lf_hf_comparison_ne128_ne32_prod.pdf \
    --vars PCP OSR OLR TLWP
