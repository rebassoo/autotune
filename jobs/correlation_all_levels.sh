#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J corr_3levels
#SBATCH -o logs/correlation_all_levels_%j.log
#SBATCH -e logs/correlation_all_levels_%j.log

cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/plot_param_output_correlation.py \
    --level "ne32=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne32_prod_v2" \
    --level "ne128=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2" \
    --level "ne256=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_all" \
    --out-dir docs --name param_output_correlation_3levels
