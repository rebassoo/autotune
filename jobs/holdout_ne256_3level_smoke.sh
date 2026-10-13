#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH -t 01:00:00
#SBATCH -J ne256_3lvl_smoke
#SBATCH -o logs/holdout_ne256_3level_smoke_%j.log
#SBATCH -e logs/holdout_ne256_3level_smoke_%j.log

# Smoke test of the 3-level path end to end on 2 ZRG features only, so a
# mistake surfaces in minutes rather than after hours of AR training.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Start: $(date)"
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/evaluate_nfidelity_holdout.py \
    --level-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne32_prod_v2 \
    --level-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
    --top-dir   /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_frontier \
    --scalers-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
    --warm-start-gp /pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_v2/mf_gp_trained.pkl \
    --folds 2 --limit-features 2 --workers 2 \
    --out-dir /pscratch/sd/r/rebassoo/autotune/results_ne256_3level_smoke
echo "End: $(date)"
