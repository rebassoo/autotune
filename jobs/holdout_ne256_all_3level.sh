#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -J ne256_all_holdout
#SBATCH -o logs/holdout_ne256_all_3level_%j.log
#SBATCH -e logs/holdout_ne256_all_3level_%j.log

# Full ne256 holdout: 4 folds over the top level, all 88 features, lower levels
# warm-started from the trained 2-level GP and frozen. Scores two baselines on
# the same held-out members (2-level GP; ne128 simulation at the same params).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# 5 fold workers x 12 BLAS threads across the node.
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

echo "Start: $(date)"
START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/evaluate_nfidelity_holdout.py \
    --level-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne32_prod_v2 \
    --level-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
    --top-dir   /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_all \
    --scalers-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
    --warm-start-gp /pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_v2/mf_gp_trained.pkl \
    --top-kernel ard \
    --folds 5 --workers 5 \
    --out-dir /pscratch/sd/r/rebassoo/autotune/results_ne256_all_3level_holdout
END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
