#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 20:00:00
#SBATCH -J ne256_cold_holdout
#SBATCH -o logs/holdout_ne256_3level_cold_%j.log
#SBATCH -e logs/holdout_ne256_3level_cold_%j.log

# Cold control for the STANDARD holdout: identical folds (seed 42) and identical
# data to job 58871667, but no --warm-start-gp, so all 47 hyperparameters per
# model are fitted from scratch. Difference vs 0.828/0.930/0.928/0.956 is the
# warm start alone.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

echo "Start: $(date)"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/evaluate_nfidelity_holdout.py \
    --level-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne32_prod_v2 \
    --level-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
    --top-dir   /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_all \
    --scalers-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
    --top-kernel ard \
    --folds 5 --workers 5 \
    --out-dir /pscratch/sd/r/rebassoo/autotune/results_ne256_3level_holdout_COLD
END=$(date +%s); echo "End: $(date)"; echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
