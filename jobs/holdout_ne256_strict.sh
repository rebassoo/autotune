#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J ne256_strict
#SBATCH -o logs/holdout_ne256_strict_%j.log
#SBATCH -e logs/holdout_ne256_strict_%j.log

# STRICT holdout: the ne32 and ne128 members sitting at each held-out ne256
# parameter point are removed from training too, so no level has seen that
# point. This is the regime the optimizer actually works in.
# No warm start: one fitted on all ne32/ne128 members would carry the held-out
# points in its hyperparameters. That means cold training, which is slow.
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
    --strict-holdout --folds 5 --workers 5 \
    --out-dir /pscratch/sd/r/rebassoo/autotune/results_ne256_strict_holdout
END=$(date +%s); echo "End: $(date)"; echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
