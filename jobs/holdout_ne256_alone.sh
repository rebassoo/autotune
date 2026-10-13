#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -J ho_ne256_alone
#SBATCH -o logs/holdout_ne256_alone_%j.log
#SBATCH -e logs/holdout_ne256_alone_%j.log

# "ne256 alone" row of the surrogate-performance table: single-fidelity GPy on
# the ne256 members only, on exactly the folds/scalers/metric the 3-level
# holdout uses. Previously done as an ad-hoc rerun; now a script.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

echo "Start: $(date)"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/evaluate_ne256_alone_holdout.py \
    --top-dir     /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_all \
    --scalers-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
    --folds 5 --workers 5 \
    --out-dir /pscratch/sd/r/rebassoo/autotune/results_ne256_alone_holdout
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
