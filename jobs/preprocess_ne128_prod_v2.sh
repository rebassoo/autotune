#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -J prep_ne128_v2
#SBATCH -o logs/preprocess_ne128_prod_v2_%j.log
#SBATCH -e logs/preprocess_ne128_prod_v2_%j.log

# Stage 1 preprocessing for the 20260608 production PPE (v2 ensembles).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export PREPROCESS_WORKERS=32

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_ne128_prod_v2_annual.yaml \
    --stage 1

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
