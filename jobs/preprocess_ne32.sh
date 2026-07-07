#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -J preprocess_ne32
#SBATCH -o logs/preprocess_ne32_%j.log
#SBATCH -e logs/preprocess_ne32_%j.log

cd /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/autotune

export PREPROCESS_WORKERS=32

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_ne32_prod_annual.yaml \
    --stage 1

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
