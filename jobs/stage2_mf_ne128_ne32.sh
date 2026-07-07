#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J stage2_mf_ne128_ne32
#SBATCH -o logs/stage2_mf_ne128_ne32_%j.log
#SBATCH -e logs/stage2_mf_ne128_ne32_%j.log

cd /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/autotune

export PYTHONUNBUFFERED=1
export KFOLD_WORKERS=5
export OMP_NUM_THREADS=25
export MKL_NUM_THREADS=25

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf_ne128_ne32_prod_annual.yaml \
    --stage 2 \
    --output-dir /pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_6params \
    --top-k-params 6 #\
#    --skip-kfold

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
