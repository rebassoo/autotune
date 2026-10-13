#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 08:00:00
#SBATCH -J mf3_ne256_train
#SBATCH -o logs/stage2a_mf3_ne256_train_%j.log
#SBATCH -e logs/stage2a_mf3_ne256_train_%j.log

# Part A: train the 3-level (ne32 -> ne128 -> ne256) AR GP and checkpoint it.
# Optimization is part B, which wants the opposite threading.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

echo "Start: $(date)"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf3_ne256_studentt_dof4_annual.yaml \
    --stage 2 --skip-optimize
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
