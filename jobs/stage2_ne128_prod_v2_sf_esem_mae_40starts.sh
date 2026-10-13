#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J ne128_v2_esem_mae
#SBATCH -o logs/stage2_ne128_prod_v2_sf_esem_mae_40starts_%j.log
#SBATCH -e logs/stage2_ne128_prod_v2_sf_esem_mae_40starts_%j.log

# Single-fidelity stage 2 on the ne128 (high-resolution) production PPE v2:
# 5-fold surrogate skill evaluation, ESEm/GPflow GP training on all 218
# members, then 40-start optimization under the plain weighted-MAE cost.
#
# Unlike the multi-fidelity path, run_stage2 has no --skip-gp/--skip-optimize,
# so k-fold, training and optimization all run in this one job.
#
# Requires stage 1 for configs/perlmutter_ne128_prod_v2_annual.yaml to have
# completed — this config reads the same preprocess_dir.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# ESEm/GPflow cannot fork, so the single-fidelity optimizer parallelises the
# 40 starts across 40 threads in one process (optimize.max_workers), one wave
# rather than 32-then-8. Those threads share the process's BLAS/TF pool, so
# one BLAS thread each keeps the node from being oversubscribed.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_ne128_prod_v2_sf_esem_mae_annual.yaml \
    --stage 2 \
    --n-xstarts 40

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
