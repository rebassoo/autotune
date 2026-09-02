#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_ne128_aurora_mae_esem
#PBS -o logs/stage2_ne128_aurora_40starts_mae_esem_matched.out
#PBS -e logs/stage2_ne128_aurora_40starts_mae_esem_matched.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_ne128_aurora_40starts_mae_esem_matched_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne128_prod_aurora_40starts_mae_esem_matched

# SF ne128 ESEm/GPflow plain-MAE run, weight-matched (PCP/OSR/OLR/TLWP =
# 0.3/0.1/0.3/0.3) -- see configs/aurora_ne128_prod_mae_esem_matched_annual.yaml.
# No existing ESEm checkpoint to reuse (unlike the GPy runs), so this trains
# the single shared-kernel ESEm GP from scratch before optimizing -- expect
# this to take noticeably longer than the GPy MAE run (which reused a cached
# checkpoint, ~8min total).
mkdir -p "${OUTPUT_DIR}"

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_ne128_prod_mae_esem_matched_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
