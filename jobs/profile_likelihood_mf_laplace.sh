#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N profile_lik_mf_laplace
#PBS -o logs/profile_likelihood_mf_laplace.out
#PBS -e logs/profile_likelihood_mf_laplace.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/profile_likelihood_mf_laplace_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_laplace
PREPROCESS_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/preprocess_output_ne128_prod

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/profile_likelihood.py \
    --run-dir "${RUN_DIR}" \
    --gp "${RUN_DIR}/mf_gp_trained.pkl" \
    --preprocess-dir "${PREPROCESS_DIR}" \
    --out "${RUN_DIR}/diagnostics/profile_likelihood.pkl" \
    --max-workers 32

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
