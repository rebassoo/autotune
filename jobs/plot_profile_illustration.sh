#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N plot_profile_illustration
#PBS -o logs/plot_profile_illustration.out
#PBS -e logs/plot_profile_illustration.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/plot_profile_illustration_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_laplace
PREPROCESS_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/preprocess_output_ne128_prod

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/plot_profile_illustration.py \
    --run-dir "${RUN_DIR}" \
    --gp "${RUN_DIR}/mf_gp_trained.pkl" \
    --preprocess-dir "${PREPROCESS_DIR}" \
    --out "${RUN_DIR}/diagnostics/profile_illustration_length_fac_vs_coeff_kh.png" \
    --param-x length_fac \
    --param-y coeff_kh

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
