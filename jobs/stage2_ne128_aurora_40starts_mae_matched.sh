#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_ne128_aurora_mae_matched
#PBS -o logs/stage2_ne128_aurora_40starts_mae_matched.out
#PBS -e logs/stage2_ne128_aurora_40starts_mae_matched.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_ne128_aurora_40starts_mae_matched_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne128_prod_aurora_40starts_mae_matched
GP_CKPT="${OUTPUT_DIR}/sf_gp_gpy_trained.pkl"

# SF ne128 GPy plain-MAE counterpart to stage2_ne128_aurora_40starts_studentt_dof4.sh,
# weight-matched (PCP/OSR/OLR/TLWP = 0.3/0.1/0.3/0.3) for a direct dof=4-vs-MAE
# comparison -- see configs/aurora_ne128_prod_mae_matched_annual.yaml. NOT the
# same run as the older scratch_autotune/results_ne128_prod_gpy_40starts MAE
# run, which used equal 0.25 weights. Reuses the same trained SF GPy GP.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne128_prod_aurora_40starts_gaussiannll/sf_gp_gpy_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_ne128_prod_mae_matched_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --sf-backend gpy

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
