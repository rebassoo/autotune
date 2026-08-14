#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_ne32_aurora_gnll_rg
#PBS -o logs/stage2_ne32_aurora_40starts_gaussiannll_regionsglobal.out
#PBS -e logs/stage2_ne32_aurora_40starts_gaussiannll_regionsglobal.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_ne32_aurora_40starts_gaussiannll_regionsglobal_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor (GPy backend) runs
# max_workers=32 forked copies of the GP, so multi-threaded BLAS underneath
# would badly oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne32_prod_aurora_40starts_gaussiannll_regionsglobal
GP_CKPT="${OUTPUT_DIR}/sf_gp_gpy_trained.pkl"
SOURCE_CKPT=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne32_prod_aurora_40starts_gaussiannll/sf_gp_gpy_trained.pkl

# Regions+global-only ablation of the SF ne32 gaussiannll run
# (configs/aurora_ne32_prod_gaussiannll_regionsglobal_annual.yaml --
# weights.zrg.zonal=0, regional/global=0.5/0.5; kappa=0, distance penalty
# off, same as the full-ZRG ne32 gaussiannll baseline). The GP surrogate
# does not depend on the zrg weights at all (those only enter cost_fn at
# optimization time), so this reuses the GPy checkpoint from the full-ZRG
# ne32 run (job 8743210) instead of retraining -- IF that job has finished
# and saved it by the time this one starts; if not, this falls through to
# training from scratch, same as that job did (hence walltime=6h and
# --kfold-folds 0 kept here too, as a safety net). K-fold is skipped
# entirely regardless: it's a diagnostic-only R²/RMSE check, unaffected by
# the zrg reweighting, and (when a retrain does happen) is the dominant
# cost for ne32's 833-member training set -- see job 8743210's script for
# the full accounting of why.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" && -f "${SOURCE_CKPT}" ]]; then
    cp "${SOURCE_CKPT}" "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_ne32_prod_gaussiannll_regionsglobal_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --sf-backend gpy \
    --kfold-folds 0

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
