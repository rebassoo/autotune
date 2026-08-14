#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_ne128_aurora_gnll_rg
#PBS -o logs/stage2_ne128_aurora_40starts_gaussiannll_regionsglobal.out
#PBS -e logs/stage2_ne128_aurora_40starts_gaussiannll_regionsglobal.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_ne128_aurora_40starts_gaussiannll_regionsglobal_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor (GPy backend) runs
# max_workers=32 forked copies of the GP, so multi-threaded BLAS underneath
# would badly oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne128_prod_aurora_40starts_gaussiannll_regionsglobal
GP_CKPT="${OUTPUT_DIR}/sf_gp_gpy_trained.pkl"

# Regions+global-only ablation of the SF ne128 gaussiannll run
# (configs/aurora_ne128_prod_gaussiannll_regionsglobal_annual.yaml --
# weights.zrg.zonal=0, regional/global=0.5/0.5; kappa=0, distance penalty
# off, same as the full-ZRG ne128 gaussiannll baseline). The GP surrogate
# does not depend on the zrg weights at all (those only enter cost_fn at
# optimization time), so this reuses the already-trained GPy checkpoint from
# the full-ZRG run instead of retraining. K-fold is also skipped entirely
# (--kfold-folds 0): it's a diagnostic-only R²/RMSE check on the same
# surrogate, already recorded from that earlier run, and is unaffected by
# the zrg reweighting.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne128_prod_aurora_40starts_gaussiannll/sf_gp_gpy_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_ne128_prod_gaussiannll_regionsglobal_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --sf-backend gpy \
    --kfold-folds 0

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
