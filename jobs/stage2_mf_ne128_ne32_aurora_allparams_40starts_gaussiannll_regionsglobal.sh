#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_aurora_gnll_rg
#PBS -o logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_gaussiannll_regionsglobal.out
#PBS -e logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_gaussiannll_regionsglobal.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_ne128_ne32_aurora_allparams_40starts_gaussiannll_regionsglobal_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor runs max_workers=32
# forked copies of the GP, so multi-threaded BLAS underneath would badly
# oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_gaussiannll_regionsglobal
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# Regions+global-only ablation of the MF kappa=0 gaussiannll baseline
# (configs/aurora_mf_ne128_ne32_prod_regionsglobal_annual.yaml --
# weights.zrg.zonal=0, regional/global=0.5/0.5; kappa=0, distance penalty
# explicitly reset to 0 -- the shared aurora_mf_ne128_ne32_prod_annual.yaml
# this was cloned from currently has training_distance_penalty_lambda=1.0
# live in it from the distpenalty_lambda1 test run). The AR1 GP does not
# depend on the zrg weights (only cost_fn does, at optimization time), so
# --skip-gp reuses the existing mf_gp_trained.pkl from the full-ZRG
# kappa=0 gaussiannll run instead of retraining -- no k-fold, no GP
# training, straight to optimization.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_gaussiannll/mf_gp_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_mf_ne128_ne32_prod_regionsglobal_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --skip-gp

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
