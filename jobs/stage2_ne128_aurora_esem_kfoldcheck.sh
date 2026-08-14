#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_ne128_esem_kfoldck
#PBS -o logs/stage2_ne128_aurora_esem_kfoldcheck.out
#PBS -e logs/stage2_ne128_aurora_esem_kfoldcheck.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_ne128_aurora_esem_kfoldcheck_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne128_prod_aurora_esem_kfoldcheck

# Apples-to-apples check of whether ESEm's default composite kernel
# (RBF + Linear + Polynomial + Bias, additive -- see esem/__init__.py's
# gp_model default) actually fits SF ne128 better than the plain-RBF GPy
# backend, on the SAME preprocess_dir/kfold_data.pkl (same 5 folds) already
# used for the ne128 gaussiannll (GPy) run's k-fold numbers
# (logs/stage2_ne128_aurora_40starts_gaussiannll_20260803_224536.log:
# PCP R^2=0.697, OSR=0.710, OLR=0.438, TLWP=0.715, mean across folds).
# k-fold R^2/RMSE is independent of likelihood/cost weighting, so this
# plain config (likelihood defaults to 'mae', sf_gp_backend defaults to
# 'esem') is sufficient -- motivated by the multi-fidelity-kernel question:
# if the composite kernel doesn't clearly beat plain RBF even here, in the
# simpler single-fidelity setting, it's weaker evidence that porting it to
# the MF AR1 kernel would close the MF-vs-SF gap.
mkdir -p "${OUTPUT_DIR}"

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_ne128_prod_esem_kfoldcheck_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --sf-backend esem

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
