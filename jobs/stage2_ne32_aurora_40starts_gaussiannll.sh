#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_ne32_aurora_gaussiannll
#PBS -o logs/stage2_ne32_aurora_40starts_gaussiannll.out
#PBS -e logs/stage2_ne32_aurora_40starts_gaussiannll.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_ne32_aurora_40starts_gaussiannll_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor (GPy backend) runs
# max_workers=32 forked copies of the GP, so multi-threaded BLAS underneath
# would badly oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne32_prod_aurora_40starts_gaussiannll

# Full 19-param single-fidelity ne32 run under the Gaussian-NLL likelihood
# (configs/aurora_ne32_prod_gaussiannll_annual.yaml -- kappa=0, distance
# penalty off, weights matched to the MF gaussiannll configs: OSR downweighted
# to 0.1), for direct comparison against the MF kappa=0/kappa=1/distpenalty
# runs and against SF ne128's gaussiannll counterpart. No prior GPy-backend
# checkpoint exists for ne32 (the earlier ne32 40-start run used the ESEm
# backend and was never saved), so this trains a fresh GPy GP from scratch --
# k-fold + full-dataset training, both required for pred_var to feed the
# Gaussian-NLL cost.
#
# First attempt (job 8731250, 2026-08-03) ran the full 4h wall and was killed
# mid-way through the full-dataset training step. Its log showed each k-fold
# fold alone took ~40-43 min (GPy retrains all n_vars*n_feat=88 independent
# per-feature models per fold, and ne32's LF training set is 833 members vs
# ne128's 109 -- ~7-8x more points per model to fit): 5 folds burned ~3h20m
# before the *actual* full-dataset training (another 88-model pass, needed
# for the surrogate used downstream) even started. K-fold here is only a
# diagnostic R²/RMSE check, not used by the optimizer, so --kfold-folds 1
# trades 4 of the 5 folds' coverage for ~2h40m of wall time back. Walltime
# bumped 4h->6h as well as a safety margin now that most of the time
# pressure is gone (queue itself allows up to 168h -- `qstat -Qf capacity`
# -- so this isn't near any hard cap; the earlier same-day >4h dispatch-
# failure issue from this session's PBS investigation was tied to that day's
# scheduler state, not a fixed policy, so it may or may not recur here).
mkdir -p "${OUTPUT_DIR}"

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_ne32_prod_gaussiannll_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --sf-backend gpy \
    --kfold-folds 1

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
