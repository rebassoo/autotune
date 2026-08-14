#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_aurora_studentt
#PBS -o logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt.out
#PBS -e logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor runs max_workers=32
# forked copies of the GP, so multi-threaded BLAS underneath would badly
# oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_studentt
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# Full-ZRG MF (ne128+ne32) run under the Student-t likelihood, dof=4
# (configs/aurora_mf_ne128_ne32_prod_studentt_annual.yaml -- kappa=0,
# distance penalty off, same baseline as the MF Gaussian-NLL kappa=0 run
# this is meant to compare against). The AR1 GP does not depend on the
# likelihood choice (only cost_fn does, at optimization time), so --skip-gp
# reuses the existing mf_gp_trained.pkl from the Gaussian-NLL kappa=0 run
# instead of retraining -- no k-fold, no GP training, straight to
# optimization.
#
# A local (non-PBS) smoke test of this exact config stalled on very slow
# I/O loading the 1.9GB checkpoint (down to ~2 MB/min at one point, almost
# certainly filesystem contention from other jobs sharing the home
# directory at the time) -- unrelated to the Student-t cost function itself,
# which was separately validated numerically (converges to
# zrg_cost_function_gaussian_nll exactly as dof -> infinity). This job runs
# on a dedicated PBS-allocated compute node, a different resource pool, so
# it should not hit the same contention; if it does, check elapsed time at
# the "Loading saved GP" log line first before assuming a code problem.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_gaussiannll/mf_gp_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_mf_ne128_ne32_prod_studentt_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --skip-gp

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
