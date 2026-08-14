#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=16:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_aurora_studentt1000
#PBS -o logs/stage2_mf_ne128_ne32_aurora_allparams_1000starts_studentt.out
#PBS -e logs/stage2_mf_ne128_ne32_aurora_allparams_1000starts_studentt.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_ne128_ne32_aurora_allparams_1000starts_studentt_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: with max_workers bumped to 192 (see config
# comment), oversubscription would be catastrophic if this weren't set.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_1000starts_studentt
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# 1000-start rerun of the MF Student-t (dof=4, kappa=0) baseline
# (configs/aurora_mf_ne128_ne32_prod_studentt_1000starts_annual.yaml),
# motivated by wanting more confidence in the true optimum now that
# Student-t has already brought boundary-pinning down to 3/19 (matching
# plain MAE, vs 10/19 under Gaussian-NLL kappa=0) -- more restarts are not
# expected to change *that* result (boundary-pinning looked like a genuine
# property of the cost surface's shape, confirmed consistent across 40
# independent starts), but should increase confidence that the free
# (non-pinned) parameters have found the true global optimum rather than
# one of many local ones, on a 19-D non-convex surface that 40 random
# starts may not adequately cover.
#
# max_workers bumped 32->192 in the config to keep this tractable: the
# 40-start version of this same run measured ~73min of optimize-phase wall
# time at max_workers=32 (1h53m total including ~38min of setup/checkpoint-
# load overhead, which is fixed regardless of n_xstarts). At 1000 starts
# and the same 32 workers that would be ~30h; at 192 workers (this node has
# 208 logical cores per `qstat`'s TSK column, and each forked worker is
# already single-threaded) it should be roughly ~5h, though this has not
# been empirically verified at this worker count -- if memory pressure from
# forking 192 copies of the loaded 1.9GB MF GP becomes a problem, dial
# max_workers back down and resubmit.
#
# Reuses the existing mf_gp_trained.pkl (GP training doesn't depend on
# n_xstarts or likelihood) -- no k-fold, no retraining, straight to
# optimization.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_gaussiannll/mf_gp_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_mf_ne128_ne32_prod_studentt_1000starts_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 1000 \
    --skip-gp

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
