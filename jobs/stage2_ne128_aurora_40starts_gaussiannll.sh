#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_ne128_aurora_gaussiannll
#PBS -o logs/stage2_ne128_aurora_40starts_gaussiannll.out
#PBS -e logs/stage2_ne128_aurora_40starts_gaussiannll.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_ne128_aurora_40starts_gaussiannll_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor (GPy backend) runs
# max_workers=32 forked copies of the GP, so multi-threaded BLAS underneath
# would badly oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_ne128_prod_aurora_40starts_gaussiannll
GP_CKPT="${OUTPUT_DIR}/sf_gp_gpy_trained.pkl"

# Full 19-param single-fidelity ne128 run under the Gaussian-NLL likelihood
# (configs/aurora_ne128_prod_gaussiannll_annual.yaml -- kappa=0, distance
# penalty off, weights matched to the MF gaussiannll configs: OSR downweighted
# to 0.1), for direct comparison against the MF kappa=0/kappa=1/distpenalty
# runs and against SF ne32's gaussiannll counterpart. The likelihood choice
# only changes the optimizer's cost function, not the trained GP surrogate
# itself, so reuse the GPy GP already trained for the equal-weight MAE
# 40-start ne128 run instead of retraining from scratch. K-fold evaluation
# still reruns regardless (no --skip-gp equivalent on the single-fidelity
# path), so this only saves the one full-dataset training pass, not the
# 5-fold passes.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/scratch_autotune/results_ne128_prod_gpy_40starts/sf_gp_gpy_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_ne128_prod_gaussiannll_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --sf-backend gpy

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
