#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_aurora_40starts
#PBS -o logs/stage2_mf_ne128_ne32_aurora_allparams_40starts.out
#PBS -e logs/stage2_mf_ne128_ne32_aurora_allparams_40starts.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_ne128_ne32_aurora_allparams_40starts_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor runs max_workers=32
# forked copies of the GP, so multi-threaded BLAS underneath would badly
# oversubscribe the node (see stage2_mf_ne128_ne32_allparams_100starts_parallel.sh
# on Perlmutter, where this was measured to make no difference to per-eval
# cost anyway).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_gaussiannll_hessianfix
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# The GP surrogate itself doesn't depend on the cost-function weights (those
# only enter later, at optimization time), so reuse the GP already trained
# on this exact all-19-param ne128+ne32 HF/LF data instead of retraining it.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/scratch_autotune/results_mf_ne128_ne32_prod_allparams/mf_gp_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_mf_ne128_ne32_prod_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --skip-gp

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
