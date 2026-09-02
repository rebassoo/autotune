#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=16:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_aurora_studentt_niter5
#PBS -o logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt_niter5.out
#PBS -e logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt_niter5.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt_niter5_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_studentt_niter5
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# niter=5 test: same 40 x0s (seed=50) as the niter=1 baseline
# (results_mf_ne128_ne32_prod_aurora_allparams_40starts_studentt, best
# cost=0.324) and the MIGRAD run (results_..._studentt_migrad, best
# cost=0.387, no hopping). Direct test of whether more basin-hopping per
# start improves the best-of-40 result, or whether niter=1's edge over
# MIGRAD was just luck-of-the-draw between the two local optimizers (mean
# x0->final displacement was ~equal for both methods, 1.84 vs 1.85, which
# didn't show hopping systematically reaching further). Per-start
# checkpoints make a walltime kill resumable.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_gaussiannll/mf_gp_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_mf_ne128_ne32_prod_studentt_niter5_40starts_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --skip-gp

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
