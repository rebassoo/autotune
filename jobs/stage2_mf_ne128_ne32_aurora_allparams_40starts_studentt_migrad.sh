#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_aurora_studentt_migrad40
#PBS -o logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt_migrad.out
#PBS -e logs/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt_migrad.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_ne128_ne32_aurora_allparams_40starts_studentt_migrad_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_studentt_migrad
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# 40-start MIGRAD check before committing the 1000-start run to it. A 5-start
# spot check (same GP/cost_fn, same x0s) found MIGRAD beat L-BFGS-B+
# basinhopping on 4/5 starts (one case 0.645->0.477, a 26% cost reduction)
# and ran 4-9x faster per start (400-800s vs ~3500s/start-equivalent for
# L-BFGS-B+basinhopping at max_workers=32). This run is the same GP/cost_fn
# as results_mf_ne128_ne32_prod_aurora_allparams_40starts_studentt (the
# original L-BFGS-B 40-start run) with only optimize.method changed, so the
# two are directly comparable. max_workers left at 32 (not the 1000-start
# run's 192) since this is just the pre-1000-start sanity check.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_gaussiannll/mf_gp_trained.pkl \
       "${GP_CKPT}"
fi

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_mf_ne128_ne32_prod_studentt_migrad_40starts_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --skip-gp

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
