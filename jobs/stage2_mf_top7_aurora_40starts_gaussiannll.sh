#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_aurora_top7
#PBS -o logs/stage2_mf_top7_aurora_40starts_gaussiannll.out
#PBS -e logs/stage2_mf_top7_aurora_40starts_gaussiannll.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_top7_aurora_40starts_gaussiannll_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: the process executor runs max_workers=32
# forked copies of the GP, so multi-threaded BLAS underneath would badly
# oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_top7_40starts_gaussiannll
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# Subset optimization: vary only the same 7 union top-parameters used in the
# Perlmutter top7 comparison run (top-6 by Pearson + top-6 by GP
# cost-sensitivity, unioned), freezing the other 12 at the control-run
# default. Reuses the existing 19-param GP via --skip-gp -- no retraining, no
# k-fold. Uses the Gaussian-NLL likelihood + parameter-uncertainty report
# (already set in aurora_mf_ne128_ne32_prod_annual.yaml: optimize.likelihood:
# gaussian_gp_variance, diagnostics.report_uncertainty: true), so this is
# directly comparable to both the earlier Aurora all-19-param gaussiannll run
# and the original Perlmutter top7 (MAE) run.
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
    --skip-gp \
    --vary-params length_fac,coeff_kh,autoconversion_qc_exponent,ice_sedimentation_factor,qw2tune,spa_ccn_to_nc_factor,max_total_ni

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
