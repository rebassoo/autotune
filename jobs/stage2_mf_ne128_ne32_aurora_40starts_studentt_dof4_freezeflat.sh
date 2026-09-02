#!/bin/bash
#PBS -A E3SM_Dec
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -l filesystems=home_fs:flare_fs
#PBS -q capacity
#PBS -N stage2_mf_dof4_freezeflat
#PBS -o logs/stage2_mf_ne128_ne32_aurora_40starts_studentt_dof4_freezeflat.out
#PBS -e logs/stage2_mf_ne128_ne32_aurora_40starts_studentt_dof4_freezeflat.err

set -euo pipefail

cd /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/autotune-gp-variance-likelihood

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/stage2_mf_ne128_ne32_aurora_40starts_studentt_dof4_freezeflat_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTPUT_DIR=/home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_40starts_studentt_dof4_freezeflat
GP_CKPT="${OUTPUT_DIR}/mf_gp_trained.pkl"

# Direct test of whether the 6 parameters whose Laplace stderr exceeds the
# full [0,1] box width (rain_selfcollection_breakup_diameter=4.39,
# c_diag_3rd_mom=2.44, thl2tune=1.82, cldliq_to_ice_collection_factor=1.60,
# autoconversion_radius=1.35, rain_to_ice_collection_factor=1.09 -- from
# results_mf_ne128_ne32_prod_aurora_allparams_40starts_studentt's
# parameter_uncertainty_seed50.pkl) can be safely dropped, or whether an
# interaction effect (invisible to a 1-D scan at x_best, which already
# confirmed each is individually flat across its own full range) makes them
# matter in combination. --vary-params freezes them at the physically-
# motivated control-run default (default_params.pkl) -- a genuinely
# independent reference point from x_best -- and re-optimizes only the
# remaining 13 well-constrained params. Same seed=50/niter=1/dof=4 config
# and 40 starts as the full-19-param baseline (best cost 0.323973) for
# direct comparison.
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${GP_CKPT}" ]]; then
    cp /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/results_mf_ne128_ne32_prod_aurora_allparams_40starts_studentt/mf_gp_trained.pkl \
       "${GP_CKPT}"
fi
cp -n /home/rebassoo/work/MFAutotune_backup_2026-07-21/home_work/preprocess_output_ne128_prod/default_params.pkl \
      "${OUTPUT_DIR}/default_params.pkl"

echo "Start: $(date)"
START=$(date +%s)

/lus/flare/projects/E3SM_Dec/rebassoo/conda/autotune_env/bin/python scripts/run_two_stage.py \
    --config configs/aurora_mf_ne128_ne32_prod_studentt_annual.yaml \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --n-xstarts 40 \
    --skip-gp \
    --vary-params qw2tune,length_fac,coeff_kh,coeff_km,lambda_low,lambda_high,spa_ccn_to_nc_factor,accretion_prefactor,deposition_nucleation_exponent,max_total_ni,ice_sedimentation_factor,autoconversion_prefactor,autoconversion_qc_exponent

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
