#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -J mf3_n85_freeze6
#SBATCH -o logs/stage2b_mf3_ne256_n85_freeze6_%j.log
#SBATCH -e logs/stage2b_mf3_ne256_n85_freeze6_%j.log

# Re-optimize the n=85 warm/iso surrogate with the 6 parameters whose
# total-order Sobol index is < 0.01 for every variable and misfit
# (sobol_n85: deposition_nucleation_exponent, thl2tune,
# rain_selfcollection_breakup_diameter, autoconversion_radius,
# rain_to_ice_collection_factor, c_diag_3rd_mom) frozen, the other 13 free.
# FREEZE=fitted : frozen at the full 19-parameter run's best fit
# FREEZE=default: frozen at the control-run values (default_params.pkl,
#                 clipped to the sampled range)
# Compare best cost and the 13 free values with the 19-parameter run.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
S=/pscratch/sd/r/rebassoo/autotune
FULL=$S/results_mf3_ne256_n85_studentt_dof4
OUT=$S/results_mf3_ne256_n85_freeze6_${FREEZE:?set FREEZE=fitted or default}
VARY=qw2tune,length_fac,coeff_kh,coeff_km,lambda_low,lambda_high,spa_ccn_to_nc_factor,cldliq_to_ice_collection_factor,accretion_prefactor,max_total_ni,ice_sedimentation_factor,autoconversion_prefactor,autoconversion_qc_exponent
EXTRA=()
[ "$FREEZE" = fitted ] && EXTRA=(--freeze-from "$FULL")

echo "Start: $(date)  FREEZE=$FREEZE"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \
    --stage 2 --skip-gp --n-xstarts 40 --output-dir "$OUT" \
    --vary-params "$VARY" "${EXTRA[@]}"
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
