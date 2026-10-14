#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J mf3_calib
#SBATCH -o logs/calibrate_model_error_mf3_%j.log
#SBATCH -e logs/calibrate_model_error_mf3_%j.log

# Iterative model-error calibration (docs/model_error_residual_calibration.md):
# optimize -> residuals -> sigma_delta per variable x {zonal, regional} ->
# re-optimize, until the best point and sigma stop moving. Needs the trained
# surrogate (mf_gp_trained.pkl in the config's output_dir) -- submit with
# --dependency=afterok:<training job>.
#
# Each 40-start round on the 3-level GP is ~3.5-4.5 h, so 3-4 rounds plus the
# two sensitivity runs may not fit in 24 h. Everything resumes: just resubmit
# this script with the same CONFIG/OUT and it continues where it stopped.
#
# Override at submit time, e.g.
#   sbatch --export=ALL,CONFIG=configs/<new>.yaml,OUT=/pscratch/.../calib_<new> \
#          jobs/calibrate_model_error_mf3.sh
# ROUND0=<dir of an existing 40-start optimization without model error> reuses
# it as round 0 (saves one round).
CONFIG=${CONFIG:-configs/perlmutter_mf3_ne256_studentt_dof4_annual.yaml}
OUT=${OUT:-/pscratch/sd/r/rebassoo/autotune/calib_mf3_ne256_studentt_dof4}
GLOBAL_ARGS=${GLOBAL_ARGS:-"--global OSR=2.0 --global OLR=2.0"}   # W/m^2, CERES-level

cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# One BLAS thread per forked optimizer worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"; START=$(date +%s)
echo "CONFIG=$CONFIG OUT=$OUT ROUND0=${ROUND0:-none} $GLOBAL_ARGS"
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/calibrate_model_error.py \
    --config "$CONFIG" --out-dir "$OUT" --n-xstarts 40 \
    ${ROUND0:+--round0-dir "$ROUND0"} $GLOBAL_ARGS --max-rounds 4 --sensitivity
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
