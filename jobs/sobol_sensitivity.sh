#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -J sobol
#SBATCH -o logs/sobol_sensitivity_%j.log
#SBATCH -e logs/sobol_sensitivity_%j.log

# Sobol sensitivity of the surrogate's outputs to the tuning parameters
# (lambda_low/high held at the best fit). Override with env vars:
#   GP_DIR  results dir holding mf_gp_trained.pkl and the best fit
#   CONFIG  matching optimization config
#   OUT     output dir;  NBASE  Saltelli base size (power of 2)
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
S=/pscratch/sd/r/rebassoo/autotune
GP_DIR=${GP_DIR:-$S/results_mf3_ne256_n85_studentt_dof4}
CONFIG=${CONFIG:-configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml}
OUT=${OUT:-$S/sobol_n85}
NBASE=${NBASE:-2048}
echo "Start: $(date)"
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/sobol_sensitivity.py \
    --gp-pkl $GP_DIR/mf_gp_trained.pkl --config $CONFIG --theta-from $GP_DIR \
    --n-base $NBASE --out $OUT
echo "End: $(date)"
