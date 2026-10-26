#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J plot_structerr
#SBATCH -o logs/plot_structural_error_%j.log
#SBATCH -e logs/plot_structural_error_%j.log

# Misfit at the uncalibrated optimum vs. PPE spread and GP sd (the evidence for
# the per-variable model-error sigma). Needs a compute node: loading the
# 3-level GP exceeds the login-node memory limit.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/plot_structural_error.py \
    --calib-log $S/calib_n72iso_E3_zeroglobal/calibration_log.json --round 0 \
    --gp-pkl $S/results_mf3_ne256_n72iso_studentt_dof4/mf_gp_trained.pkl \
    --top-dir $P/preprocess_output_ne256_n85_was72 \
    --scalers-dir $P/preprocess_output_ne128_prod_v2 \
    --title "Misfit at the optimum vs. PPE spread — n=72, warm start, isotropic top kernel, before model error" \
    --out docs/structural_error_n72iso
