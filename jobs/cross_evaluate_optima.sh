#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J cross_eval
#SBATCH -o logs/cross_evaluate_optima_%j.log
#SBATCH -e logs/cross_evaluate_optima_%j.log

# 3-level MF optimum vs ne256-alone optimum, each predicted by both surrogates,
# misfit in the ne128-normalized space the MF cost uses. Needs a compute node:
# the 3-level GP exceeds the login-node memory limit.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/cross_evaluate_optima.py \
    --surrogate "MF=$S/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl@$P/preprocess_output_ne128_prod_v2" \
    --surrogate "ne256=$S/results_ne256_n85_sf_gpy_studentt_dof4/sf_gp_gpy_trained.pkl@$P/preprocess_output_ne256_n85" \
    --optimum "MF=$S/results_mf3_ne256_n85_studentt_dof4" \
    --optimum "ne256=$S/results_ne256_n85_sf_gpy_studentt_dof4" \
    --obs-dir $P/preprocess_output_ne256_n85 --norm-dir $P/preprocess_output_ne128_prod_v2 \
    --out docs/cross_eval_mf3_vs_ne256alone_n85.csv
