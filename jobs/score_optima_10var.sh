#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -J score10
#SBATCH -o logs/score_optima_10var_%j.log
#SBATCH -e logs/score_optima_10var_%j.log
# Score default, the 10-var and the 4-var optima on all 10 variables (merged GP).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONPATH="$(pwd)/src:$(pwd)" OMP_NUM_THREADS=8
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/score_optima_10var.py \
    --gp-pkl $S/results_mf3_ne256_n85_10var/mf_gp_trained.pkl \
    --scalers-dir $P/preprocess_output_ne128_prod_v2_10var \
    --ref $S/results_mf3_ne256_n85_10var \
    --opt 10var=$S/results_mf3_ne256_n85_10var \
    --opt 4var=$S/results_mf3_ne256_n85_studentt_dof4 \
    --opt-rank 10var_r5=$S/results_mf3_ne256_n85_10var:5 \
    --out-csv docs/score_optima_10var.csv
