#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -J plot_comparison
#SBATCH -o logs/plot_scenario_comparison_%j.log
#SBATCH -e logs/plot_scenario_comparison_%j.log

cd /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/autotune

export PYTHONUNBUFFERED=1

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/plot_scenario_comparison.py \
    --mf-dir    /pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_allparams_50starts \
    --mf-config configs/perlmutter_mf_ne128_ne32_prod_annual.yaml \
    --hf-dir    /pscratch/sd/r/rebassoo/autotune/results_ne128_prod_50starts \
    --hf-config configs/perlmutter_ne128_prod_annual.yaml \
    --lf-dir    /pscratch/sd/r/rebassoo/autotune/results_ne32_prod_50starts \
    --lf-config configs/perlmutter_ne32_prod_annual.yaml \
    --out-dir   /pscratch/sd/r/rebassoo/autotune/comparison_50starts
