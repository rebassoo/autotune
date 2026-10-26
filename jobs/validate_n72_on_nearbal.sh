#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -J val_n72_nearbal
#SBATCH -o logs/validate_n72_on_nearbal_%j.log
#SBATCH -e logs/validate_n72_on_nearbal_%j.log

# Out-of-sample check: predict the 13 near-balance ne256 members with the n=72
# surrogates (trained without them) before retraining on n=85.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
for gp in /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_studentt_dof4/mf_gp_trained.pkl \
          /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_studentt_dof4_COLD/mf_gp_trained.pkl \
          /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n72iso_studentt_dof4/mf_gp_trained.pkl \
          /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n72iso_studentt_dof4_COLD/mf_gp_trained.pkl; do
    [ -f "$gp" ] || { echo "skip (not trained yet): $gp"; continue; }
    tag=$(basename $(dirname $gp))
    /pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/validate_on_new_members.py \
        --gp-pkl "$gp" \
        --top-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_n85 \
        --scalers-dir /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne128_prod_v2 \
        --select _nearbal \
        --out /pscratch/sd/r/rebassoo/autotune/validate_nearbal/${tag}.pkl
done
