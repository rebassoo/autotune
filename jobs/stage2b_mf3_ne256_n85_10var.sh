#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J mf3_n85_10var
#SBATCH -o logs/stage2b_mf3_ne256_n85_10var_%j.log
#SBATCH -e logs/stage2b_mf3_ne256_n85_10var_%j.log

# 10-variable optimization (configs/perlmutter_mf3_ne256_n85_10var_annual.yaml):
#  1. merge the 4-variable GP (PCP/OSR/OLR/TLWP) and the 6-variable GP
#     (jobs/train_mf_6var.sh LEVEL=3) into one 10-variable GP, no retraining;
#  2. optimize it, Student-t dof=4, 40 starts, global SLP and the few-land-cell
#     TAS_LAND features masked.
# Submit with --dependency=afterok:<train_mf_6var LEVEL=3 job id>.
# The 4-variable runs took 3-5 h at 41 GB; ~2.5x the variables -> ~10-13 h.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
OUT=$S/results_mf3_ne256_n85_10var
set -e
echo "Start: $(date)"; START=$(date +%s)
if [ ! -f "$OUT/mf_gp_trained.pkl" ]; then
    $PY scripts/merge_gp_vars.py \
        --gp "$S/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl=PCP,OSR,OLR,TLWP@$P/preprocess_output_ne128_prod_v2" \
        --gp "$S/results_mf3_ne256_n85_6var/mf_gp_trained.pkl=NETTOA,TAS_LAND,SLP,U200,U850,Z500@$P/preprocess_output_ne128_prod_v2_6var" \
        --scalers-dir $P/preprocess_output_ne128_prod_v2_10var \
        --out "$OUT/mf_gp_trained.pkl"
fi
echo "Merged: $(date)"
$PY scripts/run_two_stage.py \
    --config configs/perlmutter_mf3_ne256_n85_10var_annual.yaml \
    --stage 2 --skip-gp --n-xstarts 40 --output-dir "$OUT"
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
