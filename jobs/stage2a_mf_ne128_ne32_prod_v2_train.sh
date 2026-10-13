#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J mf_v2_train
#SBATCH -o logs/stage2a_mf_ne128_ne32_prod_v2_train_%j.log
#SBATCH -e logs/stage2a_mf_ne128_ne32_prod_v2_train_%j.log

# Stage 2, part A, on the 20260608 production PPEs (v2): 5-fold multi-fidelity
# surrogate skill evaluation, then AR1 GP training on the full dataset.
# Optimization is skipped here and run by stage2b, which needs the opposite
# threading setup (one BLAS thread per forked optimizer process).
# The trained GP is checkpointed to <output_dir>/mf_gp_trained.pkl.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# 5 k-fold workers x 25 BLAS threads, filling the full node.
export KFOLD_WORKERS=5
export OMP_NUM_THREADS=25
export MKL_NUM_THREADS=25

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf_ne128_ne32_prod_v2_annual.yaml \
    --stage 2 \
    --skip-optimize

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
