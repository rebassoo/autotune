#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J mf_v2_opt
#SBATCH -o logs/stage2b_mf_ne128_ne32_prod_v2_allparams_40starts_laplace_%j.log
#SBATCH -e logs/stage2b_mf_ne128_ne32_prod_v2_allparams_40starts_laplace_%j.log

# Stage 2, part B: 40-start optimization over all 19 parameters against the
# AR1 GP trained by stage2a, under the Laplace (GP-variance) likelihood.
# --skip-gp reloads <output_dir>/mf_gp_trained.pkl rather than retraining.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# One BLAS thread per worker: 32 processes x multi-threaded BLAS would badly
# oversubscribe the node, and BLAS threading measured as worthless here
# (232 vs 238 ms/cost-eval). optimize_parallel also enforces this at runtime.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf_ne128_ne32_prod_v2_annual.yaml \
    --stage 2 \
    --skip-gp \
    --n-xstarts 40

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
