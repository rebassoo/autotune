#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 06:00:00
#SBATCH -J stage2_ne128_gpy
#SBATCH -o logs/stage2_ne128_gpy_40starts_%j.log
#SBATCH -e logs/stage2_ne128_gpy_40starts_%j.log

# Single-fidelity ne128 with the GPy backend (--sf-backend gpy):
#   - k-fold, GP training, and prediction all use GPy (independent per-feature
#     GPRegression models), not ESEm/GPflow. No TensorFlow is run.
#   - run_stage2 auto-selects the PROCESS executor for the GPy backend (GPy is
#     not thread-safe but forks cleanly), so the 40 starts run in parallel like
#     the multi-fidelity job. Because no TF runs, the fork is safe.
#   - the trained GPy GP is pickled to sf_gp_gpy_trained.pkl and reloaded on a
#     rerun (unlike the ESEm backend, which cannot be pickled).
#
# Output dir is distinct from the ESEm ne128 results so nothing is overwritten;
# the ESEm default path is completely unaffected.
cd /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/autotune

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: 32 process workers x threaded BLAS would
# oversubscribe the node. optimize_parallel also enforces this via threadpoolctl.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_ne128_prod_annual.yaml \
    --stage 2 \
    --sf-backend gpy \
    --output-dir /pscratch/sd/r/rebassoo/autotune/results_ne128_prod_gpy_40starts \
    --n-xstarts 40

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
