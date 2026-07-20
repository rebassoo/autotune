#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J stage2_mf_100par
#SBATCH -o logs/stage2_mf_ne128_ne32_allparams_100starts_parallel_%j.log
#SBATCH -e logs/stage2_mf_ne128_ne32_allparams_100starts_parallel_%j.log

# 100 starts via the process executor (optimize.executor=process,
# max_workers=32). Serial this is ~50h; measured 2.95x on 4 workers
# (74% efficiency, memory-bandwidth bound), so expect roughly 2-4h.
#
# NOTE: runs from the autotune-parallel worktree, which holds the parallel
# optimizer. The main autotune/ checkout is still the serial code.
cd /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/autotune-parallel

export PYTHONUNBUFFERED=1
# One BLAS thread per worker: 32 workers x multi-threaded BLAS would badly
# oversubscribe the node, and BLAS threading measured as worthless here
# (232 vs 238 ms/cost-eval). optimize_parallel also enforces this at runtime.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf_ne128_ne32_prod_annual.yaml \
    --stage 2 \
    --output-dir /pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_allparams_100starts_parallel \
    --n-xstarts 100 \
    --skip-gp

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
