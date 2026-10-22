#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -J mf3_n85_cold_opt
#SBATCH -o logs/stage2b_mf3_ne256_n85_COLD_studentt_dof4_%j.log
#SBATCH -e logs/stage2b_mf3_ne256_n85_COLD_studentt_dof4_%j.log

# Part B: 40-start optimization against the 3-level GP under Student-t dof=4,
# with the post-hoc inverse-Hessian parameter-uncertainty report (error bars).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# One BLAS thread per forked optimizer worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_cold_annual.yaml \
    --stage 2 --skip-gp --n-xstarts 40
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
