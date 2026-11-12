#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -J ne256_n85_sf
#SBATCH -o logs/stage2_ne256_n85_sf_gpy_studentt_dof4_%j.log
#SBATCH -e logs/stage2_ne256_n85_sf_gpy_studentt_dof4_%j.log

# Single-fidelity ne256 (n=85 members only) GPy surrogate under Student-t dof=4:
# 5-fold skill, training on all 85 members, then 40-start optimization.
# Compare its optimum with the 3-level run (results_mf3_ne256_n85_studentt_dof4).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_ne256_n85_sf_gpy_studentt_dof4_annual.yaml \
    --stage 2 --n-xstarts 40
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
