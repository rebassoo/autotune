#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J ne128_v2_studentt4
#SBATCH -o logs/stage2_ne128_prod_v2_sf_gpy_studentt_dof4_40starts_%j.log
#SBATCH -e logs/stage2_ne128_prod_v2_sf_gpy_studentt_dof4_40starts_%j.log

# Single-fidelity stage 2 on the ne128 (high-resolution) production PPE v2
# under a Student-t (dof=4) noise model: 5-fold surrogate skill evaluation,
# GPy GP training on all 218 members, then 40-start optimization.
#
# Uses the GPy backend, not ESEm: the Student-t cost weights each location by
# that location's predictive variance, and ESEm's single joint GP does not
# provide a genuinely per-feature variance. This also makes the surrogate
# architecturally identical to the AR1 model minus the low-fidelity data, so
# pairing this with the MF Student-t run isolates the fidelity question.
#
# No GP to reuse (the ESEm MAE run's sf_gp_trained.pkl is a different
# backend and a different object), so this trains its own.
# run_stage2 has no --skip-gp/--skip-optimize, so k-fold, training and
# optimization all run in this one job.
#
# Requires stage 1 for configs/perlmutter_ne128_prod_v2_annual.yaml.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# GPy forces the process executor for single-fidelity, and optimize_parallel
# caps BLAS to one thread per forked worker; keep the job env consistent.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_ne128_prod_v2_sf_gpy_studentt_dof4_annual.yaml \
    --stage 2 \
    --n-xstarts 40

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
