#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J mf_v2_studentt4
#SBATCH -o logs/stage2_mf_ne128_ne32_prod_v2_studentt_dof4_40starts_%j.log
#SBATCH -e logs/stage2_mf_ne128_ne32_prod_v2_studentt_dof4_40starts_%j.log

# Multi-fidelity 40-start optimization on the 20260608 production PPEs (v2)
# under a Student-t (dof=4) noise model.
#
# The AR1 GP does not depend on the cost function, so this reuses the
# surrogate trained by stage2a for the Laplace run rather than spending hours
# retraining an identical model. The k-fold evaluation is likewise identical
# and is skipped along with training (--skip-gp covers both).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
# One BLAS thread per forked optimizer worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TRAINED_GP=/pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_v2/mf_gp_trained.pkl
OUT_DIR=/pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_v2_studentt_dof4_40starts
mkdir -p "${OUT_DIR}"

# Reuse the trained AR1 GP if the Laplace run produced one; otherwise train a
# fresh one here. --skip-gp errors out rather than training, so only pass it
# when the checkpoint is actually in place.
SKIP_GP=""
if [ -f "${TRAINED_GP}" ]; then
    ln -sf "${TRAINED_GP}" "${OUT_DIR}/mf_gp_trained.pkl"
    SKIP_GP="--skip-gp"
    echo "Reusing trained AR1 GP: ${TRAINED_GP}"
else
    echo "No trained AR1 GP at ${TRAINED_GP} — training one in this job."
fi

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf_ne128_ne32_prod_v2_studentt_dof4_annual.yaml \
    --stage 2 \
    ${SKIP_GP} \
    --n-xstarts 40

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
