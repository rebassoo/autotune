#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 03:00:00
#SBATCH -J twin_eval
#SBATCH -o logs/twin_eval_%j.log
#SBATCH -e logs/twin_eval_%j.log

# Follow-up to jobs/twin_ne256.sh, for each twin: reload the 40 checkpointed starts
# (no new optimization), rewrite the results CSV and diagnostics with the corrected
# lowest-cost ranking, and evaluate the cost and prediction at the true parameters
# (--eval-truth -> diagnostics/eval_truth.pkl). CPU only; on the GPU allocation
# because the e3sm CPU allocation is exhausted.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
echo "Start: $(date)"
for t in ${TWINS:-m060_3level m060_2level m202_3level m202_2level}; do
    echo "===== $t $(date)"
    $PY scripts/run_two_stage.py --config $S/twin_preprocess/$t/config.yaml --stage 2 --skip-gp \
        --n-xstarts 40 --eval-truth $S/twin_preprocess/$t/truth.pkl || echo "FAILED $t"
done
echo "End: $(date)"
