#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 04:30:00
#SBATCH -o logs/twin_mcmc_%x_%j.log
#SBATCH -e logs/twin_mcmc_%x_%j.log

# Longer posterior sampling for a PCA-likelihood twin (follow-up to jobs/twin_pca.sh):
# reloads the 40 checkpointed starts of results_twin_<TWIN>_pca (no re-optimization)
# and runs emcee for 10000 steps with the 64 walkers started at every start endpoint
# within 10 cost units of the best, so each competitive mode is entered. Writes
# diagnostics/posterior_10000steps.pkl. Needs TWIN (m060_3level or m202_3level).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
: "${TWIN:?set TWIN, e.g. m202_3level}"
SRC=$S/results_twin_$TWIN
OUT=${SRC}_pca
CFG=$S/twin_preprocess/$TWIN/config_pca.yaml
set -e
echo "Start: $(date)  $TWIN likelihood=pca_gaussian, 10000-step MCMC"; START=$(date +%s)
test -f "$CFG" -a -f "$OUT/mf_gp_trained.pkl"   # made by jobs/twin_pca.sh
$PY scripts/run_two_stage.py --config "$CFG" --stage 2 --skip-gp --n-xstarts 40 \
    --eval-truth "$S/twin_preprocess/$TWIN/truth.pkl" --mcmc-steps 10000 --mcmc-init-dcost 10
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
