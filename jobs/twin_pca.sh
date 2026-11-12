#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 05:00:00
#SBATCH -o logs/twin_pca_%x_%j.log
#SBATCH -e logs/twin_pca_%x_%j.log

# Twin experiment with the PCA likelihood (likelihood: pca_gaussian, K=20 EOFs, error
# covariance from the 3-level surrogate's holdout errors with the target member left
# out; scripts/build_pca_likelihood.py -> twin_preprocess/<TWIN>/pca_basis_k20.pkl).
# Reuses the trained surrogate of results_twin_<TWIN> (no retraining): 40 starts,
# cost at the truth (--eval-truth), then emcee posterior sampling (64 walkers x
# 2000 steps) to test whether the true parameters fall inside the posterior.
# Needs TWIN, a 3-level twin (m060_3level or m202_3level).
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
echo "Start: $(date)  $TWIN likelihood=pca_gaussian"; START=$(date +%s)
mkdir -p "$OUT"
ln -sfn "$(readlink -f "$SRC/mf_gp_trained.pkl")" "$OUT/mf_gp_trained.pkl"
$PY scripts/set_config_keys.py "$S/twin_preprocess/$TWIN/config.yaml" "$CFG" \
    optimize.likelihood=pca_gaussian "optimize.pca_basis=$S/twin_preprocess/$TWIN/pca_basis_k20.pkl" \
    "paths.output_dir=$OUT"
$PY scripts/run_two_stage.py --config "$CFG" --stage 2 --skip-gp --n-xstarts 40 \
    --eval-truth "$S/twin_preprocess/$TWIN/truth.pkl" --mcmc-steps 2000
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
