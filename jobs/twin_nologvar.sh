#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 04:00:00
#SBATCH -o logs/twin_nologvar_%x_%j.log
#SBATCH -e logs/twin_nologvar_%x_%j.log

# Twin experiment rerun with the Student-t cost's 0.5*log(variance) term dropped
# (optimize.logvar_weight: 0): does recovery improve when the optimizer is no longer
# rewarded for moving to where the surrogate is confident? Reuses the trained
# surrogate of results_twin_<TWIN> (no retraining), 40 starts, then evaluates the
# truth (--eval-truth). Needs TWIN, e.g. m202_3level.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
: "${TWIN:?set TWIN, e.g. m202_3level}"
SRC=$S/results_twin_$TWIN
OUT=${SRC}_nologvar
CFG=$S/twin_preprocess/$TWIN/config_nologvar.yaml
set -e
echo "Start: $(date)  $TWIN logvar_weight=0"; START=$(date +%s)
mkdir -p "$OUT"
ln -sfn "$(readlink -f "$SRC/mf_gp_trained.pkl")" "$OUT/mf_gp_trained.pkl"
$PY scripts/set_config_keys.py "$S/twin_preprocess/$TWIN/config.yaml" "$CFG" \
    optimize.logvar_weight=0.0 "paths.output_dir=$OUT"
$PY scripts/run_two_stage.py --config "$CFG" --stage 2 --skip-gp --n-xstarts 40 \
    --eval-truth "$S/twin_preprocess/$TWIN/truth.pkl"
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
