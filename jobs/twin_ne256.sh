#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 08:00:00
#SBATCH -o logs/twin_ne256_%x_%j.log
#SBATCH -e logs/twin_ne256_%x_%j.log

# Perfect-model (twin) experiment, 4 variables, Student-t dof=4, 40 starts: one
# ne256 member replaces the observations (scripts/make_twin_experiment.py).
#   LEVEL=3: member held out of ne256; retrain the top level (lower levels
#            warm-started and frozen), then optimize.
#   LEVEL=2: reuse the trained ne32+ne128 GP, optimize against the same ne256 target;
#            the ne128-ne256 difference then acts as a known structural error.
# Needs TAG (short name) and MEMBER (full name). Runs on the GPU allocation for its
# CPUs because the e3sm CPU allocation is exhausted; no GPU code is used.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
: "${TAG:?set TAG}" "${MEMBER:?set MEMBER}" "${LEVEL:?set LEVEL=2 or 3}"
TWIN=$S/twin_preprocess/${TAG}_${LEVEL}level
OUT=$S/results_twin_${TAG}_${LEVEL}level
set -e
echo "Start: $(date)  $TAG LEVEL=$LEVEL"; START=$(date +%s)
mkdir -p "$TWIN" "$OUT"
if [ "$LEVEL" = 3 ]; then
    BASE=configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml
    HOLD=--hold-out
else
    BASE=configs/perlmutter_mf_ne128_ne32_prod_v2_studentt_dof4_annual.yaml
    HOLD=
    ln -sfn "$S/results_mf_ne128_ne32_prod_v2/mf_gp_trained.pkl" "$OUT/mf_gp_trained.pkl"
fi
$PY scripts/make_twin_experiment.py --base-config $BASE --truth-dir $P/preprocess_output_ne256_n85 \
    --member "$MEMBER" $HOLD --twin-dir "$TWIN" --output-dir "$OUT" --out-config "$TWIN/config.yaml"
if [ "$LEVEL" = 3 ]; then
    OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 $PY scripts/run_two_stage.py \
        --config "$TWIN/config.yaml" --stage 2 --skip-optimize
    echo "Trained: $(date)"
fi
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY scripts/run_two_stage.py \
    --config "$TWIN/config.yaml" --stage 2 --skip-gp --n-xstarts 40
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
