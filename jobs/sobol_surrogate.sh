#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 10:00:00
#SBATCH -o logs/sobol_surrogate_%x_%j.log
#SBATCH -e logs/sobol_surrogate_%x_%j.log

# Global and regional Sobol studies for one surrogate, to compare surrogates trained
# on different resolutions over the same box (sobol_box_bounds_n85.npz) with the
# lambdas held at the 3-level best fit (global study) and the same 5000 boxes
# (regional study, same seed).
#   NAME    output suffix, e.g. ne32 or ne128ne32
#   GP_PKL  the surrogate pickle
#   TRAIN   set to a preprocess dir to first train a single-fidelity GPy surrogate
#           on it (ne128-normalized outputs) and write GP_PKL
# The regional study is checkpointed; resubmitting resumes it.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
CFG=configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml
: "${NAME:?set NAME}" "${GP_PKL:?set GP_PKL}"
set -e
echo "Start: $(date)  $NAME  $GP_PKL"
if [ -n "$TRAIN" ] && [ ! -f "$GP_PKL" ]; then
    $PY scripts/train_sf_gpy.py --preprocess-dir "$TRAIN" \
        --y-scalers-dir $P/preprocess_output_ne128_prod_v2 --workers 64 --out "$GP_PKL"
    echo "Trained: $(date)"
fi
if [ ! -f $S/sobol_n85_$NAME/sobol.pkl ]; then
    OMP_NUM_THREADS=32 $PY scripts/sobol_sensitivity.py --gp-pkl "$GP_PKL" --config $CFG \
        --theta-from $S/results_mf3_ne256_n85_studentt_dof4 --box-bounds $S/sobol_box_bounds_n85.npz \
        --n-base 2048 --out $S/sobol_n85_$NAME
    echo "Global Sobol done: $(date)"
fi
$PY scripts/regional_sobol.py --gp-pkl "$GP_PKL" --config $CFG --pca-basis $S/pca_basis_k20_n85.pkl \
    --box-bounds $S/sobol_box_bounds_n85.npz --n-boxes 5000 --workers 32 --out $S/regional_sobol_n85_$NAME
echo "End: $(date)"
