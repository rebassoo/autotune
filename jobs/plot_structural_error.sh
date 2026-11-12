#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J plot_structerr
#SBATCH -o logs/plot_structural_error_%j.log
#SBATCH -e logs/plot_structural_error_%j.log

# Misfit at the uncalibrated optimum vs. PPE spread and GP sd (the evidence for
# the per-variable model-error sigma). Needs a compute node: loading the
# 3-level GP exceeds the login-node memory limit.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
# TAG=n72iso (default) or n85; e.g.  sbatch --export=ALL,TAG=n85 jobs/plot_structural_error.sh
# TAG=pcponly or pcpg0: the n=85 PCP-weight runs (optimum read from the run's
# projection_data_seed50.pkl instead of a calibration log).
TAG=${TAG:-n72iso}
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
GP85=$S/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl
case $TAG in
    n72iso) TOP=$P/preprocess_output_ne256_n85_was72; N=72 ;;
    n85)    TOP=$P/preprocess_output_ne256_n85;       N=85 ;;
    pcponly) TOP=$P/preprocess_output_ne256_n85; N=85; WHAT="PCP only (other variables weight 0)" ;;
    pcpg0)   TOP=$P/preprocess_output_ne256_n85; N=85; WHAT="PCP global weight 0 (zonal/regional 0.5)" ;;
    *) echo "unknown TAG=$TAG"; exit 1 ;;
esac
if [ -n "$WHAT" ]; then
    THETA=$($PY -c "import pickle,json; print(json.dumps(pickle.load(open('$S/results_mf3_ne256_n85_$TAG/diagnostics/projection_data_seed50.pkl','rb'))['best_params_norm'].tolist()))")
    $PY scripts/plot_structural_error.py --theta "$THETA" --gp-pkl $GP85 --top-dir $TOP \
        --scalers-dir $P/preprocess_output_ne128_prod_v2 \
        --title "Misfit at the optimum vs. PPE spread — n=85, $WHAT" \
        --out docs/structural_error_n85_$TAG
    exit
fi
$PY scripts/plot_structural_error.py \
    --calib-log $S/calib_${TAG}_E3_zeroglobal/calibration_log.json --round 0 \
    --gp-pkl $S/results_mf3_ne256_${TAG}_studentt_dof4/mf_gp_trained.pkl \
    --top-dir $TOP \
    --scalers-dir $P/preprocess_output_ne128_prod_v2 \
    --title "Misfit at the optimum vs. PPE spread — n=$N, warm start, isotropic top kernel, before model error" \
    --out docs/structural_error_${TAG}
