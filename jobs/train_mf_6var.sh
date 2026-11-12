#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J mf_6var
#SBATCH -o logs/train_mf_6var_%j.log
#SBATCH -e logs/train_mf_6var_%j.log

# Train the surrogate for the six observables added in the 10-variable
# preprocessing (NETTOA, TAS_LAND, SLP, U200, U850, Z500), to evaluate the
# 19-parameter optimum on them. LEVEL picks the step:
#   2 : ne32 + ne128 AR GP (no k-fold; ~10 h)
#   3 : ne256 top level, warm-started from LEVEL=2's GP (~1 h)
# Submit LEVEL=3 with --dependency=afterok:<LEVEL=2 job id>.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
case ${LEVEL:?set LEVEL=2 or 3} in
    2) CFG=configs/perlmutter_mf_ne128_ne32_prod_v2_6var_annual.yaml ;;
    3) CFG=configs/perlmutter_mf3_ne256_n85_6var_annual.yaml ;;
    *) echo "unknown LEVEL=$LEVEL"; exit 1 ;;
esac
echo "Start: $(date)  LEVEL=$LEVEL"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config $CFG --stage 2 --skip-optimize
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
