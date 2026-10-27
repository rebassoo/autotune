#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -J prep_10var
#SBATCH -o logs/preprocess_10var_%j.log
#SBATCH -e logs/preprocess_10var_%j.log

# Stage 1 with the 10 observables (PCP, OSR, OLR, TLWP + NETTOA, TAS_LAND, SLP,
# U200, U850, Z500) for one resolution. The ERA5 obs files must exist first
# (scripts/make_era5_obs_ne32pg2.py).
#   sbatch --export=ALL,RES=ne32_prod_v2,EXPECT=984  jobs/preprocess_10var.sh
#   sbatch --export=ALL,RES=ne128_prod_v2,EXPECT=227 jobs/preprocess_10var.sh
#   sbatch --export=ALL,RES=ne256_n85,EXPECT=85      jobs/preprocess_10var.sh
set -euo pipefail
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export PREPROCESS_WORKERS=24
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
CFG=configs/perlmutter_${RES}_10var_annual.yaml
PRE=$($PY -c "import sys; sys.path.insert(0,'src'); from autotune_gp.config import load_config; print(load_config('$CFG').paths.preprocess_dir)")

echo "Start: $(date)  config $CFG"
START=$(date +%s)
$PY scripts/run_two_stage.py --config $CFG --stage 1

# Same member count as the 4-variable preprocessing, or stop (an unreadable
# member is skipped silently by stage 1).
N=$($PY -c "import pickle; print(len(pickle.load(open('$PRE/run_list.pkl','rb'))['sim_names']))")
echo "members preprocessed: $N (expected $EXPECT)"
[ "$N" -eq "$EXPECT" ] || { echo "ERROR: member count"; exit 1; }
END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 60 ))m $(( (END - START) % 60 ))s"
