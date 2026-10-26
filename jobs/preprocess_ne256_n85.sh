#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -J prep_ne256_n85
#SBATCH -o logs/preprocess_ne256_n85_%j.log
#SBATCH -e logs/preprocess_ne256_n85_%j.log

# Stage 1 for the ne256 members copied from Frontier to CFS. Only the members
# with all 13 monthly files are used; the copy is still in progress, so re-run
# this as more members land.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export PREPROCESS_WORKERS=24

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_ne256_n85_annual.yaml \
    --stage 1

# Fail (and so stop every afterok job) unless all 85 members were read; an
# unreadable member is skipped silently by stage 1.
N=$(/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python -c "import pickle; print(len(pickle.load(open('/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2/preprocess_output_ne256_n85/run_list.pkl','rb'))['sim_names']))")
echo "members preprocessed: $N"
[ "$N" -eq 85 ] || { echo "ERROR: expected 85 members"; exit 1; }

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 60 ))m $(( (END - START) % 60 ))s"
