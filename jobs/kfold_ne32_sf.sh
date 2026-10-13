#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -J kfold_ne32
#SBATCH -o logs/kfold_ne32_sf_%j.log
#SBATCH -e logs/kfold_ne32_sf_%j.log

# Single-fidelity ne32 surrogate skill: 5-fold R^2 on the 984-member ne32
# ensemble, for comparison with the ne128 and ne256 numbers.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

echo "Start: $(date)"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/evaluate_surrogate.py \
    --config configs/perlmutter_ne32_prod_v2_annual.yaml
END=$(date +%s); echo "End: $(date)"; echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
