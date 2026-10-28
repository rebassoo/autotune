#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -J mf3_n85_pcpw
#SBATCH -o logs/stage2b_mf3_ne256_n85_pcpweights_%j.log
#SBATCH -e logs/stage2b_mf3_ne256_n85_pcpweights_%j.log

# Re-optimize the n=85 warm/iso surrogate (Student-t dof=4, 40 starts, no
# model error) with different PCP weights. EXP picks the config:
#   pcponly : PCP weight 1, OSR/OLR/TLWP 0; zonal/regional/global 1/3 each
#   pcpg015 : all four variables; PCP only: zonal/regional 0.425, global 0.15
#   pcpg0   : all four variables; PCP only: zonal/regional 0.5, global 0
# Compare with results_mf3_ne256_n85_studentt_dof4 (all 1/3, best cost 0.1989).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
S=/pscratch/sd/r/rebassoo/autotune
OUT=$S/results_mf3_ne256_n85_${EXP:?set EXP=pcponly, pcpg015 or pcpg0}
mkdir -p "$OUT"
ln -sfn $S/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl "$OUT/mf_gp_trained.pkl"

echo "Start: $(date)  EXP=$EXP"; START=$(date +%s)
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf3_ne256_n85_${EXP}_annual.yaml \
    --stage 2 --skip-gp --n-xstarts 40 --output-dir "$OUT"
END=$(date +%s); echo "End: $(date)"
echo "Elapsed: $(( (END-START)/3600 ))h $(( (END-START)%3600/60 ))m"
