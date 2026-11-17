#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 08:00:00
#SBATCH -J regional_sobol
#SBATCH -o logs/regional_sobol_%j.log
#SBATCH -e logs/regional_sobol_%j.log

# Regional Sobol study (scripts/regional_sobol.py): 5000 random boxes, one of 10
# intervals per parameter, 256-point Saltelli sample of the 3-level surrogate in each;
# saves indices, absolute effects and box-centre costs. Checkpointed every 100 boxes:
# resubmitting resumes. CPU only; on the GPU allocation because e3sm is exhausted.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
S=/pscratch/sd/r/rebassoo/autotune
echo "Start: $(date)"
/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/regional_sobol.py \
    --gp-pkl $S/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl \
    --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \
    --pca-basis $S/pca_basis_k20_n85.pkl \
    --n-boxes ${NBOX:-5000} --workers ${WORKERS:-32} --out $S/regional_sobol_n85
echo "End: $(date)"
