#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 01:00:00
#SBATCH -J eval_cost_points
#SBATCH -o logs/eval_cost_points_%j.log
#SBATCH -e logs/eval_cost_points_%j.log

# Costs at named points (scripts/eval_cost_points.py): the nominal 3-level fit
# (Student-t, real obs) with the 6 low-Sobol parameters at default / fitted / random
# and all 19 at default; and the two 3-level PCA twins with the 6 set to default.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
O=$S/eval_points
$PY scripts/eval_cost_points.py --gp-pkl $S/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl \
    --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml --pca-basis $S/pca_basis_k20_n85.pkl \
    --points $O/nominal.npz --box-bounds $S/sobol_box_bounds_n85.npz --out $O/nominal.json
for t in m060_3level m202_3level; do
    $PY scripts/eval_cost_points.py --gp-pkl $S/results_twin_$t/mf_gp_trained.pkl \
        --config $S/twin_preprocess/$t/config_pca.yaml --pca-basis $S/twin_preprocess/$t/pca_basis_k20.pkl \
        --points $O/$t.npz --box-bounds $S/sobol_box_bounds_n85.npz --out $O/$t.json
done
