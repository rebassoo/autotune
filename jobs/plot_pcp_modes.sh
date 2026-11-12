#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J pcp_modes
#SBATCH -o logs/plot_pcp_modes_%j.log
#SBATCH -e logs/plot_pcp_modes_%j.log

# Precip PPE EOFs and the reachable change at the MF optimum (scripts/plot_pcp_modes.py).
# Needs a compute node: the 3-level GP exceeds the login-node memory limit.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_studentt_dof4
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
$PY scripts/plot_pcp_modes.py --top-dir $P/preprocess_output_ne256_n85 \
    --scalers-dir $P/preprocess_output_ne128_prod_v2 --gp-pkl $S/mf_gp_trained.pkl \
    --results-dir $S --pcponly-dir /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_n85_pcponly \
    --out docs/pcp_modes_n85.png
if [ -f logs/may_vs_now_scratch.py ]; then echo "==== May vs now"; $PY logs/may_vs_now_scratch.py; fi
