#!/bin/bash
#SBATCH --account=e3sm
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=autotune-gp-stage1
#SBATCH --output=logs/autotune-gp-stage1-%j.out
#SBATCH --error=logs/autotune-gp-stage1-%j.err

set -euo pipefail

module load conda
conda activate /pscratch/sd/r/rebassoo/conda/ESEm_env

cd /global/u2/r/rebassoo/work/2026_02_02_Autotuning-Repo/autotune-gp

export OMP_NUM_THREADS=1

python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 1
