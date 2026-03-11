#!/bin/bash
#SBATCH --account=e3sm
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --job-name=autotune-gp-stage2
#SBATCH --output=logs/autotune-gp-stage2-%j.out
#SBATCH --error=logs/autotune-gp-stage2-%j.err

set -euo pipefail

conda activate /pscratch/sd/r/rebassoo/conda/ESEm_env.yml

cd /global/u2/r/rebassoo/work/2026_02_02_Autotuning-Repo/autotune-gp

export OMP_NUM_THREADS=1

python scripts/run_two_stage.py --config configs/scream_autocal.yaml --stage 2
