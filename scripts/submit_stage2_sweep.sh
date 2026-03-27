#!/bin/bash
#SBATCH --account=e3sm
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --job-name=autotune-gp-sweep
#SBATCH --output=logs/autotune-gp-sweep-%A-%a.out
#SBATCH --error=logs/autotune-gp-sweep-%A-%a.err
#SBATCH --array=50-59

set -euo pipefail

module load conda
conda activate /pscratch/sd/r/rebassoo/conda/ESEm_env.yml

cd /global/u2/r/rebassoo/work/2026_02_02_Autotuning-Repo/autotune-gp

export OMP_NUM_THREADS=1

echo "Running seed=${SLURM_ARRAY_TASK_ID} on $(hostname)"

python scripts/run_two_stage.py \
    --config configs/scream_autocal.yaml \
    --stage 2 \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --n-xstarts 10
