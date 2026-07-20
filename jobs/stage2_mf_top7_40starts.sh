#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 06:00:00
#SBATCH -J stage2_mf_top7
#SBATCH -o logs/stage2_mf_top7_40starts_%j.log
#SBATCH -e logs/stage2_mf_top7_40starts_%j.log

# Subset optimization: vary only the 7 union top-parameters (top-6 by Pearson
# + top-6 by GP cost-sensitivity, unioned), freezing the other 12 at the
# control-run default (clipped to the sampled range). Reuses the existing
# 19-param GP via --skip-gp -- NO retraining, no k-fold. 40 starts via the
# process executor (executor=process, max_workers=32 in the config), so this
# is only the optimize phase, ~1-2h.
#
# The output dir must already contain the 19-param mf_gp_trained.pkl (hardlinked
# from an existing MF run) for --skip-gp to load.
#
# Runs from the autotune-parallel worktree (parallel optimizer + --vary-params).
cd /global/u2/r/rebassoo/work/2026_05_11_MultiFidelityAutotuning/autotune-parallel

export PYTHONUNBUFFERED=1
# One BLAS thread per worker (32 workers x threaded BLAS would oversubscribe);
# optimize_parallel also enforces this via threadpoolctl.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date)"
START=$(date +%s)

/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/run_two_stage.py \
    --config configs/perlmutter_mf_ne128_ne32_prod_annual.yaml \
    --stage 2 \
    --output-dir /pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_top7_40starts \
    --n-xstarts 40 \
    --skip-gp \
    --vary-params length_fac,coeff_kh,autoconversion_qc_exponent,ice_sedimentation_factor,qw2tune,spa_ccn_to_nc_factor,max_total_ni

END=$(date +%s)
echo "End: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( (END - START) % 3600 / 60 ))m $(( (END - START) % 60 ))s"
