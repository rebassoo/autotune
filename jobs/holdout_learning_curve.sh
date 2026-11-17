#!/bin/bash
#SBATCH -A e3sm_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH -J holdout_lc
#SBATCH -o logs/holdout_learning_curve_%j.log
#SBATCH -e logs/holdout_learning_curve_%j.log

# Learning curve of the 3-level surrogate's ne256 holdout skill: the same 5 folds
# over the 85 ne256 members as results_ne256_n85_3level_holdout_ISO (seed 42), but
# training each fold on a random N of its 68 ne256 training members (--train-top).
# Every N is scored on the same 85 held-out predictions. N=68 is the existing run.
# One run at a time (5 fold workers x 12 threads each); finished runs are skipped.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
run() {   # N subset_seed
    local out=$S/holdout_lc_n85/train${1}_seed${2}
    [ -f $out/nfidelity_holdout.pkl ] && return 0
    $PY scripts/evaluate_nfidelity_holdout.py \
        --level-dir $P/preprocess_output_ne32_prod_v2 --level-dir $P/preprocess_output_ne128_prod_v2 \
        --top-dir $P/preprocess_output_ne256_n85 --scalers-dir $P/preprocess_output_ne128_prod_v2 \
        --warm-start-gp $S/results_mf_ne128_ne32_prod_v2/mf_gp_trained.pkl --top-kernel iso \
        --folds 5 --workers 5 --train-top $1 --subset-seed $2 --out-dir $out > $S/holdout_lc_n85/log_train${1}_seed${2}.txt 2>&1
    echo "[$(date +%H:%M)] N=$1 seed=$2 done: $(grep -A5 'holdout skill' $S/holdout_lc_n85/log_train${1}_seed${2}.txt | grep -E 'PCP|OSR|OLR|TLWP' | tr -s ' ' | head -4 | tr '\n' ';')"
}
mkdir -p $S/holdout_lc_n85
echo "Start: $(date)"
# One run at a time: two concurrent runs (10 fold workers) ran out of memory.
# Finished runs are skipped, so the job can simply be resubmitted.
for pair in "50 0" "60 0" "20 1" "30 1" "40 1" "20 2" "30 2" "40 2"; do
    run $pair
done
echo "End: $(date)"
