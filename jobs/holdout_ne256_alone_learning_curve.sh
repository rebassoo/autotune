#!/bin/bash
# Learning curve of the ne256-alone surrogate (single-fidelity GPy on ne256 only):
# same 5 folds and the same random training subsets (--train-top N, --subset-seed s)
# as jobs/holdout_learning_curve.sh for the 3-level surrogate. About 1 min per run.
# Run inside an allocation (e.g. salloc ... srun bash jobs/holdout_ne256_alone_learning_curve.sh).
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
mkdir -p $S/holdout_lc_n85_alone
for n in 10 20 30 40 50 60; do for s in 0 1 2; do
    out=$S/holdout_lc_n85_alone/train${n}_seed${s}
    [ -f $out/ne256_alone_holdout.pkl ] && continue
    $PY scripts/evaluate_ne256_alone_holdout.py --top-dir $P/preprocess_output_ne256_n85 \
        --scalers-dir $P/preprocess_output_ne128_prod_v2 --folds 5 --workers 5 \
        --train-top $n --subset-seed $s --out-dir $out > $S/holdout_lc_n85_alone/log_train${n}_seed${s}.txt 2>&1
    echo "[$(date +%H:%M)] N=$n seed=$s: $(grep -A5 'ne256 alone (single' $S/holdout_lc_n85_alone/log_train${n}_seed${s}.txt | grep -E 'PCP|OSR|OLR|TLWP' | awk '{printf "%s %s  ", $1, $2}')"
done; done
