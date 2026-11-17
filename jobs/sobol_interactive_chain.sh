#!/bin/bash
# Run the regional Sobol studies one after another in interactive sessions
# (Perlmutter interactive QOS, 4 h each), resuming any that a session did not finish:
#   1. 3-level (scripts/regional_sobol.py via jobs/regional_sobol_interactive.sh)
#   2. ne128+ne32 2-level, 3. ne32-only (train + global + regional, jobs/sobol_surrogate.sh)
# Up to MAX_SESSIONS sessions per study; a study counts as done when its npz has all
# boxes. Waits first for WAIT_JOB (an allocation already running a study) to end.
# Hold the batch versions meanwhile: they write the same files.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
S=/pscratch/sd/r/rebassoo/autotune
P=/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2
PY=/pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python
MAX_SESSIONS=${MAX_SESSIONS:-3}
done_boxes() {   # prints done/total for a regional_sobol.npz, or 0/0
    $PY -c "import numpy as np,sys
try: d=np.load(sys.argv[1]); print(int(d['done'].sum()), len(d['done']))
except Exception: print(0, 0)" "$1"
}
is_done() { read -r a b < <(done_boxes "$1"); [ "$b" -gt 0 ] && [ "$a" -eq "$b" ]; }
run_study() {   # name, npz, command...
    local name=$1 npz=$2; shift 2
    for s in $(seq 1 $MAX_SESSIONS); do
        if is_done "$npz"; then echo "[$(date +%H:%M)] $name: complete"; return 0; fi
        echo "[$(date +%H:%M)] $name: session $s ($(done_boxes "$npz") boxes done)"
        "$@" > logs/sobol_interactive_${name}_$s.log 2>&1 || echo "[$(date +%H:%M)] $name: session $s exited non-zero"
    done
    is_done "$npz" && echo "[$(date +%H:%M)] $name: complete" || echo "[$(date +%H:%M)] $name: NOT complete after $MAX_SESSIONS sessions"
}
interactive() {   # NAME GP_PKL [TRAIN]: one 4 h session of jobs/sobol_surrogate.sh
    salloc -A e3sm_g -C gpu -q interactive -N 1 --gpus-per-node=1 -c 128 -t 04:00:00 \
        srun -n 1 -c 128 --cpu-bind=none env NAME=$1 GP_PKL=$2 TRAIN=$3 bash jobs/sobol_surrogate.sh
}
if [ -n "$WAIT_JOB" ]; then
    while squeue -h -j "$WAIT_JOB" 2>/dev/null | grep -q .; do sleep 120; done
    echo "[$(date +%H:%M)] job $WAIT_JOB finished"
fi
run_study 3level $S/regional_sobol_n85/regional_sobol.npz bash jobs/regional_sobol_interactive.sh
run_study ne128ne32 $S/regional_sobol_n85_ne128ne32/regional_sobol.npz \
    interactive ne128ne32 $S/results_mf_ne128_ne32_prod_v2/mf_gp_trained.pkl ""
run_study ne32 $S/regional_sobol_n85_ne32/regional_sobol.npz \
    interactive ne32 $S/results_ne32_prod_v2_sf_gpy/sf_gp_gpy_trained.pkl $P/preprocess_output_ne32_prod_v2
echo "[$(date +%H:%M)] chain finished"
