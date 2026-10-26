#!/bin/bash
# Rerun of the n=85 part of submit_n85_plan.sh. The first submission read only
# 72 members (the near-balance files were not readable to us); its outputs were
# renamed *_n72iso*. Run from the repo root:  bash jobs/submit_n85_rerun.sh
# Writes every job id to logs/submitted_n85.txt.
set -euo pipefail
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
S=/pscratch/sd/r/rebassoo/autotune
LOG=logs/submitted_n85.txt
sb() { sbatch --parsable "$@"; }
rec() { echo "$1=$2" | tee -a "$LOG"; }
echo "# rerun submitted $(date)" >> "$LOG"

# ---- 1. n=85 preprocessing + out-of-sample check of the n=72 surrogates -----
PRE=$(sb jobs/preprocess_ne256_n85.sh);                                   rec PRE $PRE
VAL=$(sb --dependency=afterok:$PRE jobs/validate_n72_on_nearbal.sh);      rec VAL $VAL

# ---- 2. n=85 surrogates (top kernel iso) and their 40-start optimizations ---
W85=$(sb --dependency=afterok:$PRE jobs/stage2a_mf3_ne256_n85_train.sh);  rec W85 $W85
W85O=$(sb --dependency=afterok:$W85 jobs/stage2b_mf3_ne256_n85_studentt_dof4_40starts.sh); rec W85O $W85O
C85=$(sb --dependency=afterok:$PRE jobs/stage2a_mf3_ne256_n85_cold_train.sh); rec C85 $C85
C85O=$(sb --dependency=afterok:$C85 jobs/stage2b_mf3_ne256_n85_cold_studentt.sh); rec C85O $C85O

# ---- 3. n=85 holdouts ----------------------------------------------------------
H85I=$(sb --dependency=afterok:$PRE jobs/holdout_ne256_n85_3level_iso.sh);  rec H85I $H85I
H85C=$(sb --dependency=afterok:$PRE jobs/holdout_ne256_n85_3level_cold.sh); rec H85C $H85C
H85A=$(sb --dependency=afterok:$PRE jobs/holdout_ne256_n85_alone.sh);       rec H85A $H85A

# ---- 4. model-error calibration on the warm n=85 surrogate: 3 global treatments
# All start from sigma = 0; zonal/regional sigma from residuals each round.
WCFG=configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml
WDIR=$S/results_mf3_ne256_n85_studentt_dof4
# E1: global term removed from the cost; runs its own sigma=0 round 0.
E1=$(sb --dependency=afterok:$W85 --export=ALL,CONFIG=configs/perlmutter_mf3_ne256_n85_studentt_dof4_noglobal_annual.yaml,OUT=$S/calib_n85_E1_noglobal,GP_PKL=$WDIR/mf_gp_trained.pkl,INIT_ARGS="--init-frac 0",GLOBAL_ARGS=none,SENS=0 \
     jobs/calibrate_model_error_mf3.sh);                                  rec E1 $E1
# E2: fixed observation-based global error (physical units; see the plan for sources).
E2=$(sb --dependency=afterok:$W85O --export=ALL,CONFIG=$WCFG,OUT=$S/calib_n85_E2_obsglobal,ROUND0=$WDIR,GLOBAL_ARGS="--global OSR=2.5 --global OLR=2.5 --global PCP=3.2e-9 --global TLWP=0.012",SENS=0 \
     jobs/calibrate_model_error_mf3.sh);                                  rec E2 $E2
# E3: zero global error (GP variance only).
E3=$(sb --dependency=afterok:$W85O --export=ALL,CONFIG=$WCFG,OUT=$S/calib_n85_E3_zeroglobal,ROUND0=$WDIR,GLOBAL_ARGS=none,SENS=0 \
     jobs/calibrate_model_error_mf3.sh);                                  rec E3 $E3

squeue -u "$USER" -o "%.10i %.22j %.8T %.30R"
