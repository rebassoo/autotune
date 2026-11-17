#!/bin/bash
# Run (or resume) the 3-level regional Sobol study in an interactive allocation
# (Perlmutter interactive QOS, at most 4 h). The study checkpoints every 100 boxes
# to $S/regional_sobol_n85/regional_sobol.npz, so rerunning this script after a
# session ends continues where it stopped. Usage, from the repo root:
#     bash jobs/regional_sobol_interactive.sh            # 4 h session
#     HOURS=2 bash jobs/regional_sobol_interactive.sh
# Hold the batch job (jobs/regional_sobol.sh) meanwhile: both write the same file.
cd /global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/autotune
S=/pscratch/sd/r/rebassoo/autotune
salloc -A e3sm_g -C gpu -q interactive -N 1 --gpus-per-node=1 -c 128 -t ${HOURS:-4}:00:00 \
    srun -n 1 -c 128 --cpu-bind=none env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PYTHONPATH="$(pwd)/src:$(pwd)" \
    /pscratch/sd/r/rebassoo/conda/ESEm_env/bin/python scripts/regional_sobol.py \
        --gp-pkl $S/results_mf3_ne256_n85_studentt_dof4/mf_gp_trained.pkl \
        --config configs/perlmutter_mf3_ne256_n85_studentt_dof4_annual.yaml \
        --pca-basis $S/pca_basis_k20_n85.pkl --box-bounds $S/sobol_box_bounds_n85.npz \
        --n-boxes 5000 --workers 32 --out $S/regional_sobol_n85
