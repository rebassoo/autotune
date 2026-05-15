#!/bin/bash
#PBS -l select=1:ncpus=32
#PBS -l walltime=02:00:00
#PBS -A e3sm_dec          # adjust if your Aurora allocation name differs
#PBS -q workq
#PBS -N remap-ne30-ne32
#PBS -o logs/remap-ne30-ne32.out
#PBS -e logs/remap-ne30-ne32.err

set -euo pipefail

LOGDIR="$(dirname "$0")/../logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/remap-ne30-ne32-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1
echo "Log: ${LOGFILE}"

CASEDIRS=/flare/E3SM_Dec/prod/ppe-20251106/casedirs20251223
OUTDIR=/flare/E3SM_Dec/rebassoo/remapped_ne32pg2
MAP=/home/rebassoo/map_ne30pg2_to_ne32pg2.nc
#NPAR=32
NPAR=8

# Activate ncremap env before running if not already active:
#   mamba activate /flare/E3SM_Dec/rebassoo/conda/ncremap_env
command -v ncremap &>/dev/null || { echo "ERROR: ncremap not found — activate the ncremap_env first"; exit 1; }

mkdir -p "${OUTDIR}"

# Build list of (infile, outfile) pairs for all casedirs that have ne30pg2 output
tmplist=$(mktemp)
for casedir in "${CASEDIRS}"/*/; do
    files=("${casedir}run/"1ma_ne30pg2.AVERAGE.nmonths_x1.*.nc)
    [[ -f "${files[0]}" ]] || continue
    run=$(basename "${casedir}")
    mkdir -p "${OUTDIR}/${run}/run"
    for infile in "${files[@]}"; do
        fname=$(basename "${infile}")
        outfile="${OUTDIR}/${run}/run/${fname/ne30pg2/ne32pg2}"
        [[ -f "${outfile}" ]] && continue   # skip already-done files
        echo "${infile} ${outfile}"
    done
done > "${tmplist}"

total=$(wc -l < "${tmplist}")
echo "Remapping ${total} files with ${NPAR} parallel workers..."

# Run ncremap in parallel with a fixed-width job pool
while IFS=' ' read -r infile outfile; do
    while (( $(jobs -r | wc -l) >= NPAR )); do sleep 0.2; done
    ncremap -P eamxx -m "${MAP}" -i "${infile}" -o "${outfile}" \
        && echo "  done: $(basename "${infile}")" \
        || echo "  FAILED: ${infile}" >&2 &
done < "${tmplist}"
wait

rm -f "${tmplist}"
echo "All done. Output in ${OUTDIR}"
