#!/usr/bin/env bash
#
# SP2(a) sweep: compositional command tokenization, in-distribution match.
# 2 sizes × 2 notations = 4 jobs (we already know large saturates with small from SP1).
#
# Usage:
#   bash submit_sp2a.sh           # submit all 4
#   bash submit_sp2a.sh dry-run   # print sbatch commands without submitting
#
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "dry-run" ]]; then
  DRY_RUN=1
  echo "(dry-run mode — no jobs will be submitted)"
  echo
fi

SIZES=(tiny small)
NOTATIONS=(all hindu)

mkdir -p logs

for size in "${SIZES[@]}"; do
  for notation in "${NOTATIONS[@]}"; do
    JOB_NAME="sp2a_${size}_${notation}_comp"
    CMD=(
      sbatch
      --job-name="$JOB_NAME"
      --export=ALL,MODEL_SIZE="$size",NOTATION="$notation"
      submit_sp2a.slurm
    )
    if (( DRY_RUN )); then
      printf '  %s\n' "${CMD[*]}"
    else
      "${CMD[@]}"
    fi
  done
done

if (( DRY_RUN == 0 )); then
  echo
  echo "Submitted. Monitor with:  squeue -u \$USER"
  echo "Logs at:                  logs/sp2a_<size>_<notation>_comp_<jobid>.out"
fi
