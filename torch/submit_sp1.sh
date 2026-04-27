#!/usr/bin/env bash
#
# SP1 capacity-floor sweep: 3 model sizes × 2 notations = 6 jobs.
#
# Run from torch/ on fortyfive (login node). Each job submits independently to
# gpu-8 and trains for 80K SFT steps (~1-3h depending on size). Checkpoints
# land in checkpoints/tooluse_sft_sp1_<size>_<notation>_step80000.pt.
#
# Usage:
#   bash submit_sp1.sh           # submit all 6
#   bash submit_sp1.sh dry-run   # print sbatch commands without submitting
#
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "dry-run" ]]; then
  DRY_RUN=1
  echo "(dry-run mode — no jobs will be submitted)"
  echo
fi

SIZES=(tiny small large)
NOTATIONS=(all hindu)

mkdir -p logs

for size in "${SIZES[@]}"; do
  for notation in "${NOTATIONS[@]}"; do
    JOB_NAME="sp1_${size}_${notation}"
    CMD=(
      sbatch
      --job-name="$JOB_NAME"
      --export=ALL,MODEL_SIZE="$size",NOTATION="$notation"
      submit_sp1.slurm
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
  echo "Logs at:                  logs/sp1_<size>_<notation>_<jobid>.out"
fi
