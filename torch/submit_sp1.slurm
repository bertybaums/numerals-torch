#!/usr/bin/env bash
#SBATCH --job-name=sp1
#SBATCH --partition=gpu-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# SP1 capacity-floor sweep: parameterized by MODEL_SIZE and NOTATION via --export.
# Submit via submit_sp1.sh (which fills in the export vars and the job name).

source /etc/profile
set -e

module load python/3.11.11
module load cuda/12.8

source "$HOME/venvs/numerals/bin/activate"

mkdir -p logs checkpoints

: "${MODEL_SIZE:?MODEL_SIZE must be set via --export}"
: "${NOTATION:?NOTATION must be set via --export}"

echo "Job ID:     $SLURM_JOB_ID"
echo "Job name:   $SLURM_JOB_NAME"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "NOTATION:   $NOTATION"
echo "Started:    $(date)"
echo

cd "$SLURM_SUBMIT_DIR"

CKPT_TAG="sp1_${MODEL_SIZE}_${NOTATION}"

python3 train_tooluse.py \
  --mode       sft \
  --model_size "$MODEL_SIZE" \
  --notation   "$NOTATION" \
  --max_steps  80000 \
  --max_len    80 \
  --batch_size 128 \
  --lr         3e-4 \
  --dropout    0.1 \
  --mask_loss \
  --log_every  1000 \
  --save_every 20000 \
  --ckpt_tag   "$CKPT_TAG"

echo
echo "Training finished: $(date)"
echo

# Evaluate the final checkpoint on the full test set (all 4 notation pairs)
FINAL_CKPT="checkpoints/tooluse_sft_${CKPT_TAG}_step80000.pt"

if [ -f "$FINAL_CKPT" ]; then
  echo "Evaluating $FINAL_CKPT on full test set..."
  python3 evaluate_tooluse.py \
    --ckpt       "$FINAL_CKPT" \
    --model_size "$MODEL_SIZE" \
    --max_len    80 \
    > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_eval.txt"
  echo "Eval written to logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_eval.txt"
else
  echo "WARNING: expected checkpoint not found: $FINAL_CKPT"
fi

echo
echo "Finished: $(date)"
