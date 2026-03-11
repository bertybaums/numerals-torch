#!/usr/bin/env bash
#SBATCH --job-name=tooluse
#SBATCH --partition=gpu-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output=logs/%j_tooluse.out
#SBATCH --error=logs/%j_tooluse.err

source /etc/profile
set -e

module load python/3.11.11
module load cuda/12.8

source "$HOME/venvs/numerals/bin/activate"

mkdir -p logs checkpoints

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:  $(date)"
echo

cd "$SLURM_SUBMIT_DIR"

# ── Tool-use SFT training ───────────────────────────────────────────────────
# Phase 1: SFT on gold trajectories with medium model
# Expected: ~2-3 hours on A6000

python3 train_tooluse.py \
  --mode       sft \
  --model_size medium \
  --max_steps  80000 \
  --max_len    80 \
  --batch_size 128 \
  --lr         3e-4 \
  --dropout    0.1 \
  --mask_loss \
  --log_every  1000 \
  --save_every 20000

echo
echo "Finished: $(date)"
