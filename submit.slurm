#!/usr/bin/env bash
#SBATCH --job-name=numerals
#SBATCH --partition=gpu-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7:00:00
#SBATCH --output=logs/%j_numerals.out
#SBATCH --error=logs/%j_numerals.err

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

# ── Edit these arguments for each run ─────────────────────────────────────────
#
# Phase 1 — Pretraining:
#   python3 train.py --mode pretrain --scaffold none --model_size small --max_steps 64000
#
# Phase 2 — SFT old scaffold:
#   python3 train.py --mode sft --scaffold old --model_size small --max_steps 80000 \
#     --ckpt_in checkpoints/pretrain_none_small_step64000.pt
#
# Phase 3 — RL v1-v5 (example v1):
#   python3 train.py --mode rl --scaffold old --rl_version 1 --max_steps 10000 \
#     --ckpt_in checkpoints/sft_old_small_step80000.pt
#
# Phase 4 — SFT state-seq:
#   python3 train.py --mode sft --scaffold state_seq --model_size small --max_steps 80000 \
#     --ckpt_in checkpoints/pretrain_none_small_step64000.pt
#
# Phase 5 — RL Phase 2 v3 (the stable fix):
#   python3 train.py --mode rl --scaffold state_seq --rl_version 5 \
#     --kl_coef 0.1 --sft_mix_coef 1.0 --max_steps 20000 \
#     --ckpt_in checkpoints/sft_state_seq_small_step80000.pt
#
# Phase 6 — Large model (use --time=12:00:00 or gpu-8-long):
#   python3 train.py --mode sft --scaffold state_seq --model_size large --max_steps 200000
#
# Phase 7 — Decomposition:
#   python3 train.py --mode sft --scaffold decomp --model_size small --max_steps 80000 \
#     --ckpt_in checkpoints/pretrain_none_small_step64000.pt
#
# Phase 8 — Carry-explicit (80K then 160K):
#   python3 train.py --mode sft --scaffold carry_explicit --model_size small --max_steps 80000 \
#     --ckpt_in checkpoints/pretrain_none_small_step64000.pt
#
# Phase 9 — Digit scaffold (1-999, max_len=128):
#   python3 train.py --mode sft --scaffold digit --model_size small \
#     --max_steps 80000 --max_len 128
#
# ─────────────────────────────────────────────────────────────────────────────

python3 train.py \
  --mode       rl \
  --scaffold   old \
  --rl_version 1 \
  --model_size small \
  --max_steps  10000 \
  --lr         3e-4 \
  --max_len    64 \
  --ckpt_in    checkpoints/sft_old_small_step80000.pt

echo
echo "Finished: $(date)"
