#!/usr/bin/env bash
#SBATCH --job-name=probe
#SBATCH --partition=gpu-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j_probe.out
#SBATCH --error=logs/%j_probe.err

source /etc/profile
set -e

module load python/3.11.11
module load cuda/12.8

source "$HOME/venvs/numerals/bin/activate"

mkdir -p logs

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:  $(date)"
echo

cd "$SLURM_SUBMIT_DIR"

# ── Probe existing checkpoints ───────────────────────────────────────────────
# Run probes on the best carry_explicit checkpoint.
# Adjust --ckpt and --scaffold for other models.

python3 probe.py \
  --ckpt       checkpoints/sft_carry_explicit_small_step160000.pt \
  --scaffold   carry_explicit \
  --model_size small \
  --max_len    64 \
  --positions  colon equals plus

echo
echo "Finished: $(date)"
