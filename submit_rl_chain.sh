#!/usr/bin/env bash
# Submit RL v2-v5 as a dependency chain — each starts after the previous finishes.
# Usage: bash submit_rl_chain.sh

set -e
cd ~/Numerals/torch
mkdir -p logs checkpoints

SFT_CKPT="checkpoints/sft_old_small_step80000.pt"

BASE="--model_size small --max_len 64 --lr 3e-4 --max_steps 10000 --scaffold old"

# v2: {0,+1} rewards + entropy bonus
J2=$(sbatch --parsable --job-name=rl_v2 \
    --wrap="source /etc/profile; module load python/3.11.11 cuda/12.8; \
            source \$HOME/venvs/numerals/bin/activate; cd ~/Numerals/torch; \
            python3 train.py --mode rl --rl_version 2 $BASE \
            --ckpt_in $SFT_CKPT" \
    --partition=gpu-8 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output=logs/%j_rl_v2.out --error=logs/%j_rl_v2.err)
echo "Submitted RL v2: job $J2"

# v3: {0,+1} rewards, no entropy, EMA baseline
J3=$(sbatch --parsable --job-name=rl_v3 --dependency=afterok:$J2 \
    --wrap="source /etc/profile; module load python/3.11.11 cuda/12.8; \
            source \$HOME/venvs/numerals/bin/activate; cd ~/Numerals/torch; \
            python3 train.py --mode rl --rl_version 3 $BASE \
            --ckpt_in $SFT_CKPT" \
    --partition=gpu-8 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output=logs/%j_rl_v3.out --error=logs/%j_rl_v3.err)
echo "Submitted RL v3: job $J3 (after $J2)"

# v4: baseline floor + process reward
J4=$(sbatch --parsable --job-name=rl_v4 --dependency=afterok:$J3 \
    --wrap="source /etc/profile; module load python/3.11.11 cuda/12.8; \
            source \$HOME/venvs/numerals/bin/activate; cd ~/Numerals/torch; \
            python3 train.py --mode rl --rl_version 4 $BASE \
            --ckpt_in $SFT_CKPT" \
    --partition=gpu-8 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output=logs/%j_rl_v4.out --error=logs/%j_rl_v4.err)
echo "Submitted RL v4: job $J4 (after $J3)"

# v5: KL anchoring
J5=$(sbatch --parsable --job-name=rl_v5 --dependency=afterok:$J4 \
    --wrap="source /etc/profile; module load python/3.11.11 cuda/12.8; \
            source \$HOME/venvs/numerals/bin/activate; cd ~/Numerals/torch; \
            python3 train.py --mode rl --rl_version 5 --kl_coef 0.1 $BASE \
            --ckpt_in $SFT_CKPT" \
    --partition=gpu-8 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output=logs/%j_rl_v5.out --error=logs/%j_rl_v5.err)
echo "Submitted RL v5: job $J5 (after $J4)"

echo ""
echo "Chain: $J2 → $J3 → $J4 → $J5"
echo "Monitor with: squeue -u bbaum"
