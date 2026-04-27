"""
Tool-use training: model learns to operate an external abacus simulator.

The model generates commands (+u5, +t3) and the simulator returns state updates.
This models the historical pattern where the human (model) issues instructions
to the abacus (tool) and reads back results.

Training modes:
  1. SFT on gold trajectories: teacher-forced on the full interleaved sequence
  2. RL with interleaved generation: model generates commands, receives real
     simulator feedback, gets reward based on final correctness

Sequence format (variant A):
  <A> + <B> : [0|4|7] +u5 [0|5|2]^ +t3 [0|8|2] = 82

  - Model generates: '+u5', '+t3', and the final answer '82'
  - Simulator provides: '[0|4|7]', '[0|5|2]^', '[0|8|2]'
  - The '=' token signals end of tool use; model generates answer

Usage:
  # SFT on gold trajectories
  python train_tooluse.py --mode sft --model_size medium --max_steps 80000

  # RL with simulator interaction
  python train_tooluse.py --mode rl --model_size medium --max_steps 20000 \
    --ckpt_in checkpoints/tooluse_sft_medium_step80000.pt

  # Resume from checkpoint
  python train_tooluse.py --mode sft --model_size medium --max_steps 160000 \
    --ckpt_in checkpoints/tooluse_sft_medium_step80000.pt
"""

import argparse
import copy
import os
import random
import re
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID,
    tok2id, id2tok,
    get_train_facts, get_test_facts,
    encode, decode, encode_prompt, fmt,
)
from data_abacus import (
    ABACUS_VOCAB_SIZE, APAD_ID, ABOS_ID, AEOS_ID,
    atok2id, aid2tok,
    aencode, adecode, aencode_prompt,
    make_abacus_expression,
)
from model import build_model, get_device, count_params, load_state_dict_compat
from simulator import AbacusSimulator


# ── Tool-use sequence construction ───────────────────────────────────────────

def make_tooluse_expression(A, B, variant='A', roman_A=False, roman_B=False):
    """
    Build a tool-use training expression.

    Unlike abacus SFT (which uses trace_A etc.), this constructs the sequence
    using the simulator, matching how inference will work.

    Format: '<A> + <B> : [init] +u<n> [state]^ +t<n> [state] = <C>'
    """
    sim = AbacusSimulator()
    traj = sim.gold_trajectory(A, B)
    C = A + B

    # Build the trace portion from the trajectory
    parts = []
    for step in traj:
        if step['command'] is None:
            parts.append(step['response'])  # initial state
        else:
            parts.append(step['command'])
            parts.append(step['response'])

    trace_str = ' '.join(parts)
    return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {trace_str} = {C}"


_NOTATION_PAIRS = {
    'all':   [(False, False), (False, True), (True, False), (True, True)],
    'hindu': [(False, False)],
    'roman': [(True, True)],
}


def make_tooluse_dataset(split='train', variant='A', seed=42, notation='all'):
    """
    Generate tool-use training expressions for the requested notation combos.
    Filters out B > 99 since the command set only has +u/+t (no hundreds rod).

    notation:
      'all'   — all 4 (rA, rB) combinations (default)
      'hindu' — Hindu-Arabic only (no Roman parsing burden)
      'roman' — Roman only (Roman parsing required for both operands)
    """
    if notation not in _NOTATION_PAIRS:
        raise ValueError(f"Unknown notation '{notation}', expected one of {list(_NOTATION_PAIRS)}")
    pairs = _NOTATION_PAIRS[notation]
    facts = get_train_facts() if split == 'train' else get_test_facts()
    exprs = []
    for (A, B) in facts:
        if B > 99:
            continue  # can't decompose into single-digit +u/+t commands
        for (rA, rB) in pairs:
            exprs.append(make_tooluse_expression(A, B, variant, rA, rB))
    return exprs


# ── Masking: which tokens does the model predict? ───────────────────────────

def make_generation_mask(token_ids, vocab_map):
    """
    Create a binary mask indicating which tokens the MODEL generates
    (as opposed to tokens provided by the simulator).

    Model generates:
      - command tokens: +, u, t, and the digit after them
      - everything after '=' (the final answer)

    Simulator provides:
      - initial state [H|T|U]
      - response states after commands [H|T|U] and optional ^

    Returns:
      mask: list of bool, same length as token_ids
    """
    mask = [False] * len(token_ids)
    n = len(token_ids)

    # Find the colon position (scaffold boundary)
    colon_id = vocab_map[':']
    colon_pos = None
    for i, tid in enumerate(token_ids):
        if tid == colon_id:
            colon_pos = i
            break

    if colon_pos is None:
        # No scaffold — mask everything after BOS
        for i in range(1, n):
            mask[i] = True
        return mask

    # Before colon: these are the prompt, not predicted
    # After colon: alternate between simulator-provided and model-generated

    # Find the '=' sign (end of tool use)
    eq_id = vocab_map['=']
    eq_pos = None
    for i in range(n - 1, colon_pos, -1):
        if token_ids[i] == eq_id:
            eq_pos = i
            break

    # After '=': model generates everything (the answer)
    if eq_pos is not None:
        for i in range(eq_pos, n):
            mask[i] = True

    # Between colon and '=': identify command tokens
    # Commands are: +u<d> or +t<d> (3 tokens each)
    # States are: [H|T|U] (7 tokens) optionally followed by ^
    plus_id = vocab_map['+']
    u_id = vocab_map['u']
    t_id = vocab_map['t']

    i = colon_pos + 1
    end = eq_pos if eq_pos is not None else n
    while i < end:
        tid = token_ids[i]
        if tid == plus_id:
            # This is a command: +, u/t, digit — model generates these
            mask[i] = True
            if i + 1 < end:
                mask[i + 1] = True  # u or t
            if i + 2 < end:
                mask[i + 2] = True  # digit
            i += 3
        else:
            # Simulator-provided token (state, ^, space) — not masked
            i += 1

    return mask


# ── SFT dataset with generation mask ────────────────────────────────────────

class ToolUseDataset(torch.utils.data.Dataset):
    """Dataset that encodes tool-use sequences and provides generation masks."""

    def __init__(self, split='train', variant='A', max_len=80, seed=42, notation='all'):
        self.max_len = max_len
        self.strings = make_tooluse_dataset(split, variant, seed, notation=notation)
        self.vocab_map = atok2id  # abacus vocab includes all needed tokens
        self.data = []
        for s in self.strings:
            ids = [ABOS_ID] + [self.vocab_map[c] for c in s] + [AEOS_ID]
            ids = ids[:max_len]
            pad_len = max_len - len(ids)
            ids = ids + [APAD_ID] * pad_len
            self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        seq = torch.tensor(ids, dtype=torch.long)
        input_ids  = seq[:-1]
        target_ids = seq[1:]

        # Generation mask (shifted by 1 to align with targets)
        gen_mask = make_generation_mask(ids, self.vocab_map)
        # For LM loss: we predict token at position t+1 from position t
        # So mask[i] for target[i] = gen_mask[i+1]
        target_mask = torch.tensor(gen_mask[1:], dtype=torch.bool)

        return input_ids, target_ids, target_mask


# ── SFT training ─────────────────────────────────────────────────────────────

def train_sft(model, args, device):
    """Supervised fine-tuning on gold tool-use trajectories."""
    dataset = ToolUseDataset(split='train', variant='A', max_len=args.max_len,
                             notation=args.notation)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Mixed precision context
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if device.type == 'cuda' else nullcontext())

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    model.train()
    step = args.start_step
    epoch = 0

    print(f"\nSFT training: {len(dataset)} examples, batch_size={args.batch_size}")
    print(f"Starting from step {step}, training to step {args.max_steps}")

    while step < args.max_steps:
        epoch += 1
        for input_ids, target_ids, target_mask in loader:
            if step >= args.max_steps:
                break

            input_ids   = input_ids.to(device)
            target_ids  = target_ids.to(device)
            target_mask = target_mask.to(device)

            with amp_ctx:
                logits = model(input_ids)
                # Loss only on model-generated tokens
                loss_all = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    reduction='none',
                )
                loss_all = loss_all.reshape(target_ids.shape)

                if args.mask_loss:
                    # Only backprop through model-generated tokens
                    masked_loss = loss_all * target_mask.float()
                    loss = masked_loss.sum() / target_mask.float().sum().clamp(min=1)
                else:
                    # Standard LM loss on all tokens (including simulator output)
                    pad_mask = (target_ids != APAD_ID).float()
                    loss = (loss_all * pad_mask).sum() / pad_mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                print(f"  step {step:>7d}  loss={loss.item():.4f}")

            if step % args.save_every == 0:
                path = f"checkpoints/tooluse_sft_{args.ckpt_tag}_step{step}.pt"
                torch.save({'model': model.state_dict(), 'step': step,
                            'args': vars(args)}, path)
                print(f"  saved {path}")

    # Final save
    path = f"checkpoints/tooluse_sft_{args.ckpt_tag}_step{step}.pt"
    torch.save({'model': model.state_dict(), 'step': step,
                'args': vars(args)}, path)
    print(f"  saved {path}")
    return step


# ── RL with interleaved generation ───────────────────────────────────────────

@torch.no_grad()
def generate_with_simulator(model, A, B, device, max_len, variant='A',
                            roman_A=False, roman_B=False, temperature=1.0):
    """
    Interactive generation: model generates commands, simulator returns states.

    The model sees:  <prompt> : [init_state]
    Then alternates: model generates command → simulator returns state
    Until model generates '=' → model generates answer

    Returns:
        full_ids: complete token sequence (prompt + interaction + answer)
        log_probs: log probs for model-generated tokens only
        reward: 1.0 if correct, 0.0 otherwise
    """
    sim = AbacusSimulator()
    C = A + B

    # Build prompt
    prompt_str = f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : "
    init_state = sim.reset(A)
    prompt_str += init_state + ' '

    prompt_ids = aencode_prompt(prompt_str, max_len).unsqueeze(0).to(device)
    ids = prompt_ids.clone()

    log_probs = []
    eq_id = atok2id['=']
    eos_id = AEOS_ID
    space_id = atok2id[' ']

    model.eval()
    state_str = init_state
    max_commands = 10  # safety limit

    for cmd_num in range(max_commands):
        # Model generates a command (e.g., +u5 = 3 tokens, or = for end)
        cmd_tokens = []
        for tok_idx in range(4):  # max 4 tokens per command (space + 3)
            if ids.shape[1] >= max_len:
                break
            ids_cond = ids[:, -model.block_size:]
            logits = model(ids_cond)[:, -1, :]
            probs = (logits / max(temperature, 1e-8)).softmax(-1)

            if temperature <= 0 or temperature == 1e-8:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            lp = torch.log(probs[0, next_id[0, 0]] + 1e-10)
            log_probs.append(lp)

            ids = torch.cat([ids, next_id], dim=1)
            tid = next_id[0, 0].item()
            cmd_tokens.append(tid)

            # Check for = (end of tool use) or EOS
            if tid == eq_id or tid == eos_id:
                break
            # If we've collected +, u/t, digit → command complete
            if len(cmd_tokens) >= 3 and tok_idx >= 2:
                # Might have leading space
                break

        # Decode the command
        cmd_str = ''.join(aid2tok.get(t, '?') for t in cmd_tokens).strip()

        # If model generated '=' or EOS, switch to answer generation
        if eq_id in cmd_tokens or eos_id in cmd_tokens:
            break

        # Try to execute the command on the simulator
        try:
            state_str = sim.step(state_str, cmd_str)
        except (ValueError, KeyError):
            # Invalid command — reward will be 0
            break

        # Inject simulator response into the sequence
        response_str = ' ' + state_str + ' '
        response_ids = [atok2id[c] for c in response_str]
        response_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)
        ids = torch.cat([ids, response_tensor], dim=1)

    # Generate the answer after '='
    # Add space after = if not present
    for _ in range(8):  # max 8 tokens for answer
        if ids.shape[1] >= max_len:
            break
        ids_cond = ids[:, -model.block_size:]
        logits = model(ids_cond)[:, -1, :]
        probs = (logits / max(temperature, 1e-8)).softmax(-1)

        if temperature <= 0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            next_id = torch.multinomial(probs, num_samples=1)

        lp = torch.log(probs[0, next_id[0, 0]] + 1e-10)
        log_probs.append(lp)

        ids = torch.cat([ids, next_id], dim=1)
        if next_id[0, 0].item() == eos_id:
            break

    # Compute reward
    completion = adecode(ids[0].tolist())
    # Extract answer: last number after '='
    matches = re.findall(r'=\s*(\d+)', completion)
    predicted = int(matches[-1]) if matches else -1
    reward = 1.0 if predicted == C else 0.0

    return ids[0], log_probs, reward, completion


def train_rl(model, args, device):
    """REINFORCE training with simulator interaction."""
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # SFT data for mixing (stabilization)
    sft_dataset = ToolUseDataset(split='train', variant='A', max_len=args.max_len,
                                 notation=args.notation)
    sft_loader = DataLoader(sft_dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True)
    sft_iter = iter(sft_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if device.type == 'cuda' else nullcontext())

    train_facts = get_train_facts()
    os.makedirs('checkpoints', exist_ok=True)

    step = args.start_step
    reward_sum = 0.0
    reward_count = 0

    print(f"\nRL training: {len(train_facts)} facts, max_steps={args.max_steps}")

    while step < args.max_steps:
        model.train()

        # Sample a mini-batch of facts
        batch_facts = random.sample(train_facts, min(args.batch_size, len(train_facts)))

        batch_log_probs = []
        batch_rewards = []

        for (A, B) in batch_facts:
            rA = random.choice([True, False])
            rB = random.choice([True, False])
            _, log_probs, reward, _ = generate_with_simulator(
                model, A, B, device, args.max_len,
                roman_A=rA, roman_B=rB,
                temperature=args.temperature,
            )
            if log_probs:
                batch_log_probs.append(torch.stack(log_probs))
                batch_rewards.append(reward)

        if not batch_log_probs:
            step += 1
            continue

        # REINFORCE: loss = -reward * sum(log_probs)
        # Using {0, +1} reward (not {-1, +1}) to avoid zero-gradient traps
        rewards = torch.tensor(batch_rewards, device=device)
        baseline = rewards.mean()

        policy_loss = torch.tensor(0.0, device=device)
        for lps, r in zip(batch_log_probs, batch_rewards):
            advantage = r - baseline.item()
            policy_loss = policy_loss - advantage * lps.sum()
        policy_loss = policy_loss / len(batch_log_probs)

        # SFT mixing for stability
        sft_loss = torch.tensor(0.0, device=device)
        if args.sft_mix_coef > 0:
            try:
                sft_input, sft_target, sft_mask = next(sft_iter)
            except StopIteration:
                sft_iter = iter(sft_loader)
                sft_input, sft_target, sft_mask = next(sft_iter)

            sft_input  = sft_input.to(device)
            sft_target = sft_target.to(device)

            with amp_ctx:
                sft_logits = model(sft_input)
                pad_mask = (sft_target != APAD_ID).float()
                sft_loss_all = F.cross_entropy(
                    sft_logits.reshape(-1, sft_logits.size(-1)),
                    sft_target.reshape(-1),
                    reduction='none',
                ).reshape(sft_target.shape)
                sft_loss = (sft_loss_all * pad_mask).sum() / pad_mask.sum().clamp(min=1)

        total_loss = policy_loss + args.sft_mix_coef * sft_loss

        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        reward_sum += sum(batch_rewards)
        reward_count += len(batch_rewards)
        step += 1

        if step % args.log_every == 0:
            avg_reward = reward_sum / max(reward_count, 1)
            print(f"  step {step:>7d}  reward={avg_reward:.3f}  "
                  f"policy_loss={policy_loss.item():.4f}  sft_loss={sft_loss.item():.4f}")
            reward_sum = 0.0
            reward_count = 0

        if step % args.save_every == 0:
            path = f"checkpoints/tooluse_rl_{args.ckpt_tag}_step{step}.pt"
            torch.save({'model': model.state_dict(), 'step': step,
                        'args': vars(args)}, path)
            print(f"  saved {path}")

    # Final save
    path = f"checkpoints/tooluse_rl_{args.ckpt_tag}_step{step}.pt"
    torch.save({'model': model.state_dict(), 'step': step,
                'args': vars(args)}, path)
    print(f"  saved {path}")
    return step


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Tool-use training with abacus simulator')
    p.add_argument('--mode',         required=True, choices=['sft', 'rl'])
    p.add_argument('--model_size',   choices=['tiny', 'small', 'large', 'medium', 'xlarge'], default='medium')
    p.add_argument('--notation',     choices=['all', 'hindu', 'roman'], default='all',
                   help="Training notation filter: 'all' = all 4 (rA,rB) combos, 'hindu' = Hindu-only, 'roman' = Roman-only")
    p.add_argument('--max_len',      type=int, default=80)
    p.add_argument('--batch_size',   type=int, default=64)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip',    type=float, default=1.0)
    p.add_argument('--dropout',      type=float, default=0.0)
    p.add_argument('--max_steps',    type=int, default=80000)
    p.add_argument('--log_every',    type=int, default=500)
    p.add_argument('--save_every',   type=int, default=10000)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--ckpt_in',      default=None, help='Resume from checkpoint')
    p.add_argument('--ckpt_tag',     default=None,
                   help='Tag used in saved checkpoint filenames; defaults to model_size for backward compat')
    p.add_argument('--mask_loss',    action='store_true',
                   help='SFT: only train on model-generated tokens (not simulator output)')
    p.add_argument('--temperature',  type=float, default=1.0,
                   help='RL: sampling temperature for generation')
    p.add_argument('--sft_mix_coef', type=float, default=1.0,
                   help='RL: coefficient for SFT data mixing loss')
    args = p.parse_args()
    if args.ckpt_tag is None:
        args.ckpt_tag = args.model_size
    return args


def main():
    args = parse_args()
    device = get_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use abacus vocab (includes [, ], |, etc.)
    vocab_sz = ABACUS_VOCAB_SIZE
    model = build_model(args.model_size, args.max_len, vocab_sz,
                        dropout=args.dropout).to(device)

    args.start_step = 0
    if args.ckpt_in:
        ckpt = torch.load(args.ckpt_in, map_location=device)
        load_state_dict_compat(model, ckpt['model'])
        args.start_step = ckpt.get('step', 0)
        print(f"Resumed from {args.ckpt_in} at step {args.start_step}")

    print(f"Model: {args.model_size} ({count_params(model):,} params)")
    print(f"Vocab: {vocab_sz}  |  Device: {device}")
    print(f"Mode:  {args.mode}")

    if args.mode == 'sft':
        train_sft(model, args, device)
    elif args.mode == 'rl':
        train_rl(model, args, device)


if __name__ == '__main__':
    main()
