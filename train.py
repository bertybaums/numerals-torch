"""
Unified training script for the Numerals project.

Usage examples:
  # Phase 1: Pretraining
  python train.py --mode pretrain --scaffold none --model_size small --max_steps 64000

  # Phase 2: SFT old scaffold
  python train.py --mode sft --scaffold old --model_size small --max_steps 80000 \
    --ckpt_in checkpoints/pretrain_none_small_step64000.pt

  # Phase 3: RL v1-v5
  python train.py --mode rl --scaffold old --rl_version 1 --max_steps 10000 \
    --ckpt_in checkpoints/sft_old_small_step80000.pt

  # Phase 5: RL with SFT data mixing (the stable fix)
  python train.py --mode rl --scaffold state_seq --rl_version 5 \
    --kl_coef 0.1 --sft_mix_coef 1.0 --max_steps 20000 \
    --ckpt_in checkpoints/sft_state_seq_small_step80000.pt
"""

import argparse
import copy
import os
import random
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    VOCAB_SIZE, PAD_ID, EOS_ID,
    ArithmeticDataset, make_dataset,
    get_train_facts, get_test_facts,
    encode_prompt, decode,
    make_prompt, make_expression,
    extract_answer, extract_step_answer,
)
from data_abacus import (
    ABACUS_VOCAB_SIZE, APAD_ID, AEOS_ID,
    AbacusDataset, make_abacus_dataset,
    aencode_prompt, adecode,
    make_abacus_prompt, make_abacus_expression,
    aextract_answer, is_valid_trace,
)
from model import build_model, get_device, count_params


def is_abacus(scaffold):
    return scaffold.startswith('abacus_')

def abacus_variant(scaffold):
    return scaffold.split('_')[1]  # 'abacus_A' → 'A'


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',       choices=['pretrain', 'sft', 'rl'], required=True)
    p.add_argument('--scaffold',   choices=['none', 'old', 'state_seq', 'decomp',
                                             'carry_explicit', 'digit',
                                             'abacus_A', 'abacus_B', 'abacus_C', 'abacus_D'],
                                   required=True)
    p.add_argument('--model_size', choices=['small', 'large'], default='small')
    p.add_argument('--max_steps',  type=int, default=80000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--max_len',    type=int, default=64)
    p.add_argument('--ckpt_in',    type=str, default=None)
    p.add_argument('--ckpt_out',   type=str, default=None)
    p.add_argument('--rl_version', type=int, choices=[1, 2, 3, 4, 5], default=5)
    p.add_argument('--kl_coef',    type=float, default=0.1)
    p.add_argument('--sft_mix_coef', type=float, default=0.0)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--save_every', type=int, default=10000)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


def ckpt_name(args, step):
    if args.mode == 'rl':
        mode_str = f"rl_v{args.rl_version}"
    else:
        mode_str = args.mode
    return f"checkpoints/{mode_str}_{args.scaffold}_{args.model_size}_step{step}.pt"


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, args):
    os.makedirs('checkpoints', exist_ok=True)
    path = args.ckpt_out if args.ckpt_out else ckpt_name(args, step)
    torch.save({
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step':      step,
        'mode':      args.mode,
        'scaffold':  args.scaffold,
        'model_size': args.model_size,
    }, path)
    return path


def load_checkpoint(path, model, optimizer, args, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    start_step = 0
    # Full resume only if same mode; otherwise weights-only (new phase)
    if ckpt.get('mode') == args.mode:
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed {args.mode} from step {start_step}")
    else:
        print(f"Loaded weights from {ckpt.get('mode', '?')} checkpoint (fresh optimizer)")
    return start_step


# ── Inline evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def inline_eval(model, args, device, n=100):
    model.eval()
    test_facts = get_test_facts()
    sample = random.sample(test_facts, min(n, len(test_facts)))
    correct = 0
    step_correct = 0
    step_total   = 0

    abacus = is_abacus(args.scaffold)
    variant = abacus_variant(args.scaffold) if abacus else None

    for (A, B) in sample:
        C = A + B
        rA = random.choice([False, True])
        rB = random.choice([False, True])

        if abacus:
            prompt_str = make_abacus_prompt(A, B, variant, rA, rB)
            try:
                prompt_ids = aencode_prompt(prompt_str, args.max_len).unsqueeze(0).to(device)
            except KeyError:
                continue
            gen = model.generate(prompt_ids, max_new_tokens=min(64, args.max_len - prompt_ids.shape[1]), greedy=True)
            completion = adecode(gen[0].tolist())
            if aextract_answer(completion) == C:
                correct += 1
        else:
            prompt_str = make_prompt(A, B, args.scaffold, rA, rB)
            try:
                prompt_ids = encode_prompt(prompt_str, args.max_len).unsqueeze(0).to(device)
            except KeyError:
                continue
            gen = model.generate(prompt_ids, max_new_tokens=48, greedy=True)
            completion = decode(gen[0].tolist())
            if extract_answer(completion) == C:
                correct += 1

            # Step accuracy for scaffolds that have an intermediate sum
            if args.scaffold in ('state_seq', 'carry_explicit', 'decomp'):
                B_tens = (B // 10) * 10
                if args.scaffold == 'state_seq':
                    expected_step = A + B_tens if B_tens > 0 else None
                else:
                    expected_step = None

                if expected_step is not None:
                    step_ans = extract_step_answer(completion)
                    step_total += 1
                    if step_ans == expected_step:
                        step_correct += 1

    model.train()
    acc = correct / len(sample)
    step_acc = step_correct / step_total if step_total > 0 else None
    return acc, step_acc


# ── DataLoader helper ─────────────────────────────────────────────────────────

def make_loader(args, split='train'):
    if is_abacus(args.scaffold):
        dataset = AbacusDataset(
            variant=abacus_variant(args.scaffold),
            split=split,
            max_len=args.max_len,
        )
    else:
        dataset = ArithmeticDataset(
            scaffold=args.scaffold,
            split=split,
            max_len=args.max_len,
        )
    num_workers = 0 if get_device().type == 'mps' else 2
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=(get_device().type == 'cuda'),
        drop_last=False,
    )
    return loader, dataset


# ── Pretraining / SFT ─────────────────────────────────────────────────────────

def train_supervised(model, optimizer, scheduler, args, device, start_step):
    loader, dataset = make_loader(args, split='train')
    data_iter = iter(loader)

    print(f"\nDataset: {len(dataset):,} expressions")
    print(f"{'Step':>7}  {'Loss':>8}  {'Acc':>7}  {'StepAcc':>8}")
    print('-' * 40)

    for step in range(start_step, args.max_steps):
        # Cycle through data
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        vocab_sz = ABACUS_VOCAB_SIZE if is_abacus(args.scaffold) else VOCAB_SIZE
        pad_id   = APAD_ID if is_abacus(args.scaffold) else PAD_ID
        loss   = F.cross_entropy(
            logits.reshape(-1, vocab_sz),
            y.reshape(-1),
            ignore_index=pad_id,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.eval_every == 0:
            acc, step_acc = inline_eval(model, args, device)
            step_str = f"{step_acc:.1%}" if step_acc is not None else "   —  "
            print(f"{step+1:>7}  {loss.item():>8.4f}  {acc:>7.2%}  {step_str:>8}")
            save_checkpoint(model, optimizer, step + 1, args)

        if (step + 1) % args.save_every == 0:
            path = save_checkpoint(model, optimizer, step + 1, args)
            print(f"  saved → {path}")


# ── RL training ───────────────────────────────────────────────────────────────

def compute_reward(completion_str, A, B, C, rl_version, scaffold):
    if is_abacus(scaffold):
        variant = abacus_variant(scaffold)
        correct = is_valid_trace(completion_str, A, B, variant)
        if rl_version == 1:
            return 1.0 if correct else -1.0
        return 1.0 if correct else 0.0

    answer = extract_answer(completion_str)
    correct = (answer == C)

    if rl_version in (1, 2, 3):
        if rl_version == 1:
            return 1.0 if correct else -1.0   # v1 uses {-1, +1}
        return 1.0 if correct else 0.0

    # v4 and v5: process reward (0.5 for operands, 0.5 for answer)
    operands_ok = _check_operands(completion_str, A, B, scaffold)
    return (0.5 if operands_ok else 0.0) + (0.5 if correct else 0.0)


def _check_operands(completion_str, A, B, scaffold):
    """Check whether the chain-of-thought correctly identifies both operands."""
    if scaffold == 'old':
        # Expect ': A + B =' in the completion
        pattern = rf':\s*{A}\s*\+\s*{B}\s*='
        return bool(re.search(pattern, completion_str))
    elif scaffold == 'state_seq':
        # Expect ': A + ...' — just check A appears after the colon
        colon_pos = completion_str.find(':')
        if colon_pos == -1:
            return False
        after = completion_str[colon_pos:]
        return str(A) in after and str(B % 10 if B % 10 else B) in after
    return False


def compute_rl_loss(model, ref_model, gen_ids, prompt_len,
                    reward, ema_baseline, sft_strings, args, device,
                    pad_id=None, vocab_sz=None):
    """
    Compute the RL loss for the current completion.
    gen_ids: (1, T_total) — prompt + completion tokens
    """
    pad_id   = pad_id   if pad_id   is not None else PAD_ID
    vocab_sz = vocab_sz if vocab_sz is not None else VOCAB_SIZE

    comp_ids = gen_ids[:, prompt_len:]   # (1, T_comp)
    T_comp   = comp_ids.shape[1]
    if T_comp == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Recompute log-probs of completion under current model (with gradients)
    logits       = model(gen_ids[:, :-1])                          # (1, T-1, V)
    log_probs_all = F.log_softmax(logits, dim=-1)                  # (1, T-1, V)
    # Completion positions start at index (prompt_len - 1) in the shifted sequence
    comp_log_probs = log_probs_all[:, prompt_len - 1:, :]          # (1, T_comp, V)
    comp_log_probs = comp_log_probs.gather(
        2, comp_ids.unsqueeze(-1)).squeeze(-1)                     # (1, T_comp)

    mask     = (comp_ids != pad_id).float()
    mean_lp  = (comp_log_probs * mask).sum() / mask.sum().clamp(min=1)

    v = args.rl_version

    if v == 1:
        # {-1,+1} rewards, plain EMA baseline
        baseline  = ema_baseline
        advantage = reward - baseline
        return -(advantage * mean_lp)

    elif v == 2:
        # {0,+1}, entropy bonus (too strong)
        advantage = reward - ema_baseline
        entropy   = -(log_probs_all.exp() * log_probs_all).sum(-1).mean()
        return -(advantage * mean_lp) - 0.01 * entropy

    elif v == 3:
        # {0,+1}, no entropy, EMA (collapses to 0)
        advantage = reward - ema_baseline
        return -(advantage * mean_lp)

    elif v == 4:
        # Baseline floor
        baseline  = max(0.05, ema_baseline)
        advantage = reward - baseline
        return -(advantage * mean_lp)

    elif v == 5:
        # KL anchoring + optional SFT data mixing
        baseline  = max(0.05, ema_baseline)
        advantage = reward - baseline
        policy_loss = -(advantage * mean_lp)

        kl_loss = torch.tensor(0.0, device=device)
        if ref_model is not None and args.kl_coef > 0:
            with torch.no_grad():
                ref_logits    = ref_model(gen_ids[:, :-1])
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            # One-sample KL estimate at completion positions
            pi_lp  = log_probs_all[:, prompt_len - 1:, :]
            ref_lp = ref_log_probs[:, prompt_len - 1:, :]
            kl_per = (pi_lp.exp() * (pi_lp - ref_lp)).sum(-1)   # (1, T_comp)
            kl_loss = (kl_per * mask).sum() / mask.sum().clamp(min=1)

        sft_loss = torch.tensor(0.0, device=device)
        if args.sft_mix_coef > 0.0 and sft_strings:
            sft_str = random.choice(sft_strings)
            if is_abacus(args.scaffold):
                from data_abacus import acollate_lm
                x_sft, y_sft = acollate_lm([sft_str], args.max_len)
            else:
                from data import collate_lm
                x_sft, y_sft = collate_lm([sft_str], args.max_len)
            x_sft = x_sft.to(device)
            y_sft = y_sft.to(device)
            logits_sft = model(x_sft)
            sft_loss = F.cross_entropy(
                logits_sft.reshape(-1, vocab_sz),
                y_sft.reshape(-1),
                ignore_index=pad_id,
            )

        return policy_loss + args.kl_coef * kl_loss + args.sft_mix_coef * sft_loss

    raise ValueError(f"Unknown rl_version: {v}")


def train_rl(model, optimizer, scheduler, args, device, start_step):
    # Frozen reference model for v5
    ref_model = None
    if args.rl_version == 5:
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    train_facts = get_train_facts()
    abacus = is_abacus(args.scaffold)
    variant = abacus_variant(args.scaffold) if abacus else None
    pad_id   = APAD_ID   if abacus else PAD_ID
    vocab_sz = ABACUS_VOCAB_SIZE if abacus else VOCAB_SIZE
    if abacus:
        sft_strings = make_abacus_dataset(variant, split='train') if args.sft_mix_coef > 0 else []
    else:
        sft_strings = make_dataset(args.scaffold, split='train') if args.sft_mix_coef > 0 else []

    ema_alpha    = 0.05
    ema_baseline = 0.0

    print(f"\nRL version: {args.rl_version}  |  kl_coef: {args.kl_coef}"
          f"  |  sft_mix_coef: {args.sft_mix_coef}")
    print(f"{'Step':>7}  {'Reward EMA':>10}  {'Acc':>7}  {'KL':>6}")
    print('-' * 40)

    for step in range(start_step, args.max_steps):
        A, B = random.choice(train_facts)
        C    = A + B
        rA   = random.choice([False, True])
        rB   = random.choice([False, True])

        if abacus:
            prompt_str = make_abacus_prompt(A, B, variant, rA, rB)
            try:
                prompt_ids = aencode_prompt(prompt_str, args.max_len).unsqueeze(0).to(device)
            except KeyError:
                continue
        else:
            prompt_str = make_prompt(A, B, args.scaffold, rA, rB)
            try:
                prompt_ids = encode_prompt(prompt_str, args.max_len).unsqueeze(0).to(device)
            except KeyError:
                continue
        prompt_len = prompt_ids.shape[1]

        # Generate completion (stochastic)
        model.eval()
        with torch.no_grad():
            # Cap new tokens so total sequence never exceeds block_size
            max_new = max(1, args.max_len - prompt_len)
            gen_ids = model.generate(
                prompt_ids, max_new_tokens=max_new,
                temperature=args.temperature, greedy=False,
            )
        model.train()

        completion = adecode(gen_ids[0].tolist()) if abacus else decode(gen_ids[0].tolist())
        reward     = compute_reward(completion, A, B, C, args.rl_version, args.scaffold)
        ema_baseline = (1 - ema_alpha) * ema_baseline + ema_alpha * reward

        loss = compute_rl_loss(
            model, ref_model, gen_ids, prompt_len,
            reward, ema_baseline, sft_strings, args, device,
            pad_id=pad_id, vocab_sz=vocab_sz,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % args.eval_every == 0:
            acc, _ = inline_eval(model, args, device, n=100)
            # KL diagnostic
            kl_diag = 0.0
            if ref_model is not None:
                kl_diag = _kl_diagnostic(model, ref_model, train_facts[:10], args, device)
            print(f"{step+1:>7}  {ema_baseline:>10.4f}  {acc:>7.2%}  {kl_diag:>6.3f}")
            save_checkpoint(model, optimizer, step + 1, args)

        if (step + 1) % args.save_every == 0:
            path = save_checkpoint(model, optimizer, step + 1, args)
            print(f"  saved → {path}")


@torch.no_grad()
def _kl_diagnostic(model, ref_model, facts, args, device):
    """Estimate mean KL divergence over a few examples for monitoring."""
    total_kl = 0.0
    n = 0
    for (A, B) in facts:
        prompt_str = make_prompt(A, B, args.scaffold)
        try:
            prompt_ids = encode_prompt(prompt_str, args.max_len).unsqueeze(0).to(device)
        except KeyError:
            continue
        gen = model.generate(prompt_ids, max_new_tokens=32, greedy=True)
        if gen.shape[1] <= 1:
            continue
        logits      = model(gen[:, :-1])
        ref_logits  = ref_model(gen[:, :-1])
        lp          = F.log_softmax(logits, dim=-1)
        ref_lp      = F.log_softmax(ref_logits, dim=-1)
        kl          = (lp.exp() * (lp - ref_lp)).sum(-1).mean().item()
        total_kl   += kl
        n          += 1
    return total_kl / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    vocab_sz = ABACUS_VOCAB_SIZE if is_abacus(args.scaffold) else VOCAB_SIZE
    model = build_model(args.model_size, args.max_len, vocab_sz).to(device)
    print(f"\nMode: {args.mode}  |  Scaffold: {args.scaffold}"
          f"  |  Model: {args.model_size} ({count_params(model):,} params)"
          f"  |  Vocab: {vocab_sz}  |  Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=args.max_steps,
    )

    start_step = 0
    if args.ckpt_in:
        start_step = load_checkpoint(args.ckpt_in, model, optimizer, args, device)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    if args.mode in ('pretrain', 'sft'):
        train_supervised(model, optimizer, scheduler, args, device, start_step)
    elif args.mode == 'rl':
        train_rl(model, optimizer, scheduler, args, device, start_step)

    # Final save
    path = save_checkpoint(model, optimizer, args.max_steps, args)
    print(f"\nFinal checkpoint saved → {path}")


if __name__ == '__main__':
    main()
