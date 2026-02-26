"""
Formal evaluation on the held-out test set.

Usage:
  python evaluate.py \
    --ckpt checkpoints/sft_carry_explicit_small_step160000.pt \
    --scaffold carry_explicit \
    --model_size small \
    --max_len 64

For Phase 9 (digit scaffold), add --ood to also test 4-digit OOD generalization.
"""

import argparse
import random
from collections import defaultdict

import torch

from data import (
    VOCAB_SIZE, get_test_facts, encode_prompt, decode,
    make_prompt, make_expression, extract_answer, extract_step_answer,
    from_roman,
)
from model import build_model, get_device, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--scaffold',   required=True,
                   choices=['none', 'old', 'state_seq', 'decomp', 'carry_explicit', 'digit'])
    p.add_argument('--model_size', choices=['small', 'large'], default='small')
    p.add_argument('--max_len',    type=int, default=64)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--ood',        action='store_true',
                   help='Also run OOD 4-digit tests (digit scaffold only)')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, scaffold, test_facts, device, max_len):
    model.eval()

    counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    step_correct = 0
    step_total   = 0
    carry_correct = 0
    carry_total   = 0

    for (A, B) in test_facts:
        C = A + B

        if scaffold in ('digit',):
            # digit scaffold: Hindu-Arabic only
            notation_pairs = [(False, False)]
        elif scaffold == 'none':
            notation_pairs = [(rA, rB) for rA in (False, True) for rB in (False, True)
                              for _ in [None]]  # 4 combos (result varies too but we just check answer)
            # actually for 'none', result notation matters; use all 8
            notation_pairs = [(rA, rB) for rA in (False, True) for rB in (False, True)]
        else:
            notation_pairs = [(rA, rB) for rA in (False, True) for rB in (False, True)]

        for (rA, rB) in notation_pairs:
            key = ('roman' if rA else 'hindu', 'roman' if rB else 'hindu')
            prompt_str = make_prompt(A, B, scaffold, rA, rB)
            try:
                prompt_ids = encode_prompt(prompt_str, max_len).unsqueeze(0).to(device)
            except KeyError:
                counts[key]['total'] += 1
                continue

            gen        = model.generate(prompt_ids, max_new_tokens=56, greedy=True)
            completion = decode(gen[0].tolist())
            answer     = extract_answer(completion)

            counts[key]['total']   += 1
            counts[key]['correct'] += int(answer == C)

            # ── Step accuracy diagnostics ──────────────────────────────────
            if scaffold == 'state_seq':
                B_tens = (B // 10) * 10
                if B_tens > 0:
                    expected_step = A + B_tens
                    step_ans = extract_step_answer(completion)
                    step_total   += 1
                    step_correct += int(step_ans == expected_step)

            elif scaffold == 'carry_explicit':
                # Units step accuracy
                A_units, B_units = A % 10, B % 10
                expected_units   = A_units + B_units
                step_ans = extract_step_answer(completion)
                step_total   += 1
                step_correct += int(step_ans == expected_units)

                # Carry propagation: among cases where units step is correct,
                # check whether the final answer's tens digit is right
                if step_ans == expected_units and expected_units >= 10:
                    carry_total += 1
                    expected_C_tens = A // 10 + B // 10 + 1
                    if answer == C:
                        carry_correct += 1

            elif scaffold == 'decomp':
                # Tens step accuracy (a_t×10+b_t×10=tens_sum)
                A_tens = (A // 10) * 10
                B_tens = (B // 10) * 10
                if A >= 10 and B >= 10:
                    expected_step = A_tens + B_tens
                    step_ans = extract_step_answer(completion)
                    step_total   += 1
                    step_correct += int(step_ans == expected_step)

    return counts, step_correct, step_total, carry_correct, carry_total


@torch.no_grad()
def evaluate_ood(model, device, max_len):
    """
    OOD evaluation for digit scaffold:
    1. Blind: 4-digit operands, model generates preamble + carry
    2. Oracle: correct preamble given; only carry + final answer evaluated
    """
    random.seed(0)
    ood_facts = [(random.randint(1000, 4999), random.randint(1000, 4999))
                 for _ in range(200)]

    from data import _carry_steps_str, collate_lm
    import re

    blind_correct = 0
    oracle_correct = 0
    n = len(ood_facts)

    for (A, B) in ood_facts:
        C = A + B

        # ── Blind ─────────────────────────────────────────────────────────
        prompt_str = f"{A} + {B} : "
        try:
            prompt_ids = encode_prompt(prompt_str, max_len).unsqueeze(0).to(device)
        except KeyError:
            continue
        gen        = model.generate(prompt_ids, max_new_tokens=64, greedy=True)
        completion = decode(gen[0].tolist())
        if extract_answer(completion) == C:
            blind_correct += 1

        # ── Oracle preamble ───────────────────────────────────────────────
        preamble_A = ' '.join(reversed(str(A)))
        preamble_B = ' '.join(reversed(str(B)))
        oracle_prompt = f"{A} + {B} : {preamble_A} + {preamble_B} : "
        try:
            prompt_ids = encode_prompt(oracle_prompt, max_len).unsqueeze(0).to(device)
        except KeyError:
            continue
        gen        = model.generate(prompt_ids, max_new_tokens=64, greedy=True)
        completion = decode(gen[0].tolist())
        if extract_answer(completion) == C:
            oracle_correct += 1

    return blind_correct, oracle_correct, n


def print_results(counts, step_correct, step_total, carry_correct, carry_total, scaffold):
    pairs = [('hindu', 'hindu'), ('hindu', 'roman'), ('roman', 'hindu'), ('roman', 'roman')]

    total_correct = sum(v['correct'] for v in counts.values())
    total_total   = sum(v['total']   for v in counts.values())

    print(f"\n{'Notation pair':<22}  {'Correct':>8}  {'Total':>7}  {'Accuracy':>9}")
    print('-' * 54)

    for (kA, kB) in pairs:
        key = (kA, kB)
        if key not in counts:
            continue
        v = counts[key]
        acc = v['correct'] / v['total'] if v['total'] else 0
        print(f"{kA + ' + ' + kB:<22}  {v['correct']:>8}  {v['total']:>7}  {acc:>9.1%}")

    print('-' * 54)
    overall = total_correct / total_total if total_total else 0
    print(f"{'Overall':<22}  {total_correct:>8}  {total_total:>7}  {overall:>9.1%}")

    if step_total > 0:
        label = {
            'state_seq':      'Step accuracy   ',
            'carry_explicit': 'Units step acc  ',
            'decomp':         'Tens step acc   ',
        }.get(scaffold, 'Step accuracy   ')
        print(f"\n{label}  {step_correct:>8}  {step_total:>7}  "
              f"{step_correct / step_total:>9.1%}")

    if carry_total > 0:
        print(f"{'Carry propagation':<22}  {carry_correct:>8}  {carry_total:>7}  "
              f"{carry_correct / carry_total:>9.1%}")


def main():
    args   = parse_args()
    device = get_device()
    random.seed(args.seed)

    model = build_model(args.model_size, args.max_len, VOCAB_SIZE).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"\nEvaluating: {args.ckpt}")
    print(f"Scaffold:   {args.scaffold}  |  Model: {args.model_size}"
          f" ({count_params(model):,} params)  |  Device: {device}")

    if args.scaffold == 'digit':
        from data import make_dataset
        # Use the same test split as training
        test_strings = make_dataset('digit', split='test')
        # Reconstruct facts from strings
        import re
        test_facts = []
        for s in test_strings:
            m = re.match(r'^(\d+) \+ (\d+)', s)
            if m:
                test_facts.append((int(m.group(1)), int(m.group(2))))
    else:
        test_facts = get_test_facts()

    counts, step_c, step_t, carry_c, carry_t = evaluate(
        model, args.scaffold, test_facts, device, args.max_len
    )
    print_results(counts, step_c, step_t, carry_c, carry_t, args.scaffold)

    if args.ood and args.scaffold == 'digit':
        print('\n── OOD: 4-digit operands (1000–4999) ──────────────────────')
        blind_c, oracle_c, n = evaluate_ood(model, device, args.max_len)
        print(f"Blind  (model generates preamble): {blind_c}/{n} = {blind_c/n:.1%}")
        print(f"Oracle (correct preamble given):   {oracle_c}/{n} = {oracle_c/n:.1%}")


if __name__ == '__main__':
    main()
