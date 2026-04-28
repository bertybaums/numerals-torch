"""
Evaluation for tool-use models with interactive simulator.

Unlike standard evaluation (where the model generates from a static prompt),
tool-use evaluation interleaves model generation with simulator responses.

Reports:
  - Overall accuracy (final answer correct)
  - Command validity rate (fraction of commands the simulator accepts)
  - Average number of commands per problem
  - Per-notation-pair breakdown
  - Trajectory analysis: does the model follow the optimal path?

Usage:
  python evaluate_tooluse.py \
    --ckpt checkpoints/tooluse_sft_medium_step80000.pt \
    --model_size medium \
    --max_len 80
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

import torch

from data import (
    VOCAB_SIZE, get_test_facts, fmt,
)
from data_abacus import (
    ABACUS_VOCAB_SIZE,
    atok2id, aid2tok,
    ABOS_ID, AEOS_ID, APAD_ID,
    aencode_prompt, adecode,
)
from model import build_model, get_device, count_params, load_state_dict_compat
from simulator import AbacusSimulator

_VARIANT_TO_MODE = {'A': 'opaque', 'COMP': 'compositional'}


@torch.no_grad()
def evaluate_interactive(model, test_facts, device, max_len, variant='A'):
    """
    Evaluate model with interactive simulator on test set.

    variant 'A' = opaque commands (+u5, +t3); 'COMP' = compositional (+05, +13).

    Returns detailed results per example.
    """
    model.eval()
    sim = AbacusSimulator(mode=_VARIANT_TO_MODE[variant])
    results = []

    notation_pairs = [(rA, rB) for rA in (False, True) for rB in (False, True)]

    for (A, B) in test_facts:
        if B > 99:
            continue  # can't decompose into single-digit +u/+t commands
        C = A + B
        gold_traj = sim.gold_trajectory(A, B)
        gold_cmds = [s['command'] for s in gold_traj if s['command'] is not None]

        for (rA, rB) in notation_pairs:
            notation = ('roman' if rA else 'hindu', 'roman' if rB else 'hindu')

            # Build prompt with initial state
            init_state = sim.reset(A)
            prompt_str = f"{fmt(A, rA)} + {fmt(B, rB)} : {init_state} "
            prompt_ids = aencode_prompt(prompt_str, max_len).unsqueeze(0).to(device)
            ids = prompt_ids.clone()

            state_str = init_state
            commands = []
            valid_commands = 0
            total_commands = 0
            eq_id = atok2id['=']
            eos_id = AEOS_ID

            # Interactive generation
            for cmd_num in range(10):  # safety limit
                # Generate command tokens
                cmd_tokens = []
                for _ in range(4):
                    if ids.shape[1] >= max_len:
                        break
                    ids_cond = ids[:, -model.block_size:]
                    logits = model(ids_cond)[:, -1, :]
                    next_id = logits.argmax(-1, keepdim=True)
                    ids = torch.cat([ids, next_id], dim=1)
                    tid = next_id[0, 0].item()
                    cmd_tokens.append(tid)

                    if tid == eq_id or tid == eos_id:
                        break
                    if len(cmd_tokens) >= 3:
                        break

                cmd_str = ''.join(aid2tok.get(t, '?') for t in cmd_tokens).strip()

                if eq_id in cmd_tokens or eos_id in cmd_tokens:
                    break

                total_commands += 1
                commands.append(cmd_str)

                try:
                    state_str = sim.step(state_str, cmd_str)
                    valid_commands += 1
                except (ValueError, KeyError):
                    # Invalid command — inject error state and stop
                    break

                # Inject simulator response
                response_str = ' ' + state_str + ' '
                response_ids = [atok2id[c] for c in response_str]
                response_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)
                ids = torch.cat([ids, response_tensor], dim=1)

            # Generate answer
            for _ in range(8):
                if ids.shape[1] >= max_len:
                    break
                ids_cond = ids[:, -model.block_size:]
                logits = model(ids_cond)[:, -1, :]
                next_id = logits.argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id[0, 0].item() == eos_id:
                    break

            # Extract answer
            completion = adecode(ids[0].tolist())
            matches = re.findall(r'=\s*(\d+)', completion)
            predicted = int(matches[-1]) if matches else -1
            correct = (predicted == C)

            # Check if trajectory matches gold
            follows_gold = (commands == gold_cmds)

            results.append({
                'A': A, 'B': B, 'C': C,
                'notation': notation,
                'correct': correct,
                'predicted': predicted,
                'commands': commands,
                'gold_commands': gold_cmds,
                'valid_commands': valid_commands,
                'total_commands': total_commands,
                'follows_gold': follows_gold,
                'final_state': state_str,
                'completion': completion,
            })

    return results


def print_results(results):
    """Print evaluation summary."""
    total = len(results)
    correct = sum(r['correct'] for r in results)
    print(f"\n{'='*60}")
    print(f"Tool-Use Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall accuracy: {correct}/{total} = {correct/total:.1%}")

    # Command stats
    total_cmds = sum(r['total_commands'] for r in results)
    valid_cmds = sum(r['valid_commands'] for r in results)
    gold_match = sum(r['follows_gold'] for r in results)
    avg_cmds = total_cmds / max(total, 1)

    print(f"\nCommand validity:  {valid_cmds}/{total_cmds} = {valid_cmds/max(total_cmds,1):.1%}")
    print(f"Gold trajectory:   {gold_match}/{total} = {gold_match/max(total,1):.1%}")
    print(f"Avg commands/prob: {avg_cmds:.2f}")

    # Per-notation breakdown
    notation_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        key = r['notation']
        notation_counts[key]['total'] += 1
        notation_counts[key]['correct'] += int(r['correct'])

    pairs = [('hindu', 'hindu'), ('hindu', 'roman'), ('roman', 'hindu'), ('roman', 'roman')]
    print(f"\n{'Notation pair':<22}  {'Correct':>8}  {'Total':>7}  {'Accuracy':>9}")
    print('-' * 54)
    for (kA, kB) in pairs:
        key = (kA, kB)
        if key not in notation_counts:
            continue
        v = notation_counts[key]
        acc = v['correct'] / v['total'] if v['total'] else 0
        print(f"{kA + ' + ' + kB:<22}  {v['correct']:>8}  {v['total']:>7}  {acc:>9.1%}")

    # Novel trajectories (correct but not gold)
    novel_correct = [r for r in results if r['correct'] and not r['follows_gold']]
    if novel_correct:
        print(f"\nNovel correct trajectories: {len(novel_correct)}")
        # Show up to 5 examples
        for r in novel_correct[:5]:
            cmds = ' '.join(r['commands'])
            gold = ' '.join(r['gold_commands'])
            print(f"  {r['A']}+{r['B']}={r['C']}: model=[{cmds}] gold=[{gold}]")

    # Error analysis
    wrong = [r for r in results if not r['correct']]
    if wrong:
        # Categorize errors
        invalid_cmd = sum(1 for r in wrong if r['valid_commands'] < r['total_commands'])
        wrong_answer = sum(1 for r in wrong if r['valid_commands'] == r['total_commands'])
        print(f"\nError breakdown ({len(wrong)} errors):")
        print(f"  Invalid command: {invalid_cmd}")
        print(f"  Wrong answer:    {wrong_answer}")

        # Show sample errors
        print(f"\nSample errors:")
        for r in wrong[:5]:
            print(f"  {r['A']}+{r['B']}={r['C']}: "
                  f"predicted={r['predicted']} cmds={r['commands']}")


def classify(r):
    """Coarse status / error category for examples-table grouping.

    Categories are intentionally broad for SP1; the failure typology is expected
    to crystallise once we have data from all SPs. Pre-SP4 codification per plan.
    """
    if r['correct']:
        return 'correct'
    if r['valid_commands'] < r['total_commands']:
        return 'invalid_command'
    if r['predicted'] == -1:
        return 'no_answer'
    # All commands valid, model produced an integer answer, but it's wrong
    if r['follows_gold']:
        return 'wrong_answer_on_gold_path'  # model executed gold cmds but emitted wrong number
    if r['total_commands'] != len(r['gold_commands']):
        return 'wrong_command_count'  # under- or over-shot the gold trajectory length
    return 'wrong_digits'  # right shape, wrong digit picks


def write_jsonl(results, path):
    """Dump per-example results as JSONL, with classify() applied."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for r in results:
            row = {
                'A': r['A'],
                'B': r['B'],
                'C': r['C'],
                'notation_A': r['notation'][0],
                'notation_B': r['notation'][1],
                'correct': r['correct'],
                'status': classify(r),
                'predicted': r['predicted'],
                'commands': r['commands'],
                'gold_commands': r['gold_commands'],
                'valid_commands': r['valid_commands'],
                'total_commands': r['total_commands'],
                'follows_gold': r['follows_gold'],
                'final_state': r['final_state'],
                'completion': r['completion'],
            }
            f.write(json.dumps(row) + '\n')
    print(f"Wrote per-example JSONL: {path}")


def default_jsonl_path(ckpt_path):
    """Derive logs/<ckpt-stem>_eval.jsonl from a checkpoint path."""
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join('logs', f'{stem}_eval.jsonl')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--model_size', choices=['tiny', 'small', 'large', 'medium', 'xlarge'], default='medium')
    p.add_argument('--variant',    choices=['A', 'COMP'], default='A',
                   help="Tool-use command variant — must match training: 'A' (opaque) or 'COMP' (compositional)")
    p.add_argument('--max_len',    type=int, default=80)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--jsonl_out',  default=None,
                   help='Path for per-example JSONL dump. Defaults to logs/<ckpt-stem>_eval.jsonl')
    p.add_argument('--no_jsonl',   action='store_true',
                   help='Skip JSONL output entirely')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    random.seed(args.seed)

    vocab_sz = ABACUS_VOCAB_SIZE
    model = build_model(args.model_size, args.max_len, vocab_sz).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    load_state_dict_compat(model, ckpt['model'])

    print(f"Evaluating: {args.ckpt}")
    print(f"Model: {args.model_size} ({count_params(model):,} params)")
    print(f"Device: {device}")

    test_facts = get_test_facts()
    results = evaluate_interactive(model, test_facts, device, args.max_len,
                                   variant=args.variant)
    print_results(results)

    if not args.no_jsonl:
        jsonl_path = args.jsonl_out or default_jsonl_path(args.ckpt)
        write_jsonl(results, jsonl_path)


if __name__ == '__main__':
    main()
