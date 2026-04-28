"""
SP6: Linear probing of tool-use models for internalized representations.

The motivating finding (SP1/SP2): the model's answer head sometimes emits a
wrong final number despite the simulator having delivered the correct rod
state — and sometimes emits the right answer despite the wrong rod state.
That dissociation pattern is the most concrete handle we have on what tool-use
models actually compute. The cleanest probe of it: forward-pass the FULL trace
(including the simulator's intermediate state responses), extract the hidden
state at the moment just before the model emits its answer, and ask a linear
probe whether that hidden state encodes the simulator's final state.

Probe positions (in the full trace `<A>+<B> : [s0] +cmd1 [s1]^ +cmd2 [s2] = <C>`):
  - pre_eq    : position of the last token just before `=` — usually the `]`
                that closes the simulator's final state. By the time the
                model is here, it has been given the answer in [H|T|U] form.
  - eq        : position of `=` itself. About to emit the answer.

Probe targets (per layer):
  - final_H, final_T, final_U : digits of the simulator's final state
                                 (the `correct` answer in digit form).
                                 Above-chance ⇒ the model has linearly
                                 encoded the rod state.
  - sum_value                   : integer sum A+B (regression). High R² ⇒
                                  hidden state encodes the answer.
  - A_value, B_value            : operands (regression). High R² ⇒ model
                                  has decomposed operands by this position.
  - carry_outer                 : did the simulator's final command overflow?
                                  (binary). Tells us whether the model has
                                  internalized the carry signal.

Usage:
  python3 probe_tooluse.py \
    --ckpt checkpoints/tooluse_sft_sp1_small_all_step80000.pt \
    --model_size small \
    --variant A \
    --max_len 80
"""

import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_test_facts, fmt
from data_abacus import (
    ABACUS_VOCAB_SIZE, ABOS_ID, atok2id, aid2tok, aencode_prompt,
)
from model import build_model, get_device, count_params, load_state_dict_compat
from simulator import AbacusSimulator
from probe import LinearProbe, train_probe  # reuse the linear-probe utilities


_VARIANT_TO_MODE = {'A': 'opaque', 'COMP': 'compositional'}


# ── Trace construction ───────────────────────────────────────────────────────

def build_gold_trace_string(A, B, sim, roman_A, roman_B):
    """
    Build the full gold trace string for (A, B) under the given sim variant.
    Format: '<A> + <B> : [init] +cmd [state](^?) ... = <C>'

    Returns the trace string (without BOS/EOS — those get prepended by aencode).
    """
    traj = sim.gold_trajectory(A, B)
    parts = []
    for step in traj:
        if step['command'] is None:
            parts.append(step['response'])  # initial state
        else:
            parts.append(step['command'])
            parts.append(step['response'])
    trace_body = ' '.join(parts)
    return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {trace_body} = {A + B}"


# ── Probe-position discovery ─────────────────────────────────────────────────

def find_pre_equals_position(token_ids, vocab_map):
    """Position of the token immediately before the LAST `=` in the sequence."""
    eq_id = vocab_map['=']
    positions = [i for i, tid in enumerate(token_ids) if tid == eq_id]
    if not positions:
        return None
    return positions[-1] - 1 if positions[-1] > 0 else None


def find_equals_position(token_ids, vocab_map):
    """Position of the LAST `=` token."""
    eq_id = vocab_map['=']
    positions = [i for i, tid in enumerate(token_ids) if tid == eq_id]
    return positions[-1] if positions else None


# ── Probe targets ────────────────────────────────────────────────────────────

PROBE_TARGETS = {
    # name: (n_classes_or_1, task_type)
    'final_H':    (10, 'classification'),
    'final_T':    (10, 'classification'),
    'final_U':    (10, 'classification'),
    'sum_value':  (1,  'regression'),
    'A_value':    (1,  'regression'),
    'B_value':    (1,  'regression'),
    'carry_outer': (2, 'classification'),
}


def example_targets(A, B):
    """Compute target dict for an (A, B) pair."""
    C = A + B
    final_H = (C // 100) % 10
    final_T = (C // 10) % 10
    final_U = C % 10
    # Was there any carry across the final addition?
    # (Conceptually: did the units- or tens-add overflow?)
    units_carry = (A % 10 + B % 10) >= 10
    tens_carry = ((A // 10 % 10) + (B // 10 % 10) + (1 if units_carry else 0)) >= 10
    carry_outer = int(units_carry or tens_carry)
    return {
        'final_H': final_H, 'final_T': final_T, 'final_U': final_U,
        'sum_value': C, 'A_value': A, 'B_value': B,
        'carry_outer': carry_outer,
    }


# ── Hidden state extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, sim, facts, device, max_len, notation_pairs):
    """Forward-pass each (A, B, notation) example with its full gold trace and collect hidden states.

    Returns list of dicts:
      {targets, notation, hidden_pre_eq: [layer_tensors], hidden_eq: [...]}
    """
    model.eval()
    out = []
    for (A, B) in facts:
        try:
            sim.gold_trajectory(A, B)
        except ValueError:
            continue  # operand out of range for this variant
        targets = example_targets(A, B)
        for (rA, rB) in notation_pairs:
            trace_str = build_gold_trace_string(A, B, sim, rA, rB)
            try:
                ids = aencode_prompt(trace_str, max_len).unsqueeze(0).to(device)
            except KeyError:
                continue
            ids_list = ids[0].tolist()
            pos_pre = find_pre_equals_position(ids_list, atok2id)
            pos_eq  = find_equals_position(ids_list, atok2id)
            if pos_pre is None or pos_eq is None or pos_eq >= ids.shape[1]:
                continue

            _, hidden_states = model(ids, return_hidden_states=True)
            notation = ('roman' if rA else 'hindu', 'roman' if rB else 'hindu')
            out.append({
                'A': A, 'B': B,
                'targets': targets,
                'notation': notation,
                'hidden_pre_eq': [h[0, pos_pre, :].cpu() for h in hidden_states],
                'hidden_eq':     [h[0, pos_eq,  :].cpu() for h in hidden_states],
            })
    return out


# ── Run probes ───────────────────────────────────────────────────────────────

def run_probes_at_position(records, pos_key, train_frac=0.6, seed=42, device='cpu'):
    """Train and evaluate linear probes for each (target × layer)."""
    if not records:
        return {}
    n_layers = len(records[0][pos_key])

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_frac)
    train_idx_set = set(indices[:n_train])
    train_mask = torch.tensor([i in train_idx_set for i in range(len(records))])
    test_mask = ~train_mask

    results = {}
    for target_name, (n_classes, task_type) in PROBE_TARGETS.items():
        results[target_name] = {}
        y_all = torch.tensor([r['targets'][target_name] for r in records])
        for layer in range(n_layers):
            X_all = torch.stack([r[pos_key][layer] for r in records])
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
            if len(X_train) < 20 or len(X_test) < 20:
                continue
            metrics = train_probe(X_train, y_train, X_test, y_test,
                                  n_classes, task_type, device=device)
            results[target_name][layer] = metrics
    return results


# ── Per-notation breakdown ───────────────────────────────────────────────────

def per_notation_results(records, pos_key, layer, train_frac=0.6, seed=42, device='cpu'):
    """For one specific layer, compute probe metrics broken down by notation pair on test set."""
    if not records:
        return {}
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_frac)
    train_idx_set = set(indices[:n_train])

    # Train probes on full mixed train set for each target, then evaluate per-notation.
    out = {}
    notations_all = [r['notation'] for r in records]
    train_mask = torch.tensor([i in train_idx_set for i in range(len(records))])

    for target_name, (n_classes, task_type) in PROBE_TARGETS.items():
        y_all = torch.tensor([r['targets'][target_name] for r in records])
        X_all = torch.stack([r[pos_key][layer] for r in records])

        # Train probe once
        from probe import LinearProbe as LP  # local import for clarity
        hidden_dim = X_all.shape[1]
        probe = LP(hidden_dim, n_classes if task_type == 'classification' else 1).to(device)
        if task_type == 'classification':
            y_t = y_all.long().to(device)
        else:
            y_t = y_all.float().to(device)
        X_t = X_all.float().to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        for _ in range(200):
            logits = probe(X_t[train_mask])
            if task_type == 'classification':
                loss = F.cross_entropy(logits, y_t[train_mask])
            else:
                loss = F.mse_loss(logits.squeeze(-1), y_t[train_mask])
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        with torch.no_grad():
            logits_full = probe(X_t)

        # Per-notation accuracy on test set
        out[target_name] = {}
        for notation in sorted(set(notations_all)):
            mask = torch.tensor([
                (i not in train_idx_set) and (notations_all[i] == notation)
                for i in range(len(records))
            ])
            if mask.sum() < 5:
                continue
            if task_type == 'classification':
                preds = logits_full[mask].argmax(-1)
                acc = (preds == y_t[mask]).float().mean().item()
                out[target_name][notation] = {'accuracy': acc}
            else:
                preds = logits_full[mask].squeeze(-1)
                yt = y_t[mask]
                ss_res = ((yt - preds) ** 2).sum().item()
                ss_tot = ((yt - yt.mean()) ** 2).sum().item()
                r2 = 1 - ss_res / max(ss_tot, 1e-8)
                mae = (preds - yt).abs().mean().item()
                out[target_name][notation] = {'r2': r2, 'mae': mae}
    return out


# ── Pretty printing ──────────────────────────────────────────────────────────

def print_results(results, position_name, n_layers):
    print(f"\n{'='*72}")
    print(f"Probe position: {position_name}")
    print(f"{'='*72}")
    print(f"{'Target':<14} {'Type':<6}", end='')
    for L in range(n_layers):
        print(f"  {'L'+str(L):>7}", end='')
    print()
    print('-' * (22 + 9 * n_layers))
    for target_name, (n_classes, task_type) in PROBE_TARGETS.items():
        if target_name not in results:
            continue
        metric_key = 'accuracy' if task_type == 'classification' else 'r2'
        short_type = 'cls' if task_type == 'classification' else 'reg'
        print(f"{target_name:<14} {short_type:<6}", end='')
        for L in range(n_layers):
            if L in results[target_name]:
                v = results[target_name][L][metric_key]
                print(f"  {v:>7.3f}", end='')
            else:
                print(f"  {'---':>7}", end='')
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Tool-use model probing (SP6)')
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--model_size', choices=['tiny', 'small', 'large', 'medium', 'xlarge'], default='small')
    p.add_argument('--variant',    choices=['A', 'COMP'], default='A')
    p.add_argument('--max_len',    type=int, default=80)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--n_facts',    type=int, default=500,
                   help='Number of (A, B) test facts to sample for probing (each is run with all 4 notation pairs)')
    p.add_argument('--out_json',   default=None,
                   help='Write per-layer probe metrics to this JSON path (default: derived from ckpt)')
    return p.parse_args()


def default_out_json(ckpt_path):
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join('logs', f'{stem}_probe.json')


def main():
    args = parse_args()
    device = get_device()
    random.seed(args.seed)

    model = build_model(args.model_size, args.max_len, ABACUS_VOCAB_SIZE).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    load_state_dict_compat(model, ckpt['model'])
    n_layers = model.n_layer
    print(f"Model:       {args.model_size} ({count_params(model):,} params)")
    print(f"Variant:     {args.variant}")
    print(f"Layers:      {n_layers}")
    print(f"Checkpoint:  {args.ckpt}")

    sim = AbacusSimulator(mode=_VARIANT_TO_MODE[args.variant])

    # Sample N test facts; expand to all 4 notation pairs
    facts_all = get_test_facts()
    rng = random.Random(args.seed)
    rng.shuffle(facts_all)
    facts = facts_all[:args.n_facts]
    notation_pairs = [(rA, rB) for rA in (False, True) for rB in (False, True)]
    print(f"Sampling:    {len(facts)} facts × 4 notations = {len(facts) * 4} traces")

    print("Forward-passing full traces and extracting hidden states...")
    records = extract_hidden_states(model, sim, facts, device, args.max_len, notation_pairs)
    print(f"  {len(records)} valid records")

    pre_results = run_probes_at_position(records, 'hidden_pre_eq', train_frac=0.6,
                                          seed=args.seed, device=device)
    print_results(pre_results, 'hidden_pre_eq', n_layers)

    eq_results = run_probes_at_position(records, 'hidden_eq', train_frac=0.6,
                                         seed=args.seed, device=device)
    print_results(eq_results, 'hidden_eq', n_layers)

    # Per-notation breakdown at the deepest probe-friendly layer for hidden_pre_eq
    print("\n" + "=" * 72)
    print("Per-notation breakdown (hidden_pre_eq, last layer)")
    print("=" * 72)
    last_layer = n_layers - 1
    pn = per_notation_results(records, 'hidden_pre_eq', last_layer,
                              train_frac=0.6, seed=args.seed, device=device)
    for target_name in PROBE_TARGETS:
        if target_name not in pn:
            continue
        n_classes, task_type = PROBE_TARGETS[target_name]
        metric_key = 'accuracy' if task_type == 'classification' else 'r2'
        print(f"\n  {target_name} ({task_type}):")
        for notation in sorted(pn[target_name]):
            v = pn[target_name][notation][metric_key]
            nA, nB = notation
            print(f"    {nA}+{nB}: {v:.3f}")

    if args.out_json is None:
        args.out_json = default_out_json(args.ckpt)
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    summary = {
        'ckpt': args.ckpt,
        'model_size': args.model_size,
        'variant': args.variant,
        'n_records': len(records),
        'n_layers': n_layers,
        'pre_eq': {t: {str(L): m for L, m in pre_results.get(t, {}).items()}
                   for t in PROBE_TARGETS},
        'eq': {t: {str(L): m for L, m in eq_results.get(t, {}).items()}
               for t in PROBE_TARGETS},
        'pre_eq_per_notation_last_layer': {
            t: {f"{n[0]}+{n[1]}": pn[t][n] for n in pn[t]}
            for t in pn
        },
    }
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved probe summary: {args.out_json}")


if __name__ == '__main__':
    main()
