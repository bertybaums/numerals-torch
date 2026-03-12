"""
Linear probing for internal representations in trained models.

Extracts hidden states from a trained checkpoint and trains simple linear
probes to test what information is linearly decodable at each layer.

Probe targets:
  - A_value, B_value: full integer values of operands (regression)
  - A_tens, A_units, B_tens, B_units: individual digits (10-class classification)
  - sum_value: A + B (regression)
  - carry: whether units column carries (binary classification)

Probe positions (token index within the encoded prompt):
  - colon:    the ':' token (scaffold transition point)
  - equals:   last token before '=' (model must have computed answer)
  - plus:     the '+' token (both operands partially visible)

Results are stratified by notation pair and reported per-layer.

Usage:
  python probe.py \
    --ckpt checkpoints/sft_carry_explicit_small_step160000.pt \
    --scaffold carry_explicit \
    --model_size small \
    --max_len 64

  # Abacus variant:
  python probe.py \
    --ckpt checkpoints/sft_abacus_A_small_step80000.pt \
    --scaffold abacus_A \
    --model_size small \
    --max_len 80
"""

import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import (
    VOCAB_SIZE, BOS_ID, tok2id,
    get_test_facts, get_train_facts,
    make_prompt, make_expression, encode_prompt,
    fmt, from_roman,
)
from data_abacus import (
    ABACUS_VOCAB_SIZE, ABOS_ID, atok2id,
    make_abacus_prompt, make_abacus_expression, aencode_prompt,
)
from model import build_model, get_device, count_params, load_state_dict_compat


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_abacus(scaffold):
    return scaffold.startswith('abacus_')

def abacus_variant(scaffold):
    return scaffold.split('_')[1]


def find_token_position(token_ids, target_char, vocab_map, which='first'):
    """
    Find position of target_char in token_ids.
    which: 'first' or 'last'.
    """
    target_id = vocab_map[target_char]
    positions = [i for i, tid in enumerate(token_ids) if tid == target_id]
    if not positions:
        return None
    return positions[0] if which == 'first' else positions[-1]


def find_equals_predecessor(token_ids, vocab_map):
    """Find the token position just before the last '=' token."""
    eq_id = vocab_map['=']
    positions = [i for i, tid in enumerate(token_ids) if tid == eq_id]
    if not positions:
        return None
    return positions[-1] - 1 if positions[-1] > 0 else None


# ── Dataset construction ─────────────────────────────────────────────────────

def build_probe_dataset(scaffold, split='test', seed=42):
    """
    Build a list of probe examples, each with:
      - A, B, C: integer operands and sum
      - roman_A, roman_B: notation flags
      - prompt_str: the prompt string
      - targets: dict of probe target values
    """
    facts = get_test_facts() if split == 'test' else get_train_facts()
    examples = []

    if scaffold == 'digit':
        notation_pairs = [(False, False)]
    else:
        notation_pairs = [(rA, rB) for rA in (False, True) for rB in (False, True)]

    for (A, B) in facts:
        C = A + B
        A_tens, A_units = A // 10, A % 10
        B_tens, B_units = B // 10, B % 10
        carry = int((A_units + B_units) >= 10)

        for (rA, rB) in notation_pairs:
            if is_abacus(scaffold):
                variant = abacus_variant(scaffold)
                prompt_str = make_abacus_prompt(A, B, variant, rA, rB)
            else:
                prompt_str = make_prompt(A, B, scaffold, rA, rB)

            examples.append({
                'A': A, 'B': B, 'C': C,
                'roman_A': rA, 'roman_B': rB,
                'prompt_str': prompt_str,
                'targets': {
                    'A_value':  A,
                    'B_value':  B,
                    'A_tens':   A_tens,
                    'A_units':  A_units,
                    'B_tens':   B_tens,
                    'B_units':  B_units,
                    'sum_value': C,
                    'carry':    carry,
                },
            })

    return examples


# ── Hidden state extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, examples, device, max_len, scaffold):
    """
    Run each example through the model and collect hidden states at probe positions.

    Returns:
        List of dicts, each with:
          - 'targets': target dict
          - 'notation': (str, str) notation pair
          - 'hidden_colon':  list of (n_embd,) tensors, one per layer (or None)
          - 'hidden_equals': list of (n_embd,) tensors, one per layer (or None)
          - 'hidden_plus':   list of (n_embd,) tensors, one per layer (or None)
    """
    model.eval()
    use_abacus = is_abacus(scaffold)
    vocab_map = atok2id if use_abacus else tok2id
    encode_fn = aencode_prompt if use_abacus else encode_prompt

    results = []
    for ex in examples:
        prompt_str = ex['prompt_str']
        try:
            prompt_ids = encode_fn(prompt_str, max_len).unsqueeze(0).to(device)
        except KeyError:
            continue

        logits, hidden_states = model(prompt_ids, return_hidden_states=True)
        # hidden_states: list of (1, T, n_embd) tensors

        ids_list = prompt_ids[0].tolist()

        # Find probe positions
        pos_colon  = find_token_position(ids_list, ':', vocab_map, 'first')
        pos_equals = find_equals_predecessor(ids_list, vocab_map)
        pos_plus   = find_token_position(ids_list, '+', vocab_map, 'first')

        notation = ('roman' if ex['roman_A'] else 'hindu',
                     'roman' if ex['roman_B'] else 'hindu')

        result = {
            'targets': ex['targets'],
            'notation': notation,
        }

        for name, pos in [('hidden_colon', pos_colon),
                          ('hidden_equals', pos_equals),
                          ('hidden_plus', pos_plus)]:
            if pos is not None and pos < prompt_ids.shape[1]:
                result[name] = [hs[0, pos, :].cpu() for hs in hidden_states]
            else:
                result[name] = None

        results.append(result)

    return results


# ── Linear probe ─────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Simple linear probe: hidden_dim -> n_classes (or 1 for regression)."""
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


PROBE_CONFIGS = {
    # name: (n_classes, task_type)
    # n_classes=1 for regression, >1 for classification
    'A_value':   (101, 'regression'),   # 1-100
    'B_value':   (101, 'regression'),   # 1-100
    'A_tens':    (11,  'classification'),  # 0-10 (10 when A=100)
    'A_units':   (10,  'classification'),  # 0-9
    'B_tens':    (11,  'classification'),  # 0-10 (10 when B=100)
    'B_units':   (10,  'classification'),  # 0-9
    'sum_value': (201, 'regression'),   # 2-200
    'carry':     (2,   'classification'),  # 0 or 1
}


def train_probe(X_train, y_train, X_test, y_test, n_classes, task_type,
                lr=1e-2, epochs=200, device='cpu'):
    """
    Train a linear probe and return test metrics.

    Args:
        X_train, X_test: (N, hidden_dim) float tensors
        y_train, y_test: (N,) targets (int for classification, float for regression)
        n_classes: number of output classes (1 for regression)
        task_type: 'classification' or 'regression'

    Returns:
        dict with 'accuracy' (classification) or 'r2' and 'mae' (regression)
    """
    hidden_dim = X_train.shape[1]

    if task_type == 'classification':
        probe = LinearProbe(hidden_dim, n_classes).to(device)
        y_train_t = y_train.long().to(device)
        y_test_t  = y_test.long().to(device)
    else:
        probe = LinearProbe(hidden_dim, 1).to(device)
        y_train_t = y_train.float().to(device)
        y_test_t  = y_test.float().to(device)

    X_train_t = X_train.float().to(device)
    X_test_t  = X_test.float().to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Training
    probe.train()
    for epoch in range(epochs):
        logits = probe(X_train_t)
        if task_type == 'classification':
            loss = F.cross_entropy(logits, y_train_t)
        else:
            loss = F.mse_loss(logits.squeeze(-1), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    probe.eval()
    with torch.no_grad():
        logits = probe(X_test_t)
        if task_type == 'classification':
            preds = logits.argmax(dim=-1)
            acc = (preds == y_test_t).float().mean().item()
            return {'accuracy': acc}
        else:
            preds = logits.squeeze(-1)
            mae = (preds - y_test_t).abs().mean().item()
            ss_res = ((y_test_t - preds) ** 2).sum().item()
            ss_tot = ((y_test_t - y_test_t.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / max(ss_tot, 1e-8)
            return {'r2': r2, 'mae': mae}


# ── Main probing pipeline ────────────────────────────────────────────────────

def run_probes(hidden_data, probe_position, train_frac=0.6, seed=42, device='cpu'):
    """
    Run all probes for a given position across all layers.

    Args:
        hidden_data: list of dicts from extract_hidden_states
        probe_position: 'hidden_colon', 'hidden_equals', or 'hidden_plus'
        train_frac: fraction of data for training the probe
        seed: random seed for train/test split
        device: torch device

    Returns:
        results: dict[target_name][layer] = metric_dict
        notation_results: dict[notation_pair][target_name][layer] = metric_dict
    """
    # Filter to examples that have this position
    valid = [d for d in hidden_data if d[probe_position] is not None]
    if not valid:
        return {}, {}

    n_layers = len(valid[0][probe_position])
    hidden_dim = valid[0][probe_position][0].shape[0]

    # Split into train/test for probe
    rng = random.Random(seed)
    indices = list(range(len(valid)))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_frac)
    train_idx = set(indices[:n_train])

    results = {}
    notation_results = defaultdict(lambda: {})

    for target_name, (n_classes, task_type) in PROBE_CONFIGS.items():
        results[target_name] = {}

        for layer in range(n_layers):
            # Build X and y
            X_all = torch.stack([d[probe_position][layer] for d in valid])
            y_all = torch.tensor([d['targets'][target_name] for d in valid])

            mask_train = torch.tensor([i in train_idx for i in range(len(valid))])
            X_train = X_all[mask_train]
            y_train = y_all[mask_train]
            X_test  = X_all[~mask_train]
            y_test  = y_all[~mask_train]

            if len(X_train) < 10 or len(X_test) < 10:
                continue

            metrics = train_probe(X_train, y_train, X_test, y_test,
                                  n_classes, task_type, device=device)
            results[target_name][layer] = metrics

        # Per-notation breakdown (test set only)
        notations = set(d['notation'] for d in valid)
        for notation in notations:
            notation_results[notation][target_name] = {}
            for layer in range(n_layers):
                # Use same train/test split but filter by notation in test
                X_all = torch.stack([d[probe_position][layer] for d in valid])
                y_all = torch.tensor([d['targets'][target_name] for d in valid])
                notations_all = [d['notation'] for d in valid]

                # Train on all train data (not filtered by notation)
                mask_train = torch.tensor([i in train_idx for i in range(len(valid))])
                X_train = X_all[mask_train]
                y_train = y_all[mask_train]

                # Test only on this notation pair
                mask_test_notation = torch.tensor([
                    (i not in train_idx) and (notations_all[i] == notation)
                    for i in range(len(valid))
                ])
                X_test = X_all[mask_test_notation]
                y_test = y_all[mask_test_notation]

                if len(X_train) < 10 or len(X_test) < 5:
                    continue

                metrics = train_probe(X_train, y_train, X_test, y_test,
                                      n_classes, task_type, device=device)
                notation_results[notation][target_name][layer] = metrics

    return results, dict(notation_results)


# ── Pretty printing ──────────────────────────────────────────────────────────

def print_probe_results(results, notation_results, position_name, n_layers):
    """Print a summary table of probe results."""
    print(f"\n{'='*70}")
    print(f"Probe position: {position_name}")
    print(f"{'='*70}")

    # Overall results
    print(f"\n{'Target':<14} {'Type':<6}", end='')
    for layer in range(n_layers):
        print(f"  {'L'+str(layer):>7}", end='')
    print()
    print('-' * (22 + 9 * n_layers))

    for target_name, (n_classes, task_type) in PROBE_CONFIGS.items():
        if target_name not in results:
            continue
        metric_key = 'accuracy' if task_type == 'classification' else 'r2'
        short_type = 'cls' if task_type == 'classification' else 'reg'
        print(f"{target_name:<14} {short_type:<6}", end='')
        for layer in range(n_layers):
            if layer in results[target_name]:
                val = results[target_name][layer][metric_key]
                print(f"  {val:>7.3f}", end='')
            else:
                print(f"  {'---':>7}", end='')
        print()

    # Per-notation breakdown (only for classification targets, last layer)
    if notation_results:
        print(f"\nPer-notation breakdown (last layer):")
        last_layer = n_layers - 1
        notations = sorted(notation_results.keys())

        for notation in notations:
            nA, nB = notation
            label = f"  {nA}+{nB}"
            print(f"\n{label}:")
            for target_name, (n_classes, task_type) in PROBE_CONFIGS.items():
                if target_name not in notation_results[notation]:
                    continue
                metric_key = 'accuracy' if task_type == 'classification' else 'r2'
                if last_layer in notation_results[notation][target_name]:
                    val = notation_results[notation][target_name][last_layer][metric_key]
                    print(f"    {target_name:<14} {val:.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Linear probing for hidden representations')
    p.add_argument('--ckpt',       required=True, help='Path to trained checkpoint')
    p.add_argument('--scaffold',   required=True,
                   choices=['none', 'old', 'state_seq', 'decomp', 'carry_explicit', 'digit',
                            'abacus_A', 'abacus_B', 'abacus_C', 'abacus_D'])
    p.add_argument('--model_size', choices=['small', 'large', 'medium', 'xlarge'], default='small')
    p.add_argument('--max_len',    type=int, default=64)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--positions',  nargs='+', default=['colon', 'equals', 'plus'],
                   choices=['colon', 'equals', 'plus'],
                   help='Which token positions to probe')
    p.add_argument('--probe_epochs', type=int, default=200,
                   help='Training epochs for each linear probe')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    random.seed(args.seed)

    # Load model
    use_abacus = is_abacus(args.scaffold)
    vocab_sz = ABACUS_VOCAB_SIZE if use_abacus else VOCAB_SIZE
    model = build_model(args.model_size, args.max_len, vocab_sz).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    load_state_dict_compat(model, ckpt['model'])

    n_layers = model.n_layer
    print(f"Model:    {args.model_size} ({count_params(model):,} params)")
    print(f"Scaffold: {args.scaffold}")
    print(f"Layers:   {n_layers}")
    print(f"Checkpoint: {args.ckpt}")

    # Build probe dataset from test facts
    print("\nBuilding probe dataset...")
    examples = build_probe_dataset(args.scaffold, split='test', seed=args.seed)
    print(f"  {len(examples)} examples")

    # Extract hidden states
    print("Extracting hidden states...")
    hidden_data = extract_hidden_states(model, examples, device, args.max_len, args.scaffold)
    print(f"  {len(hidden_data)} examples with valid hidden states")

    # Run probes for each position
    position_map = {
        'colon':  'hidden_colon',
        'equals': 'hidden_equals',
        'plus':   'hidden_plus',
    }

    for pos_name in args.positions:
        pos_key = position_map[pos_name]
        n_valid = sum(1 for d in hidden_data if d[pos_key] is not None)
        if n_valid == 0:
            print(f"\nSkipping position '{pos_name}': no valid examples")
            continue

        print(f"\nTraining probes at position '{pos_name}' ({n_valid} examples)...")
        results, notation_results = run_probes(
            hidden_data, pos_key,
            train_frac=0.6, seed=args.seed, device=device,
        )
        print_probe_results(results, notation_results, pos_name, n_layers)

    print("\nDone.")


if __name__ == '__main__':
    main()
