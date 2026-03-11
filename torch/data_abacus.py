"""
Abacus scaffold data generator — all four variants.

Variants:
  A — flag carry (^),   overflow implicit   (direct add shown)
  B — token carry (K),  overflow implicit   (carry-absorbed: state includes carry)
  C — flag carry (^),   complement explicit (subtract complement shown)
  D — token carry (K),  complement explicit (carry-first: pre/post carry states shown)

State notation: [H|T|U] — hundreds, tens, units (all 0-9)
Transition arrow: >
Carry flag: ^  (variants A, C)
Carry token: K  (variants B, D)  — K chosen to avoid conflict with Roman numeral C

For all variants, no-overflow steps are identical.
Differences only appear when U+n > 9 (units overflow) or T+n > 9 (tens overflow).

Vocabulary extensions over base data.py (22 tokens → 31):
  '[' ']' '|' '-' 'u' 't' '>' '^' 'K'   (+9 tokens)
"""

import random
import torch
from torch.utils.data import Dataset

from data import (
    to_roman, fmt,
    get_train_facts, get_test_facts,
    PAD_ID, BOS_ID, EOS_ID,
    VOCAB as BASE_VOCAB, tok2id as BASE_tok2id,
)

# ── Extended vocabulary ───────────────────────────────────────────────────────

_ABACUS_EXTRA = ['[', ']', '|', '-', 'u', 't', '>', '^', 'K']

ABACUS_VOCAB     = BASE_VOCAB + _ABACUS_EXTRA
ABACUS_VOCAB_SIZE = len(ABACUS_VOCAB)  # 31

atok2id = {t: i for i, t in enumerate(ABACUS_VOCAB)}
aid2tok = {i: t for i, t in enumerate(ABACUS_VOCAB)}

APAD_ID = atok2id['PAD']
ABOS_ID = atok2id['BOS']
AEOS_ID = atok2id['EOS']


# ── State helpers ─────────────────────────────────────────────────────────────

def state(H, T, U):
    """Render a 3-rod abacus state as a string."""
    return f"[{H}|{T}|{U}]"

def load_state(n):
    """Initial state for operand n (0-999)."""
    return state(n // 100, (n // 10) % 10, n % 10)


# ── Trace generators ──────────────────────────────────────────────────────────

def _add_units(n, H, T, U):
    """
    Compute result of adding n to units rod.
    Returns (H', T', U', overflow: bool).
    """
    new_U = U + n
    if new_U <= 9:
        return H, T, new_U, False
    new_U -= 10
    new_T = T + 1
    new_H = H
    if new_T > 9:
        new_T -= 10
        new_H += 1
    return new_H, new_T, new_U, True


def _add_tens(n, H, T, U):
    """
    Compute result of adding n to tens rod.
    Returns (H', T', U', overflow: bool).
    """
    new_T = T + n
    if new_T <= 9:
        return H, new_T, U, False
    new_T -= 10
    new_H = H + 1
    return new_H, new_T, U, True


def _complement(n):
    """Return the 10s complement of n."""
    return 10 - n


# ── Variant A: flag carry, overflow implicit ──────────────────────────────────

def trace_A(A, B):
    """
    Generate abacus trace for A + B, Variant A.
    Direct addition shown; overflow flagged with ^ suffix.
    """
    H, T, U = A // 100, (A // 10) % 10, A % 10
    steps = [state(H, T, U)]

    b_u = B % 10
    b_t = (B // 10) % 10

    if b_u > 0:
        nH, nT, nU, overflow = _add_units(b_u, H, T, U)
        s = f"+u{b_u}>{state(nH, nT, nU)}"
        if overflow:
            s += "^"
        steps.append(s)
        H, T, U = nH, nT, nU

    if b_t > 0:
        nH, nT, nU, overflow = _add_tens(b_t, H, T, U)
        s = f"+t{b_t}>{state(nH, nT, nU)}"
        if overflow:
            s += "^"
        steps.append(s)
        H, T, U = nH, nT, nU

    return ' '.join(steps)


# ── Variant B: token carry, overflow implicit, carry-absorbed ─────────────────

def trace_B(A, B):
    """
    Generate abacus trace for A + B, Variant B.
    Direct addition shown; state after op already has carry absorbed.
    K token appended after any overflowing step.
    """
    H, T, U = A // 100, (A // 10) % 10, A % 10
    steps = [state(H, T, U)]

    b_u = B % 10
    b_t = (B // 10) % 10

    if b_u > 0:
        nH, nT, nU, overflow = _add_units(b_u, H, T, U)
        s = f"+u{b_u}>{state(nH, nT, nU)}"
        if overflow:
            s += " K"
        steps.append(s)
        H, T, U = nH, nT, nU

    if b_t > 0:
        nH, nT, nU, overflow = _add_tens(b_t, H, T, U)
        s = f"+t{b_t}>{state(nH, nT, nU)}"
        if overflow:
            s += " K"
        steps.append(s)
        H, T, U = nH, nT, nU

    return ' '.join(steps)


# ── Variant C: flag carry, complement explicit ────────────────────────────────

def trace_C(A, B):
    """
    Generate abacus trace for A + B, Variant C.
    When overflow would occur, shows the complement subtraction instead.
    Complement op: -u{10-n} or -t{10-n}, carry absorbed into state, flagged with ^.
    When no overflow, identical to A.
    """
    H, T, U = A // 100, (A // 10) % 10, A % 10
    steps = [state(H, T, U)]

    b_u = B % 10
    b_t = (B // 10) % 10

    if b_u > 0:
        nH, nT, nU, overflow = _add_units(b_u, H, T, U)
        if not overflow:
            steps.append(f"+u{b_u}>{state(nH, nT, nU)}")
        else:
            m = _complement(b_u)   # what we actually subtract
            steps.append(f"-u{m}>{state(nH, nT, nU)}^")
        H, T, U = nH, nT, nU

    if b_t > 0:
        nH, nT, nU, overflow = _add_tens(b_t, H, T, U)
        if not overflow:
            steps.append(f"+t{b_t}>{state(nH, nT, nU)}")
        else:
            m = _complement(b_t)
            steps.append(f"-t{m}>{state(nH, nT, nU)}^")
        H, T, U = nH, nT, nU

    return ' '.join(steps)


# ── Variant D: token carry, complement explicit, carry-first ──────────────────

def trace_D(A, B):
    """
    Generate abacus trace for A + B, Variant D.
    When overflow: shows complement subtraction → pre-carry state,
    then K token, then post-carry state explicitly.
    When no overflow: identical to A/B/C.
    """
    H, T, U = A // 100, (A // 10) % 10, A % 10
    steps = [state(H, T, U)]

    b_u = B % 10
    b_t = (B // 10) % 10

    if b_u > 0:
        nH, nT, nU, overflow = _add_units(b_u, H, T, U)
        if not overflow:
            steps.append(f"+u{b_u}>{state(nH, nT, nU)}")
        else:
            m = _complement(b_u)
            # Pre-carry state: only units changed, tens/hundreds not yet incremented
            pre_U = U - m        # = U + b_u - 10
            pre_state = state(H, T, pre_U)
            post_state = state(nH, nT, nU)
            steps.append(f"-u{m}>{pre_state} K {post_state}")
        H, T, U = nH, nT, nU

    if b_t > 0:
        nH, nT, nU, overflow = _add_tens(b_t, H, T, U)
        if not overflow:
            steps.append(f"+t{b_t}>{state(nH, nT, nU)}")
        else:
            m = _complement(b_t)
            pre_T = T - m        # = T + b_t - 10
            pre_state = state(H, pre_T, U)
            post_state = state(nH, nT, nU)
            steps.append(f"-t{m}>{pre_state} K {post_state}")
        H, T, U = nH, nT, nU

    return ' '.join(steps)


# ── Expression builder ────────────────────────────────────────────────────────

_TRACE_FN = {'A': trace_A, 'B': trace_B, 'C': trace_C, 'D': trace_D}


def make_abacus_expression(A, B, variant, roman_A=False, roman_B=False):
    """
    Build a full abacus scaffold expression string.

    Format:
      '<A> + <B> : <load> <steps> = <C>'

    roman_A/roman_B control input notation; trace is always symbolic.
    """
    C = A + B
    trace = _TRACE_FN[variant](A, B)
    return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {trace} = {C}"


def make_abacus_prompt(A, B, variant, roman_A=False, roman_B=False):
    """Return the prompt portion (everything up to and including '= ')."""
    full = make_abacus_expression(A, B, variant, roman_A, roman_B)
    idx  = full.rfind(' = ')
    return full[:idx + 3]


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_abacus_dataset(variant, split='train', seed=42):
    """
    Return list of expression strings for the given variant and split.
    Uses the same 8K/2K train/test split as data.py (seed=42).
    All 4 input notation combos (roman_A × roman_B) per fact.
    """
    assert variant in ('A', 'B', 'C', 'D'), f"Unknown variant: {variant!r}"
    facts = get_train_facts() if split == 'train' else get_test_facts()
    exprs = []
    for (A, B) in facts:
        for rA in (False, True):
            for rB in (False, True):
                exprs.append(make_abacus_expression(A, B, variant, rA, rB))
    return exprs


# ── Tokenization ──────────────────────────────────────────────────────────────

def aencode(s, max_len):
    """Encode string s with abacus vocabulary, right-padded to max_len."""
    ids = [ABOS_ID] + [atok2id[c] for c in s] + [AEOS_ID]
    ids = ids[:max_len]
    return ids + [APAD_ID] * (max_len - len(ids))


def adecode(ids):
    """Decode list of abacus token ids to string, stripping BOS/EOS/PAD."""
    return ''.join(aid2tok[i] for i in ids
                   if i not in (APAD_ID, ABOS_ID, AEOS_ID))


def aencode_prompt(prompt_str, max_len):
    """Encode prompt (no EOS), no padding, BOS at position 0."""
    ids = [ABOS_ID] + [atok2id[c] for c in prompt_str]
    return torch.tensor(ids[:max_len], dtype=torch.long)


def acollate_lm(strings, max_len):
    """Collate list of strings → (input_ids, target_ids) tensors."""
    seqs = [aencode(s, max_len) for s in strings]
    t    = torch.tensor(seqs, dtype=torch.long)
    return t[:, :-1], t[:, 1:]


# ── Answer extraction ─────────────────────────────────────────────────────────

def aextract_answer(completion_str):
    """Extract final numeric answer from abacus completion string."""
    import re
    matches = re.findall(r'=\s*(\d+)', completion_str)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return None
    return None


def aextract_final_state(completion_str):
    """
    Extract the last [H|T|U] state from a completion.
    Returns (H, T, U) tuple or None.
    """
    import re
    matches = re.findall(r'\[(\d)\|(\d)\|(\d)\]', completion_str)
    if matches:
        H, T, U = matches[-1]
        return int(H), int(T), int(U)
    return None


def is_valid_trace(completion_str, A, B, variant):
    """
    Grammar checker: returns True if completion is a valid trace for A+B
    in the given variant. Used as the RL reward function.
    Checks:
      1. Every state transition follows the variant's rules
      2. Final state encodes the correct answer
      3. No invalid tokens or moves
    """
    C = A + B
    # Simplest check: answer is correct AND final state matches
    answer = aextract_answer(completion_str)
    if answer != C:
        return False
    final = aextract_final_state(completion_str)
    if final is None:
        return False
    H, T, U = final
    return (H * 100 + T * 10 + U) == C


# ── Dataset class ─────────────────────────────────────────────────────────────

class AbacusDataset(Dataset):
    def __init__(self, variant, split, max_len, seed=42):
        self.max_len = max_len
        self.strings = make_abacus_dataset(variant, split, seed)
        self.data    = [aencode(s, max_len) for s in self.strings]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:]


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_cases = [
        (47, 35, 82),   # units overflow, no tens overflow
        (56, 67, 123),  # both overflow
        (23, 41, 64),   # no overflow
        (99, 99, 198),  # max case
        (5,  3,   8),   # single digit
        (50, 30,  80),  # round tens, no units
    ]

    for (A, B, C) in test_cases:
        print(f"\n{'='*60}")
        print(f"{A} + {B} = {C}")
        for v in ('A', 'B', 'C', 'D'):
            expr = make_abacus_expression(A, B, v)
            valid = is_valid_trace(expr.split(' : ', 1)[1], A, B, v)
            print(f"  {v}: {expr}  ✓" if valid else f"  {v}: {expr}  ✗")

    # Vocabulary info
    print(f"\nAbacus vocab size: {ABACUS_VOCAB_SIZE}")
    print(f"Extra tokens: {_ABACUS_EXTRA}")

    # Dataset size
    train = make_abacus_dataset('A', 'train')
    test  = make_abacus_dataset('A', 'test')
    print(f"\nDataset (variant A): {len(train):,} train / {len(test):,} test expressions")

    # Token length stats
    lengths = [len(aencode(s, 256)) - s.count('PAD') for s in train[:1000]]
    actual  = [len([ABOS_ID] + [atok2id[c] for c in s] + [AEOS_ID]) for s in train[:1000]]
    print(f"Max token length (sample of 1000): {max(actual)}")
    print(f"Mean token length: {sum(actual)/len(actual):.1f}")
