"""
Tokenizer, vocabulary, data generation for all scaffold formats, and train/test split.

Vocabulary (22 tokens, character-level):
  PAD=0  BOS=1  EOS=2
  '0'-'9' = 3-12
  'I','V','X','L','C' = 13-17
  '+','=',':',' ' = 18-21
"""

import random
import torch
from torch.utils.data import Dataset

# ── Vocabulary ────────────────────────────────────────────────────────────────

_SPECIAL = ['PAD', 'BOS', 'EOS']
_DIGITS  = list('0123456789')
_ROMAN   = list('IVXLC')
_OPS     = ['+', '=', ':', ' ']

VOCAB    = _SPECIAL + _DIGITS + _ROMAN + _OPS
VOCAB_SIZE = len(VOCAB)  # 22

tok2id = {t: i for i, t in enumerate(VOCAB)}
id2tok = {i: t for i, t in enumerate(VOCAB)}

PAD_ID = tok2id['PAD']   # 0
BOS_ID = tok2id['BOS']   # 1
EOS_ID = tok2id['EOS']   # 2


# ── Roman numeral conversion ──────────────────────────────────────────────────

_TO_ROMAN_TABLE = [
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'),  (9, 'IX'),  (5, 'V'),  (4, 'IV'), (1, 'I'),
]

def to_roman(n):
    """Convert integer 1-199 to Roman numeral string."""
    result = ''
    for value, numeral in _TO_ROMAN_TABLE:
        while n >= value:
            result += numeral
            n -= value
    return result

def from_roman(s):
    """Convert Roman numeral string to integer."""
    roman_vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    total = 0
    for i, ch in enumerate(s):
        v = roman_vals[ch]
        if i + 1 < len(s) and roman_vals[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total

def fmt(n, roman=False):
    return to_roman(n) if roman else str(n)


# ── Train/test split ──────────────────────────────────────────────────────────

def get_splits(seed=42):
    """
    Return (train_facts, test_facts) as lists of (A, B) pairs.
    All ordered pairs (A, B) with A, B in [1, 100].
    Fixed 8000/2000 train/test split by seed.
    """
    all_facts = [(a, b) for a in range(1, 101) for b in range(1, 101)]
    rng = random.Random(seed)
    rng.shuffle(all_facts)
    return all_facts[2000:], all_facts[:2000]  # train, test

_TRAIN_FACTS, _TEST_FACTS = get_splits()

def get_train_facts():
    return _TRAIN_FACTS

def get_test_facts():
    return _TEST_FACTS


# ── Scaffold format generators ────────────────────────────────────────────────

def make_expression(A, B, scaffold, roman_A=False, roman_B=False, roman_C=False):
    """
    Generate a single training expression string.
    Chain-of-thought (after ':') is always Hindu-Arabic regardless of notation flags.
    roman_C only applies to plain (no scaffold) format.
    """
    C = A + B

    if scaffold == 'none':
        return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} = {fmt(C, roman_C)}"

    if scaffold == 'old':
        # A + B : a + b = C  (translation scaffold)
        return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {A} + {B} = {C}"

    if scaffold == 'state_seq':
        # A + B : a + B_tens = mid [+ B_units = C]
        B_tens  = (B // 10) * 10
        B_units = B % 10
        mid     = A + B_tens
        if B_tens > 0 and B_units > 0:
            s = f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {A} + {B_tens} = {mid} + {B_units} = {C}"
        elif B_tens == 0:
            s = f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {A} + {B_units} = {C}"
        else:  # B_units == 0, round tens
            s = f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {A} + {B_tens} = {C}"
        return s

    if scaffold == 'decomp':
        # A + B : a_t×10+b_t×10=tens a_u+b_u=units = C
        A_tens  = (A // 10) * 10
        A_units = A % 10
        B_tens  = (B // 10) * 10
        B_units = B % 10
        if A >= 10 and B >= 10:
            tens_sum  = A_tens + B_tens
            units_sum = A_units + B_units
            return (f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : "
                    f"{A_tens}+{B_tens}={tens_sum} {A_units}+{B_units}={units_sum} = {C}")
        else:
            # One or both operands are single-digit: single step
            return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {A}+{B}={C} = {C}"

    if scaffold == 'carry_explicit':
        # A + B : a_u+b_u=units_sum [a_t+b_t[+carry]=tens_sum] = C
        A_units = A % 10
        B_units = B % 10
        A_tens  = A // 10
        B_tens  = B // 10
        units_sum = A_units + B_units
        carry     = units_sum // 10
        C_tens    = A_tens + B_tens + carry

        if A_tens == 0 and B_tens == 0:
            # Both single-digit: no tens column
            return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : {A_units}+{B_units}={units_sum} = {C}"
        else:
            if carry:
                tens_step = f"{A_tens}+{B_tens}+{carry}={C_tens}"
            else:
                tens_step = f"{A_tens}+{B_tens}={C_tens}"
            return (f"{fmt(A, roman_A)} + {fmt(B, roman_B)} : "
                    f"{A_units}+{B_units}={units_sum} {tens_step} = {C}")

    if scaffold == 'digit':
        # A + B : [digits A rtl] + [digits B rtl] : [carry steps] = C
        # Hindu-Arabic only; no roman flags used
        preamble_A = ' '.join(reversed(str(A)))
        preamble_B = ' '.join(reversed(str(B)))
        steps = _carry_steps_str(A, B)
        return f"{A} + {B} : {preamble_A} + {preamble_B} : {steps} = {C}"

    raise ValueError(f"Unknown scaffold: {scaffold!r}")


def _carry_steps_str(A, B):
    """Generate carry-explicit step strings for A+B (variable digit length)."""
    da = [int(c) for c in reversed(str(A))]
    db = [int(c) for c in reversed(str(B))]
    n  = max(len(da), len(db))
    da += [0] * (n - len(da))
    db += [0] * (n - len(db))
    steps = []
    carry = 0
    for i in range(n):
        col_sum  = da[i] + db[i] + carry
        carry_in = carry
        carry    = col_sum // 10
        if carry_in:
            steps.append(f"{da[i]}+{db[i]}+{carry_in}={col_sum}")
        else:
            steps.append(f"{da[i]}+{db[i]}={col_sum}")
    return ' '.join(steps)


def make_dataset(scaffold, split='train', seed=42, digit_range=(1, 100)):
    """
    Return list of expression strings for the given scaffold and split.
    For 'digit' scaffold, uses digit_range (default 1-999 from Phase 9).
    For all other scaffolds, uses the fixed 1-100 fact split.
    """
    if scaffold == 'digit':
        lo, hi = digit_range if digit_range != (1, 100) else (1, 999)
        all_facts = [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1)]
        rng = random.Random(seed)
        rng.shuffle(all_facts)
        n_test  = max(1, len(all_facts) // 5)
        if split == 'train':
            facts = all_facts[n_test:]
        else:
            facts = all_facts[:n_test]
        return [make_expression(a, b, 'digit') for (a, b) in facts]

    facts = get_train_facts() if split == 'train' else get_test_facts()

    if scaffold == 'none':
        # All 8 notation combinations for pretraining
        exprs = []
        for (A, B) in facts:
            for rA in (False, True):
                for rB in (False, True):
                    for rC in (False, True):
                        exprs.append(make_expression(A, B, 'none', rA, rB, rC))
        return exprs

    # Scaffolds: all 4 input combinations (rA, rB); result is always Hindu-Arabic
    exprs = []
    for (A, B) in facts:
        for rA in (False, True):
            for rB in (False, True):
                exprs.append(make_expression(A, B, scaffold, rA, rB))
    return exprs


# ── Tokenization ──────────────────────────────────────────────────────────────

def encode(s, max_len):
    """
    Encode string s as [BOS] + chars + [EOS], right-padded to max_len with PAD.
    Right-padding keeps BOS at position 0 so positional embeddings are consistent
    between training and evaluation.
    Returns a list of token ids.
    """
    ids = [BOS_ID] + [tok2id[c] for c in s] + [EOS_ID]
    ids = ids[:max_len]
    pad_len = max_len - len(ids)
    return ids + [PAD_ID] * pad_len

def decode(ids):
    """Decode list of token ids to string, stripping BOS/EOS/PAD."""
    tokens = []
    for i in ids:
        if i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        tokens.append(id2tok[i])
    return ''.join(tokens)

def encode_prompt(prompt_str, max_len):
    """
    Encode a prompt string (no EOS) for generation — no padding.
    BOS is at position 0, matching training layout.
    """
    ids = [BOS_ID] + [tok2id[c] for c in prompt_str]
    ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)


# ── Collation ─────────────────────────────────────────────────────────────────

def collate_lm(strings, max_len):
    """
    Collate a list of strings into (input_ids, target_ids) tensors for LM training.
    input  = encoded[:-1]  (all but last token)
    target = encoded[1:]   (all but first token)
    """
    seqs = [encode(s, max_len) for s in strings]
    t    = torch.tensor(seqs, dtype=torch.long)
    return t[:, :-1], t[:, 1:]


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(completion_str):
    """
    Extract the final numeric answer from a completion string.
    Looks for the last '= <number>' pattern. Returns int or None.
    """
    parts = completion_str.replace('BOS', '').replace('EOS', '').replace('PAD', '')
    # Find all '= N' patterns; take the last one
    import re
    matches = re.findall(r'=\s*(\d+)', parts)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return None
    return None


def extract_step_answer(completion_str):
    """
    Extract the first intermediate sum from a state_seq or carry_explicit completion.
    For state_seq: first '= N' after the ':'
    Returns int or None.
    """
    import re
    colon_pos = completion_str.find(':')
    if colon_pos == -1:
        return None
    after_colon = completion_str[colon_pos + 1:]
    matches = re.findall(r'=\s*(\d+)', after_colon)
    if matches:
        try:
            return int(matches[0])
        except ValueError:
            return None
    return None


# ── Prompt construction for eval/RL ──────────────────────────────────────────

def make_prompt(A, B, scaffold, roman_A=False, roman_B=False):
    """
    Return the prompt portion of an expression (everything up to and including
    the final '= ' that the model must complete).
    For 'none' scaffold, prompt is 'A + B = '.
    For scaffolded formats, prompt is the full expression up to the final ' = '.
    """
    C = A + B

    if scaffold == 'none':
        return f"{fmt(A, roman_A)} + {fmt(B, roman_B)} = "

    # Build the full expression and truncate at the last ' = '
    full = make_expression(A, B, scaffold, roman_A, roman_B)
    # Find last occurrence of ' = '
    idx = full.rfind(' = ')
    if idx == -1:
        return full
    return full[:idx + 3]  # include ' = '


# ── Dataset class ─────────────────────────────────────────────────────────────

class ArithmeticDataset(Dataset):
    def __init__(self, scaffold, split, max_len, seed=42, digit_range=(1, 999)):
        self.max_len = max_len
        if scaffold == 'digit':
            self.strings = make_dataset(scaffold, split, seed, digit_range)
        else:
            self.strings = make_dataset(scaffold, split, seed)
        # Pre-encode all strings
        self.data = [encode(s, max_len) for s in self.strings]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:]  # input, target
