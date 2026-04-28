"""
Stateless abacus simulator for tool-use interaction.

The model issues commands and the simulator returns the resulting abacus state.
This separates "knowing what to do" from "simulating the computation."

Two command-tokenization modes:

  mode='opaque'        (default; SP1 baseline)
    Commands use opaque rod letters: '+u5' (units), '+t3' (tens).
    Each command is 3 tokens: '+', 'u'|'t', digit.
    Defined only for rods 0,1 (the existing 22-token base vocab has no h/th).

  mode='compositional' (SP2)
    Commands use a numeric rod index: '+05' (rod 0 = units), '+13' (rod 1 = tens),
    '+25' (rod 2 = hundreds), and so on. Each command is still 3 tokens:
    '+', digit (rod_idx), digit (value). The rod index reuses the existing
    digit vocab — so a hundreds-rod command needs no new tokens. This is the
    mechanism that lets the model potentially generalize across rod indices.

Rod count is parameterized via `n_rods` (default 3 for backward compat with SP1).
State notation: `[d_{n-1}|...|d_1|d_0]` — highest rod leftmost, units rightmost.
For n_rods=3: '[H|T|U]'. For n_rods=4: '[K|H|T|U]'.
The carry overflow flag '^' is appended when carry escapes the highest rod.

Usage:
    sim = AbacusSimulator()                                    # 3-rod opaque
    state = sim.reset(47)                                      # '[0|4|7]'
    state = sim.step(state, '+u5')                             # '[0|5|2]^'

    sim = AbacusSimulator(mode='compositional', n_rods=4)      # 4-rod compositional
    state = sim.reset(247)                                     # '[0|2|4|7]'
    state = sim.step(state, '+25')                             # add 5 to hundreds
"""

import re


# Map between rod index (positional) and the opaque rod letter, where defined.
# Opaque mode is only meaningful for rods 0,1 (no opaque letter for rod ≥ 2).
_ROD_IDX_TO_LETTER = {0: 'u', 1: 't'}
_ROD_LETTER_TO_IDX = {v: k for k, v in _ROD_IDX_TO_LETTER.items()}


class AbacusSimulator:
    """Stateless N-rod abacus simulator (variant A: flag carry, overflow implicit).

    Parameters:
        mode    — 'opaque' (SP1) or 'compositional' (SP2)
        n_rods  — number of rods (default 3); state has digits 0..10^n - 1
    """

    def __init__(self, mode='opaque', n_rods=3):
        if mode not in ('opaque', 'compositional'):
            raise ValueError(f"mode must be 'opaque' or 'compositional', got {mode!r}")
        if mode == 'opaque' and n_rods > 2:
            # Opaque mode only defines letter codes for rods 0,1; refuse rather than silently misbehave.
            # Higher-rod operations under opaque mode would have to invent new tokens — not in scope.
            pass  # We allow it, but gold_trajectory will only emit +u/+t commands; carry handles higher rods automatically.
        if n_rods < 1:
            raise ValueError(f"n_rods must be ≥ 1, got {n_rods}")
        self.mode = mode
        self.n_rods = n_rods
        self.max_value = 10 ** n_rods - 1

    def reset(self, A):
        """Initialize abacus with operand A. Returns state string."""
        if A < 0 or A > self.max_value:
            raise ValueError(f"A={A} out of range [0, {self.max_value}] for {self.n_rods}-rod abacus")
        digits = [(A // 10**i) % 10 for i in range(self.n_rods)]  # digits[0]=units
        return self._format_state(digits)

    def step(self, state_str, command_str):
        """Execute a single-rod add command; return new state with optional '^' overflow flag.

        '^' fires when the *targeted rod* overflowed during this command —
        i.e., digits[rod_idx] + n >= 10. This matches SP1's data semantics
        (the flag is a per-step "this step had a carry" marker, not a global
        "the abacus overflowed" marker).
        """
        digits = self.parse_state(state_str)
        op, rod_idx, n = self._parse_command(command_str)

        if op != '+':
            raise ValueError(f"Only '+' supported in this build, got {op!r}")
        if rod_idx < 0 or rod_idx >= self.n_rods:
            raise ValueError(f"Rod {rod_idx} out of range [0, {self.n_rods}) for {self.n_rods}-rod sim")

        # SP1-compatible overflow semantics: did the targeted rod itself overflow?
        overflow = (digits[rod_idx] + n) >= 10

        # Propagate carry leftward through higher rods (regardless of overflow flag).
        carry = n
        i = rod_idx
        while i < self.n_rods and carry > 0:
            total = digits[i] + carry
            digits[i] = total % 10
            carry = total // 10
            i += 1
        # If carry > 0 here, it escaped the highest rod — silently dropped.
        # (Caller can detect this via state_to_int comparison if needed.)

        result = self._format_state(digits)
        if overflow:
            result += '^'
        return result

    def parse_state(self, state_str):
        """Parse '[d_{n-1}|...|d_0]' (with optional trailing '^') into [d_0, d_1, ..., d_{n-1}]."""
        s = state_str.rstrip('^')
        m = re.match(r'\[(\d(?:\|\d)*)\]$', s)
        if not m:
            raise ValueError(f"Invalid state: {state_str!r}")
        parts = m.group(1).split('|')
        if len(parts) != self.n_rods:
            raise ValueError(
                f"Expected {self.n_rods}-rod state, got {len(parts)} digits in {state_str!r}"
            )
        # display order is highest|...|units; reverse to get digits[0]=units
        return [int(p) for p in reversed(parts)]

    def state_to_int(self, state_str):
        digits = self.parse_state(state_str)
        return sum(d * 10**i for i, d in enumerate(digits))

    def _format_state(self, digits):
        """Render a digit list (digits[0]=units) as '[d_{n-1}|...|d_0]'."""
        return '[' + '|'.join(str(d) for d in reversed(digits)) + ']'

    def _parse_command(self, cmd):
        """Parse a command string into (op, rod_idx, n).

        opaque mode:        '+u5' -> ('+', 0, 5);  '+t3' -> ('+', 1, 3)
        compositional mode: '+05' -> ('+', 0, 5);  '+13' -> ('+', 1, 3); '+24' -> ('+', 2, 4)
        """
        if self.mode == 'opaque':
            m = re.match(r'([+-])([ut])(\d)$', cmd)
            if not m:
                raise ValueError(f"Invalid opaque command: {cmd!r}")
            return m.group(1), _ROD_LETTER_TO_IDX[m.group(2)], int(m.group(3))
        else:  # compositional
            m = re.match(r'([+-])(\d)(\d)$', cmd)
            if not m:
                raise ValueError(f"Invalid compositional command: {cmd!r}")
            return m.group(1), int(m.group(2)), int(m.group(3))

    def _format_command(self, op, rod_idx, n):
        """Render a command in the active mode."""
        if self.mode == 'opaque':
            if rod_idx not in _ROD_IDX_TO_LETTER:
                raise ValueError(f"Opaque mode has no letter for rod {rod_idx}")
            return f"{op}{_ROD_IDX_TO_LETTER[rod_idx]}{n}"
        else:
            if not 0 <= rod_idx <= 9:
                raise ValueError(f"Compositional rod_idx must be 0-9 (single digit), got {rod_idx}")
            return f"{op}{rod_idx}{n}"

    def gold_trajectory(self, A, B):
        """
        Sequence of (command, response) pairs for A + B.

        Decomposes B by digit position and emits one command per nonzero digit:
        rod 0 (units), rod 1 (tens), rod 2 (hundreds), .... Stops at the highest
        rod the simulator supports; B's higher digits would have to be handled
        by carry from below, which won't happen except via the model's commands.

        In opaque mode, only rods 0 and 1 are emitted (no letter for higher
        rods); B with a hundreds digit can't be decomposed and raises.
        """
        if A < 0 or A > self.max_value:
            raise ValueError(f"A={A} out of range [0, {self.max_value}]")
        if B < 0 or B > self.max_value:
            raise ValueError(f"B={B} out of range [0, {self.max_value}]")

        # Decompose B into rod-indexed digits
        b_digits = [(B // 10**i) % 10 for i in range(self.n_rods)]

        if self.mode == 'opaque':
            # opaque has letters only for rods 0,1 — refuse if B has higher digits
            for i, d in enumerate(b_digits):
                if i >= 2 and d > 0:
                    raise ValueError(
                        f"B={B} has nonzero rod-{i} digit but opaque mode only emits +u/+t commands"
                    )

        state = self.reset(A)
        trajectory = [{'command': None, 'response': state}]

        for i, d in enumerate(b_digits):
            if d == 0:
                continue
            if self.mode == 'opaque' and i not in _ROD_IDX_TO_LETTER:
                # Already raised above; defensive
                raise ValueError(f"opaque mode can't emit a command for rod {i}")
            cmd = self._format_command('+', i, d)
            state = self.step(state.rstrip('^'), cmd)
            trajectory.append({'command': cmd, 'response': state})

        return trajectory


if __name__ == '__main__':
    test_cases_3 = [
        (47, 35, 82),
        (56, 67, 123),
        (23, 41, 64),
        (99, 99, 198),
        (5, 3, 8),
        (50, 30, 80),
        (100, 50, 150),  # A=100: loads as [1|0|0]
        (1, 99, 100),    # near-boundary
    ]
    for mode in ('opaque', 'compositional'):
        sim = AbacusSimulator(mode=mode, n_rods=3)
        print(f'\n--- 3-rod, mode={mode} ---')
        for A, B, C in test_cases_3:
            traj = sim.gold_trajectory(A, B)
            final = sim.state_to_int(traj[-1]['response'].rstrip('^'))
            status = 'ok' if final == C else 'FAIL'
            steps = ' '.join(
                f"{t['command']}>{t['response']}" if t['command'] else t['response']
                for t in traj
            )
            print(f'{A:3d} + {B:3d} = {C:3d}  [{status}]  {steps}')

    # 4-rod, compositional only (opaque has no letter for the thousands rod)
    test_cases_4 = [
        (247, 358, 605),    # rod-2 commands required
        (1234, 567, 1801),
        (50, 950, 1000),
        (9999, 1, 10000),   # overflows the highest rod
        (47, 35, 82),       # same as 3-rod test, padded
    ]
    sim = AbacusSimulator(mode='compositional', n_rods=4)
    print('\n--- 4-rod, mode=compositional ---')
    for A, B, C in test_cases_4:
        try:
            traj = sim.gold_trajectory(A, B)
            final_state = traj[-1]['response']
            final_int = sim.state_to_int(final_state)
            overflow = '^' in final_state
            # If overflow occurred, the integer value wraps; handle both checks
            status = 'ok' if (final_int == C or (overflow and final_int + 10**4 == C)) else 'FAIL'
            steps = ' '.join(
                f"{t['command']}>{t['response']}" if t['command'] else t['response']
                for t in traj
            )
            print(f'{A:4d} + {B:4d} = {C:5d}  [{status}]  {steps}')
        except ValueError as e:
            print(f'{A:4d} + {B:4d} = {C:5d}  [SKIP] {e}')
