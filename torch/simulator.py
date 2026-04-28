"""
Stateless abacus simulator for tool-use interaction.

The model issues commands and the simulator returns the resulting abacus state.
This separates "knowing what to do" from "simulating the computation."

Two command-tokenization modes are supported:

  mode='opaque'        (default; SP1 baseline)
    Commands use opaque rod letters: '+u5' (units), '+t3' (tens).
    Each command is 3 tokens: '+', 'u'|'t', digit.

  mode='compositional' (SP2)
    Commands use a numeric rod index: '+05' (rod 0 = units), '+13' (rod 1 = tens).
    Each command is still 3 tokens: '+', digit (rod_idx), digit (value).
    The rod index is itself a digit in the existing 0-9 vocab — so extending to
    a hundreds rod is '+2d', a thousands rod is '+3d', and so on, with no new
    vocabulary tokens. This is the mechanism that lets the model potentially
    generalize across rod indices it hasn't seen during training.

Usage:
    sim = AbacusSimulator()                       # opaque (default)
    state = sim.reset(47)                         # '[0|4|7]'
    state = sim.step(state, '+u5')                # '[0|5|2]^'

    sim = AbacusSimulator(mode='compositional')
    state = sim.reset(47)                         # '[0|4|7]'
    state = sim.step(state, '+05')                # '[0|5|2]^'
"""

import re
from data_abacus import _add_units, _add_tens, state as fmt_state


# Map between rod index (positional) and the opaque rod letter, where defined.
# 0=units, 1=tens, 2=hundreds. Higher rods have no opaque letter.
_ROD_IDX_TO_LETTER = {0: 'u', 1: 't'}
_ROD_LETTER_TO_IDX = {v: k for k, v in _ROD_IDX_TO_LETTER.items()}


class AbacusSimulator:
    """Stateless abacus simulator for variant A (flag carry, overflow implicit).

    The `mode` parameter selects the command tokenization:
      'opaque'        — '+u5', '+t3'   (default; SP1)
      'compositional' — '+05', '+13'   (SP2)
    """

    def __init__(self, mode='opaque'):
        if mode not in ('opaque', 'compositional'):
            raise ValueError(f"mode must be 'opaque' or 'compositional', got {mode!r}")
        self.mode = mode

    def reset(self, A):
        """Initialize abacus with operand A (0-999). Returns state string."""
        return fmt_state(A // 100, (A // 10) % 10, A % 10)

    def step(self, state_str, command_str):
        """
        Execute command on state.

        Args:
            state_str: '[H|T|U]'
            command_str: '+u5'/'+t3' (opaque mode) or '+05'/'+13' (compositional)

        Returns:
            response: state string, optionally with '^' suffix for overflow
        """
        H, T, U = self.parse_state(state_str)
        op, rod_idx, n = self._parse_command(command_str)

        if rod_idx == 0:
            nH, nT, nU, overflow = _add_units(n, H, T, U)
        elif rod_idx == 1:
            nH, nT, nU, overflow = _add_tens(n, H, T, U)
        else:
            raise ValueError(f"Unsupported rod index: {rod_idx} (3-rod simulator supports rods 0,1)")

        result = fmt_state(nH, nT, nU)
        if overflow:
            result += '^'
        return result

    def parse_state(self, state_str):
        """Parse '[H|T|U]' -> (H, T, U)."""
        m = re.match(r'\[(\d)\|(\d)\|(\d)\]', state_str)
        if not m:
            raise ValueError(f"Invalid state: {state_str!r}")
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    def state_to_int(self, state_str):
        """Convert '[H|T|U]' to integer."""
        H, T, U = self.parse_state(state_str)
        return H * 100 + T * 10 + U

    def _parse_command(self, cmd):
        """Parse a command string into (op, rod_idx, n).

        opaque mode:        '+u5' -> ('+', 0, 5);  '+t3' -> ('+', 1, 3)
        compositional mode: '+05' -> ('+', 0, 5);  '+13' -> ('+', 1, 3)
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
            return f"{op}{_ROD_IDX_TO_LETTER[rod_idx]}{n}"
        else:
            return f"{op}{rod_idx}{n}"

    def gold_trajectory(self, A, B):
        """
        Return the gold sequence of (command, response) pairs for A+B.

        Decomposes B into units and tens digits. Only supports B in [0, 99]
        since the 3-rod simulator only handles rods 0 (units) and 1 (tens).
        A can be up to 999 (loaded directly onto the abacus).

        Returns:
            list of dicts with keys 'command' and 'response'

        Raises:
            ValueError: if B > 99 (can't be decomposed into single-digit commands)
        """
        if B > 99:
            raise ValueError(f"B={B} exceeds 99; 3-rod simulator can only target rods 0,1")

        state = self.reset(A)
        trajectory = [{'command': None, 'response': state}]  # initial state

        b_u = B % 10
        b_t = (B // 10) % 10

        if b_u > 0:
            cmd = self._format_command('+', 0, b_u)
            state = self.step(state.rstrip('^'), cmd)
            trajectory.append({'command': cmd, 'response': state})

        if b_t > 0:
            cmd = self._format_command('+', 1, b_t)
            state = self.step(state.rstrip('^'), cmd)
            trajectory.append({'command': cmd, 'response': state})

        return trajectory


if __name__ == '__main__':
    test_cases = [
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
        sim = AbacusSimulator(mode=mode)
        print(f'\n--- mode={mode} ---')
        for A, B, C in test_cases:
            traj = sim.gold_trajectory(A, B)
            final = sim.state_to_int(traj[-1]['response'].rstrip('^'))
            status = 'ok' if final == C else 'FAIL'
            steps = ' '.join(
                f"{t['command']}>{t['response']}" if t['command'] else t['response']
                for t in traj
            )
            print(f'{A:3d} + {B:3d} = {C:3d}  [{status}]  {steps}')
