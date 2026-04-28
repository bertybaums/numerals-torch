#!/usr/bin/env python3
"""
Build a markdown examples table from one or more eval JSONLs.

Usage:
  python examples/build_examples_table.py \
      --title "SP1 Capacity Floor" \
      --output examples/SP1_capacity_floor.md \
      --condition tiny_all=logs/tooluse_sft_sp1_tiny_all_step80000_eval.jsonl \
      --condition tiny_hindu=logs/tooluse_sft_sp1_tiny_hindu_step80000_eval.jsonl

Each --condition is LABEL=PATH; the label appears in tables. The script reads
examples/anchors.json (alongside this file) for the longitudinal anchor set.

Output structure:
  1. Headline accuracy table (overall + 4 notation pairs per condition)
  2. Status breakdown (counts per error category per condition)
  3. Anchor problem matrix (✓/✗ per anchor × notation × condition)
  4. Failure gallery (representative misses per condition × status)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import Counter, defaultdict


# Order in which we report notation pairs in tables/galleries.
NOTATION_PAIRS = [('hindu', 'hindu'), ('hindu', 'roman'),
                  ('roman', 'hindu'), ('roman', 'roman')]
NOTATION_LABELS = ['h+h', 'h+r', 'r+h', 'r+r']

STATUS_ORDER = [
    'correct',
    'invalid_command',
    'wrong_digits',
    'wrong_command_count',
    'wrong_answer_on_gold_path',
    'no_answer',
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Return a list of row dicts from a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_anchors(path=None):
    """Read examples/anchors.json. Defaults to file alongside this script."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, 'anchors.json')
    with open(path) as f:
        return json.load(f)


# ── Aggregations ─────────────────────────────────────────────────────────────

def index_by_problem(rows):
    """Index rows by (A, B, notation_A, notation_B). Each value is one row."""
    idx = {}
    for r in rows:
        key = (r['A'], r['B'], r['notation_A'], r['notation_B'])
        idx[key] = r
    return idx


def per_pair_accuracy(rows):
    """Return dict {(notation_A, notation_B): (correct, total)}."""
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r['notation_A'], r['notation_B'])
        counts[key][1] += 1
        if r['correct']:
            counts[key][0] += 1
    return {k: tuple(v) for k, v in counts.items()}


def overall_accuracy(rows):
    correct = sum(1 for r in rows if r['correct'])
    return correct, len(rows)


def status_counts(rows):
    return Counter(r['status'] for r in rows)


# ── Markdown rendering ───────────────────────────────────────────────────────

def fmt_pct(num, denom):
    if denom == 0:
        return '–'
    return f"{100 * num / denom:.1f}%"


def render_headline_table(conditions):
    """Per-condition overall + per-notation accuracy."""
    lines = ['## Headline accuracy', '']
    header = '| Condition | N | Overall | ' + ' | '.join(NOTATION_LABELS) + ' |'
    sep = '|---|---:|---:|' + '---:|' * len(NOTATION_LABELS)
    lines.append(header)
    lines.append(sep)
    for label, rows in conditions:
        c, n = overall_accuracy(rows)
        ppa = per_pair_accuracy(rows)
        cells = []
        for pair in NOTATION_PAIRS:
            pc, pt = ppa.get(pair, (0, 0))
            cells.append(fmt_pct(pc, pt))
        lines.append(f"| {label} | {n} | {fmt_pct(c, n)} | " + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def render_status_table(conditions):
    """Per-condition error-category breakdown."""
    lines = ['## Status breakdown', '']
    header = '| Condition | ' + ' | '.join(STATUS_ORDER) + ' |'
    sep = '|---|' + '---:|' * len(STATUS_ORDER)
    lines.append(header)
    lines.append(sep)
    for label, rows in conditions:
        sc = status_counts(rows)
        cells = [str(sc.get(s, 0)) for s in STATUS_ORDER]
        lines.append(f"| {label} | " + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def cell_glyph(row):
    """One glyph for an anchor cell. ✓ = correct, ✗ = wrong, ? = missing."""
    if row is None:
        return '?'
    return '✓' if row['correct'] else '✗'


def render_anchor_matrix(conditions, anchors):
    """Per-anchor result matrix across all conditions × notation pairs.

    If no anchor problem appears in any condition's JSONL (the test set
    doesn't overlap the anchor set — common for OOD evals), the section is
    suppressed entirely with a one-line note.
    """
    indexed = [(label, index_by_problem(rows)) for label, rows in conditions]

    # First pass: are any anchors actually present anywhere?
    any_present = False
    for a in anchors['anchors']:
        A, B = a['A'], a['B']
        for _label, idx in indexed:
            for (nA, nB) in NOTATION_PAIRS:
                if (A, B, nA, nB) in idx:
                    any_present = True
                    break
            if any_present:
                break
        if any_present:
            break

    if not any_present:
        return ['## Anchor problems', '',
                "_(Anchor set does not overlap this test set — section suppressed. "
                "Anchors are scoped to the standard test distribution; OOD evals "
                "should add their own anchor set if longitudinal comparison is desired.)_",
                '']

    lines = ['## Anchor problems', '',
             "Each row is one anchor; each cell shows model status across the 4 "
             "notation pairs (hindu+hindu, hindu+roman, roman+hindu, roman+roman). "
             "✓ = correct final answer; ✗ = wrong; ? = anchor not present in JSONL.",
             '']

    cond_labels = [label for label, _ in conditions]
    header_top = '| Anchor | A+B=C | Category |'
    header_sep = '|---|---|---|'
    for label in cond_labels:
        header_top += f' {label} |'
        header_sep += '---|'
    lines.append(header_top)
    lines.append(header_sep)

    for a in anchors['anchors']:
        A, B, C = a['A'], a['B'], a['C']
        cat = a['category']
        row_cells = [a['id'], f"{A}+{B}={C}", cat]
        for _label, idx in indexed:
            glyphs = []
            for (nA, nB) in NOTATION_PAIRS:
                row = idx.get((A, B, nA, nB))
                glyphs.append(cell_glyph(row))
            row_cells.append(' '.join(glyphs))
        lines.append('| ' + ' | '.join(row_cells) + ' |')
    lines.append('')
    return lines


def render_failure_gallery(conditions, k_per_status=3):
    """Sample representative misses per (condition, status)."""
    lines = ['## Failure gallery',
             '',
             f"Up to {k_per_status} representative misses per (condition, status). "
             "Format: `A+B=C (notation): predicted=P cmds=[...] gold=[...]`. "
             "When the trace ends at a non-matching final state, the simulator's "
             "view of the rod register is shown after `→ state=`.",
             '']
    for label, rows in conditions:
        lines.append(f'### {label}')
        lines.append('')
        # Group by status
        by_status = defaultdict(list)
        for r in rows:
            if r['status'] == 'correct':
                continue
            by_status[r['status']].append(r)
        if not by_status:
            lines.append('_(no errors — all examples correct)_')
            lines.append('')
            continue
        for status in STATUS_ORDER:
            if status == 'correct':
                continue
            misses = by_status.get(status, [])
            if not misses:
                continue
            lines.append(f'**{status}** ({len(misses)} total)')
            lines.append('')
            for r in misses[:k_per_status]:
                tag = f"{r['notation_A'][0]}+{r['notation_B'][0]}"
                cmds = ' '.join(r['commands']) if r['commands'] else '(none)'
                gold = ' '.join(r['gold_commands']) if r['gold_commands'] else '(none)'
                state = r.get('final_state', '?')
                lines.append(
                    f"- `{r['A']}+{r['B']}={r['C']}` ({tag}): "
                    f"predicted=**{r['predicted']}** cmds=[{cmds}] "
                    f"gold=[{gold}] → state={state}"
                )
            lines.append('')
    return lines


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_conditions(arg_list):
    """Parse --condition LABEL=PATH entries into [(label, path), ...]."""
    out = []
    for entry in arg_list:
        if '=' not in entry:
            raise ValueError(f"--condition expects LABEL=PATH, got '{entry}'")
        label, path = entry.split('=', 1)
        out.append((label.strip(), path.strip()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--title',     required=True, help='Markdown H1 title')
    p.add_argument('--output',    required=True, help='Output markdown path')
    p.add_argument('--condition', action='append', required=True, default=[],
                   help='LABEL=PATH for one eval JSONL; repeat for multiple conditions')
    p.add_argument('--anchors',   default=None,
                   help='Path to anchors.json (default: examples/anchors.json next to this script)')
    p.add_argument('--k_per_status', type=int, default=3,
                   help='Failures to show per (condition, status) in the gallery')
    args = p.parse_args()

    cond_specs = parse_conditions(args.condition)
    conditions = []
    for label, path in cond_specs:
        if not os.path.isfile(path):
            print(f"  WARN: {path} not found — skipping condition '{label}'", file=sys.stderr)
            continue
        rows = load_jsonl(path)
        if not rows:
            print(f"  WARN: {path} empty — skipping condition '{label}'", file=sys.stderr)
            continue
        conditions.append((label, rows))

    if not conditions:
        sys.exit("No usable conditions; aborting.")

    anchors = load_anchors(args.anchors)

    today = _dt.date.today().isoformat()
    lines = [f'# {args.title}',
             '',
             f'Generated: {today}',
             '',
             f'Anchors: `{os.path.relpath(args.anchors) if args.anchors else "examples/anchors.json"}` '
             f'(scope: {anchors.get("scope", "?")}, {len(anchors["anchors"])} entries)',
             '']
    lines += render_headline_table(conditions)
    lines += render_status_table(conditions)
    lines += render_anchor_matrix(conditions, anchors)
    lines += render_failure_gallery(conditions, args.k_per_status)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
