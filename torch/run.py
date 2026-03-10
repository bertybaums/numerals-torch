#!/usr/bin/env python3
"""
Autoresearch experiment runner for Numerals.
Runs training + formal evaluation and prints a standardized summary.

Usage:
  python run.py --mode sft --scaffold carry_explicit --max_steps 20000
  python run.py --mode sft --scaffold carry_explicit --max_steps 20000 > run.log 2>&1

All arguments are forwarded to train.py. The runner then automatically
evaluates the final checkpoint and prints a machine-readable summary block.
"""

import glob
import os
import re
import subprocess
import sys
import time


def parse_arg(args, name, default=None):
    """Extract a named argument from the arg list."""
    for i, a in enumerate(args):
        if a == f"--{name}" and i + 1 < len(args):
            return args[i + 1]
    return default


def main():
    args = sys.argv[1:]

    scaffold = parse_arg(args, "scaffold")
    model_size = parse_arg(args, "model_size", "small")
    max_len = parse_arg(args, "max_len", "64")
    max_steps = parse_arg(args, "max_steps", "20000")
    mode = parse_arg(args, "mode")

    if not scaffold or not mode:
        print("ERROR: --mode and --scaffold are required")
        sys.exit(1)

    t_start = time.time()

    # ── Training ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    train_result = subprocess.run([sys.executable, "train.py"] + args)
    if train_result.returncode != 0:
        print("\n---")
        print("status:           crash")
        sys.exit(1)

    t_train = time.time() - t_start

    # ── Find checkpoint ───────────────────────────────────────────────────────
    ckpt_pattern = f"checkpoints/*_{scaffold}_{model_size}_step{max_steps}.pt"
    ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)
    if not ckpts:
        print(f"\nNo checkpoint found matching {ckpt_pattern}")
        print("\n---")
        print("status:           crash")
        sys.exit(1)

    ckpt_path = ckpts[-1]

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    eval_result = subprocess.run(
        [
            sys.executable, "evaluate.py",
            "--ckpt", ckpt_path,
            "--scaffold", scaffold,
            "--model_size", model_size,
            "--max_len", max_len,
        ],
        capture_output=True,
        text=True,
    )

    eval_output = eval_result.stdout + eval_result.stderr
    print(eval_output)

    # Parse overall accuracy
    accuracy = 0.0
    for line in eval_output.split("\n"):
        if line.strip().startswith("Overall"):
            m = re.search(r"(\d+\.?\d*)%", line)
            if m:
                accuracy = float(m.group(1))

    t_total = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print("---")
    print(f"accuracy:         {accuracy:.1f}")
    print(f"mode:             {mode}")
    print(f"scaffold:         {scaffold}")
    print(f"model_size:       {model_size}")
    print(f"max_steps:        {max_steps}")
    print(f"checkpoint:       {ckpt_path}")
    print(f"training_seconds: {t_train:.1f}")
    print(f"total_seconds:    {t_total:.1f}")


if __name__ == "__main__":
    main()
