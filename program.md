# Numerals Autoresearch

Autonomous research loop for the Numerals project: studying how tiny transformer LMs learn arithmetic via chain-of-thought scaffolding.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `writeup.md` — full research documentation and results history
   - `torch/model.py` — MiniGPT transformer architecture (~13K or ~52K params)
   - `torch/data.py` — tokenizer, vocabularies, 6 scaffold formats
   - `torch/data_abacus.py` — abacus scaffold variants A–D
   - `torch/train.py` — unified training script (pretrain / SFT / RL)
   - `torch/evaluate.py` — **READ ONLY** — the fixed evaluation metric
   - `torch/run.py` — **READ ONLY** — experiment runner wrapper
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once confirmed, kick off experimentation.

## Experimentation

Each experiment trains + evaluates in one shot:

```bash
cd torch && python run.py --mode sft --scaffold carry_explicit --max_steps 20000 > run.log 2>&1
```

**What you CAN modify:**
- `torch/model.py` — architecture (layers, heads, embeddings, attention patterns)
- `torch/data.py` — scaffold formats, tokenization, data augmentation
- `torch/train.py` — optimizer, LR, schedule, training loop, RL algorithm, batch size

**What you CANNOT modify:**
- `torch/evaluate.py` — the fixed ground truth metric
- `torch/run.py` — the experiment runner wrapper
- `torch/data_abacus.py` — abacus data generator (you can use it but don't change it)
- Do not install new packages. Use only what's already importable (torch, standard library).

**The goal: maximize overall test accuracy.** Higher is better.

The current best result (from the writeup) is **92.1%** on carry_explicit scaffold, small model, 160K SFT steps. Your job is to beat that — or find insights about *why* certain approaches work.

**Step budget:** Default is 20,000 steps (~3–5 min on MPS, ~1–2 min on A6000). This gives enough signal to compare experiments directionally. Use more steps (40K–80K) only when validating a promising change at full convergence. Each formal evaluation adds ~1–2 min.

**Simplicity criterion**: All else equal, simpler is better. A small accuracy gain from ugly complexity isn't worth it. Removing code for equal or better results is a great outcome. Weigh complexity cost against improvement magnitude.

**The first run**: Always establish the baseline first — run the training script unmodified.

## Output format

The `run.py` wrapper prints a summary block:

```
---
accuracy:         79.4
mode:             sft
scaffold:         carry_explicit
model_size:       small
max_steps:        20000
checkpoint:       checkpoints/sft_carry_explicit_small_step20000.pt
training_seconds: 180.5
total_seconds:    210.3
```

Extract the key metric: `grep "^accuracy:" run.log`

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header and 5 columns:

```
commit	accuracy	steps	status	description
```

1. git commit hash (short, 7 chars)
2. accuracy achieved (e.g. 79.4) — use 0.0 for crashes
3. steps trained (e.g. 20000)
4. status: `keep`, `discard`, or `crash`
5. short text description of the experiment

Example:

```
commit	accuracy	steps	status	description
a1b2c3d	52.3	20000	keep	baseline carry_explicit SFT 20K
b2c3d4e	55.1	20000	keep	increase lr to 1e-3
c3d4e5f	49.8	20000	discard	switch to GeLU
d4e5f6g	0.0	20000	crash	double model width (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at git state: current branch/commit
2. Modify code with an experimental idea
3. git commit
4. Run: `cd torch && python run.py --mode sft --scaffold carry_explicit --max_steps 20000 > run.log 2>&1`
5. Read results: `grep "^accuracy:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the traceback and attempt a fix. If you can't fix it after a few attempts, give up on that idea.
7. Record results in results.tsv (do NOT commit results.tsv — leave it untracked)
8. If accuracy improved (higher), keep the commit and advance the branch
9. If accuracy is equal or worse, `git reset` back to where you started

**Timeout**: If a run exceeds 15 minutes, kill it and treat as failure.

**Crashes**: Use judgment — fix typos/imports and re-run; skip fundamentally broken ideas.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Do NOT ask "should I keep going?" The human may be away and expects you to work *indefinitely* until manually stopped. If you run out of ideas, re-read `writeup.md`, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.

## Research context

**What's known** (from writeup.md):
- Scaffold design dominates: carry_explicit (92.1%) >> decomp (42.3%) >> state_seq (14.8%) for the 13K model
- Carry-explicit works because it eliminates the carry bottleneck (units sum shown explicitly)
- The large model (52K) reaches 99.5% on state_seq — capacity solves the problem
- RL adds little when SFT demonstrations are complete
- {0, +1} rewards required — {-1, +1} creates zero-gradient traps
- SFT data mixing is needed to prevent RL format collapse

**Promising directions to explore:**

*Architecture (model.py):*
- Attention pattern changes (head count, head_dim trade-offs at fixed param budget)
- Positional encoding alternatives (relative, rotary, ALiBi)
- Layer norm type/placement (RMSNorm, pre vs post)
- Embedding dimension vs depth trade-offs
- Output head modifications

*Training (train.py):*
- Learning rate / schedule (warmup, cosine, cyclic)
- Optimizer (SGD+momentum, learning rate per parameter group)
- Curriculum learning (easy additions first, gradually harder)
- Data augmentation (more notation mixing, expression ordering)
- Batch size effects
- Gradient accumulation

*Scaffolds (data.py):*
- New scaffold formats exposing more structure
- Shorter scaffolds (fewer tokens = more capacity budget for the answer)
- Hybrid scaffolds combining best elements of carry_explicit + decomp
- Right-to-left processing (units first, like how humans do it)

*RL (train.py):*
- RL fine-tuning on carry_explicit (never tried — the scaffold is strong enough that RL might help push past 92%)
- Better reward shaping (partial credit for scaffold steps)
- PPO instead of REINFORCE

*Cross-cutting:*
- Weight initialization strategies
- Regularization (dropout, weight decay tuning)
- Ensemble or checkpoint averaging
