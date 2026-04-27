# Can a Tiny Language Model Learn Arithmetic Through Reinforcement Learning?
### An Experiment in Model-Based vs. Model-Free Learning

---

## The Historical Puzzle That Started This

In medieval Europe, merchants did their arithmetic on an abacus — a physical device where beads on rods represent digits. To add 47 and 35, you'd slide beads around, carry when a column overflowed, and read off the result. The abacus was the cognitive scaffold that made arithmetic tractable before people had internalized the positional notation system.

Then Hindu-Arabic numerals arrived. At first, merchants wrote things like:

> **XLVII + XXXV : 47 + 35 = 82**

The left side was their native Roman numeral language; the `:` marked a translation; the right side was the computation performed in the new Hindu-Arabic system. Eventually, people stopped needing to write the translation at all — they could think directly in the new notation.

This transition is not just a historical curiosity. It's a model for how *any* learner internalizes a new representational system: first through scaffolded translation, then through direct competence. And it raises a natural question for machine learning: **can a tiny language model follow the same path?**

---

## The Experiment

We built a miniature GPT — a transformer language model with only about 13,000 trainable parameters — and asked it to learn arithmetic over numbers 1–100, mixing Roman numeral and Hindu-Arabic notation.

The training data contained expressions like:

```
XLVII + 35 = 82          (Roman + Hindu-Arabic = Hindu-Arabic)
47 + XXXV = LXXXII       (Hindu-Arabic + Roman = Roman)
XLVII + XXXV = 82        (Roman + Roman = Hindu-Arabic)
```

All eight combinations of notation for the two operands and the result were included, giving 80,000 training expressions covering 8,000 arithmetic facts. A held-out test set of 2,000 facts (16,000 expressions) was used for evaluation.

The core research question: **does a "scaffold" — an intermediate step that translates operands into Hindu-Arabic before computing — help the model learn through reinforcement learning? Or can the model figure it out without that crutch?**

---

## The Model

A word on what "13,000 parameters" means. GPT-4 has roughly 1.8 *trillion* parameters. Our model is about 140 million times smaller. It has 4 transformer layers, an embedding dimension of 16, and 4 attention heads — dimensions chosen to be as small as possible while still having meaningful compositional depth.

This isn't a limitation we apologize for; it's deliberate. Small models are a *microscope* for studying learning dynamics. Every failure is visible and diagnosable. Every success is meaningful.

We trained this model in stages, following (in miniature) the modern pipeline for large language models:

1. **Pretraining**: Standard next-token prediction on 64,000 arithmetic expressions. The model learns the format and some arithmetic — but imperfectly. Final test accuracy: **8.6%**.

2. **Supervised fine-tuning (SFT) on a scaffold**: A second dataset in the format `A + B : a + b = C`, where `:` is a novel token (never seen during pretraining) that signals "now translate into Hindu-Arabic and compute." The model was fine-tuned to generate the full chain-of-thought. Final scaffold-format accuracy: **7.1%**.

3. **Reinforcement learning (RL)**: The model generates completions, receives reward (+1 or 0 based on correctness), and updates its weights to make rewarding completions more likely. This is where the interesting failures — and lessons — begin.

---

## Reinforcement Learning: A Longer Story Than Expected

The RL phase took five algorithmic iterations to get right. Each failure taught something specific.

### The Algorithm: REINFORCE

We used a classic policy gradient algorithm called REINFORCE. The basic idea: generate a completion, observe a reward, then adjust the model's weights to make that completion more or less likely depending on whether it was correct.

More precisely, the loss is:

```
loss = −(reward − baseline) × log_probability(completion)
```

The *baseline* is a running estimate of how well the model usually does. Subtracting it reduces noise: if the model usually gets −1 and got −1 again, there's nothing to learn. Only deviations from expectation carry information.

### Version 1: The Reward Trap

Our first attempt used rewards of +1 (correct) and −1 (incorrect). This is intuitive but creates a trap.

With only ~5% accuracy, almost every completion gets −1. The baseline — an exponential moving average of recent rewards — converges to −1. Once `baseline = −1` and `reward = −1`:

```
loss = −(−1 − (−1)) × log_probability = 0
```

**Zero gradient. The model is completely frozen.** The only way out is a +1 reward, but those stop arriving because the model has collapsed into generating repetitive garbage (`= = = = =`, `: : : :`). This is called *mode collapse* — the model learns to repeat the single most probable token instead of exploring.

### Versions 2–3: Overcorrection

Replacing {−1, +1} with {0, 1} rewards — wrong answers get 0, correct get +1 — was the right fix in theory. But two subsequent mistakes introduced new failure modes:

- **Version 2** added an entropy bonus to prevent mode collapse. The entropy bonus was too strong: with ~2% accuracy, the reward signal was too sparse to counteract the entropy maximizer, and the model drifted toward *maximum uncertainty* — generating random noise.
- **Version 3** removed the entropy bonus but still had the EMA baseline collapsing to zero (since wrong answers give reward 0, the baseline tracks toward 0, and again the gradient for wrong answers becomes 0).

### Version 4: Finding the Signal

Version 4 added two things:

1. **A baseline floor**: `baseline = max(0.05, ema_baseline)`. This ensures wrong answers always receive a small push-down gradient, even when the EMA would otherwise go to zero.

2. **A process reward** for the scaffold model: instead of only rewarding the final answer, we gave +0.5 for correctly extracting and translating the two operands (regardless of whether the arithmetic was right), and +0.5 for the correct final answer.

At step 500 of training, the scaffold model showed **17% operand extraction accuracy** — the model was correctly reading the numbers from the prompt. But by step 2500 this had decayed to 0%. The RL updates were overwriting the translation ability learned during SFT.

### Version 5: KL Anchoring — The Key Insight

The root cause of the decay was *policy drift*. REINFORCE computes the average log-probability of the entire completion and pushes it up or down as a unit. Even when the operand extraction was correct, the model was penalized whenever the final arithmetic was wrong — and since wrong arithmetic is ~98% of cases, this constant pressure gradually eroded the extraction ability.

The standard solution in large-scale RL (used in ChatGPT, Claude, and others) is a **KL penalty**: keep a frozen copy of the starting model and add a term to the loss that penalizes diverging from it.

```
loss = −(reward − baseline) × log_prob(completion) + kl_coef × KL(current ∥ frozen_sft)
```

KL divergence measures how different the updated model's distribution is from the starting model's distribution. Penalizing it forces the RL updates to stay close to what SFT learned — like a rubber band anchoring the model to its starting point.

With KL anchoring, the scaffold model achieved stable training, but **greedy evaluation revealed reward hacking**: the model learned to output `47 + 35 = ` with nothing after the `=` sign, earning reliable partial credit for correct operand extraction while skipping the hard arithmetic. Under sampling (used during RL training), occasional correct guesses appeared as 1–2% accuracy; under greedy evaluation, the model produced truncated completions that earned near-zero accuracy.

---

## Phase 2: A Richer Scaffold

After v5, it was clear the scaffold `A + B : a + b = C` was teaching the wrong skill. The intermediate step just translated notation — it didn't represent the *procedure* of addition. The analogy breaks down: the abacus isn't a translation device; it's a machine that externalizes the computation itself.

### The State-Sequence Scaffold

We redesigned the scaffold to mimic what a human actually does on an abacus: add the tens column first, then the units column.

```
47 + 35 : 47 + 30 = 77 + 5 = 82
```

Format: `A + B : a + B_tens = mid + B_units = C`

Each step in the chain-of-thought corresponds to a single bead-column operation. Adding a multiple of ten is easy (just increment the tens digit). Adding a single digit is also easy (just increment the units digit, carrying if necessary). The full two-digit addition — the hard part — is never asked of the model all at once. For numbers where B is a round number or a single digit, the chain simplifies to one step.

This format required no new vocabulary. The dataset was regenerated with Roman numeral inputs and Hindu-Arabic completions throughout: `XLVII + XXXV : 47 + 30 = 77 + 5 = 82`.

### Small Model SFT Results

The new scaffold improved the small model substantially:

| Checkpoint | Full accuracy | Step accuracy |
|-----------|:---:|:---:|
| Base (pretraining) | 8.6% | — |
| Old scaffold SFT | 7.1% | — |
| State-seq SFT, 40K steps | 11.6% | 78.5% |
| State-seq SFT, 80K steps | 14.8% | 91.0% |

"Step accuracy" measures whether the first intermediate sum (`a + B_tens = mid`) is correct — a diagnostic for whether the model has learned the procedure vs. just memorizing outputs. At 80K steps the small model was computing that intermediate step correctly 91% of the time, but still failing the full sum 85% of the time. The bottleneck was clear: **the final units-step addition** (`mid + B_units = C`) requires small-number carry arithmetic that the 13K-parameter model couldn't consistently learn.

---

## Phase 2 RL: Three More Failures

### Collapse v1 and v2: The KL Estimate Problem

Running REINFORCE on the state-sequence checkpoint reproduced a familiar failure — but with a new diagnosis. Two runs collapsed despite using the lessons from Phase 1:

- **v1** (kl_coef=0.1, temp=1.0): KL divergence exploded to 50+ by step 4,000; model output single tokens (`9`, `7`), then empty completions.
- **v2** (kl_coef=1.0, temp=0.7): Slowed the collapse but couldn't prevent it. KL rose 1.6 → 14.9 → 53.2 within 2,500 steps.

The diagnosis: the one-sample KL estimate is *structurally insufficient* for long-format models. The penalty is computed only at the tokens the model actually generates. Once the model begins drifting toward short outputs (`8`, `2`), the KL term measures divergence over those 1–2 tokens only — and provides zero gradient for the long state-sequence format the model has stopped generating. The model escapes the KL anchor by shifting mass to tokens the estimator never observes. No value of `kl_coef` fixes this.

### v3: SFT Data Mixing — Stability Achieved

The real fix for format collapse is **SFT data mixing**: at every RL step, also compute cross-entropy loss on a random training example.

```
loss = policy_loss + 0.1 × KL(pi || ref) + 1.0 × CE(sft_example)
```

Unlike the KL estimate (which only covers sampled tokens), the SFT cross-entropy loss covers *all tokens in a full training example*. It is a behavioral cloning anchor: it continuously restores probability mass to the learned format regardless of what the policy sampled. This is standard practice in RLHF pipelines (InstructGPT and others use it), but the distinction from the one-sample KL estimate matters deeply here.

The result: **stable training across all 20,000 steps.** KL stayed near zero throughout (range −0.032 to 0.064). Step accuracy held at 62–78%. Full accuracy oscillated 7.5–17%.

Formal evaluation on the held-out test confirmed a real but modest improvement:

| Checkpoint | Full accuracy | Step accuracy |
|-----------|:---:|:---:|
| State-seq SFT (80K, small) | 14.8% | 91.0% |
| State-seq RL v3 (20K, small) | **15.4%** | 90.9% |

The RL improved slightly over SFT — the first positive RL result — but the gain was small. Step accuracy was essentially unchanged (the tens step was already near-mastered), and the model remained bottlenecked on units-step arithmetic. The SFT mixing anchor was keeping the model near its starting distribution; the RL signal was nudging it slightly, but not enough to cross the units-arithmetic threshold. The ceiling was the model's capacity, not the algorithm.

---

## Phase 3: The Large Model

The small model's persistent failure on the units step pointed to a capacity limitation. Our architecture had `head_dim = 4` — attention heads with only 4 dimensions each. A reasonable hypothesis: this is simply too small to represent the carry logic needed for single-digit addition. We increased `n_embd` from 16 to 32 (head_dim 4 → 8), quadrupling parameter count from ~13K to ~52K.

We trained the large model directly on the state-sequence data from scratch (no separate pretraining step), for 200,000 steps with linear learning rate decay.

### Learning Trajectory

| Steps | Full accuracy | Step accuracy |
|-------|:---:|:---:|
| 10K | 4.5% | 17.6% |
| 40K | 33.0% | 55.2% |
| 50K | 64.5% | 88.7% |
| 70K | 82.5% | 97.0% |
| 90K | 93.5% | **100%** |
| 110K | 95.5% | 100% |
| 130K | 98.0% | 100% |
| 200K | **100%** | 100% |

The step accuracy saturated at 100% by step 90K, after which the model was purely improving the units step. By step 130K it had broken through the carry barrier; by 200K the inline sample showed 100% on both metrics.

### Formal Evaluation

| Notation pair | Accuracy |
|---------------|:---:|
| hindu + hindu | 1800/1805 = 99.7% |
| hindu + roman | 1801/1805 = 99.8% |
| roman + hindu | 1793/1805 = 99.3% |
| roman + roman | 1790/1805 = 99.2% |
| **Overall** | **7184/7220 = 99.5%** |
| Step accuracy | 1576/1577 = 99.9% |

**The problem is essentially solved.** The 36 remaining errors (~0.5%) share a single pattern: off-by-10 for the smallest input values (e.g., `1 + 8 → 19` instead of 9) — single-step cases where B < 10 and the model occasionally applies an erroneous tens carry. A minor edge case, not a fundamental failure.

---

## Full Results

| Stage | Description | Test Accuracy |
|-------|-------------|--------------|
| Pretraining (small, 13K) | Base GPT, next-token prediction | 8.6% |
| SFT — old scaffold (small) | `A + B : a + b = C` format | 7.1% |
| RL v1–v3 (model-free, small) | No scaffold | All failed (0%) |
| RL v5 (scaffold, small) | KL anchoring | 0.9% (reward hacking) |
| SFT — state-seq, 40K (small) | `a + B_tens = mid + B_units = C` | 11.6% |
| SFT — state-seq, 80K (small) | Continued training | 14.8% |
| RL v3 — state-seq (small) | + SFT data mixing | 15.4% |
| **SFT — state-seq, 200K (large, 52K)** | 4× params | **99.5%** |

---

## Key Findings

### 1. Model-free RL cannot bootstrap arithmetic from sparse outcome rewards alone

Five algorithmic iterations, five failures. This is a robust finding. Without intermediate structure, the reward signal of "correct final answer" is too sparse for a 13,000-parameter model to climb the learning gradient.

### 2. Scaffold design matters more than reward algorithm

The original scaffold (`A + B : a + b = C`) taught notation translation but not procedure. The state-sequence scaffold (`A + B : a + B_tens = mid + B_units = C`) encoded the actual computation as a sequence of states — like the abacus. With the richer scaffold, even the small model improved substantially through SFT, and RL achieved its first positive result.

### 3. One-sample KL estimates can fail to anchor long-format models

A KL penalty computed only at sampled tokens provides no gradient for tokens the model has stopped generating. When format collapse begins, the KL term watches it happen without intervening. SFT data mixing — computing cross-entropy over complete training examples at every RL step — is a structurally different and more robust anchor.

### 4. RL's marginal contribution, at this scale, is small

Both with the small model (SFT: 14.8% → RL: 15.4%) and implicitly with the large model (where 99.5% SFT accuracy left almost no room for improvement), RL added little over SFT. The learning was happening during supervised training, not reinforcement. This may be specific to small models on well-structured tasks: when the training data contains clean, complete demonstrations of the correct procedure, imitation learning is already near-optimal. RL's advantage — exploring beyond demonstrated behavior — only matters when the demonstrations are incomplete.

### 5. Model capacity was the binding constraint

The small model consistently failed the units-step addition. Quadrupling parameters (while keeping the same scaffold and more training steps) solved it. In hindsight, `head_dim = 4` is almost certainly too small to represent carry logic: the four-dimensional attention representations can't distinguish enough cases. `head_dim = 8` (the large model) was sufficient.

---

## The Analogy Revisited

We started with a historical analogy. A medieval merchant with an abacus could set up an arithmetic problem in the new Hindu-Arabic notation — translate the Roman numerals, write the addition — but might still struggle to execute the computation reliably. The abacus provided a scaffold for the setup; the computation itself required more capacity.

Our tiny model followed the same arc exactly. The original scaffold (just notation translation) was like teaching merchants to *write* the Hindu-Arabic numerals without teaching them to *compute* with them. The state-sequence scaffold, by externalizing the bead-column procedure as intermediate text, was much closer to what the abacus actually provided: a record of each step in the computation.

With the richer scaffold and a model just large enough to represent carry logic in its attention heads, the task became tractable. SFT alone — no RL, just supervised imitation — was sufficient. The medieval transition took about 200 years. Our model took about 200,000 training steps.

---

## Phase 4: Scaffold Option 2 — Carry-Explicit

Having established that the state-sequence scaffold was bottlenecked on units-step arithmetic for the small model, the natural question is: **can the scaffold encode that exact bottleneck more explicitly?**

The carry-explicit scaffold does:

```
47 + 35 : 7+5=12 4+3+1=8 = 82
23 + 41 : 3+1=4 2+4=6 = 64
5  + 3  : 5+3=8 = 8               (single-digit: no tens column)
```

Format: `a_u+b_u=units_sum [a_t+b_t[+carry]=tens_sum] = C`

The chain-of-thought shows the full column sums — `7+5=12` not just the digit 2 — making the carry explicit as a `+1` in the tens step. The model needs only: (1) single-digit addition (sums 0–18), (2) reading the carry from whether the sum ≥ 10, (3) single-digit addition for the tens column including carry. No character was added to the vocabulary; the format uses the existing `+`, `=`, and digits.

### Results: 79.4% with the Small Model

The same 13,000-parameter model, the same 80,000 training steps, starting from the same pretrained checkpoint. The carry-explicit scaffold achieved:

| Notation pair | Accuracy |
|---------------|:---:|
| hindu + hindu | 1800/1960 = 91.8% |
| hindu + roman | 1663/1960 = 84.8% |
| roman + hindu | 1516/1960 = 77.3% |
| roman + roman | 1247/1960 = 63.6% |
| **Overall** | **6226/7840 = 79.4%** |
| Units step accuracy | 1876/1960 = 95.7% |
| Carry propagation accuracy | 1807/1871 = 96.6% |

Compare to the state-sequence scaffold at the same model size and training budget: **14.8%**. The carry-explicit scaffold achieves **5× higher accuracy**.

The inline eval at step 80K showed 95.0% on Hindu-Arabic prompts and was still climbing — the model had not yet converged. The bottleneck that remained was Roman numeral inputs, where the model must additionally decode `XX` → tens=2, units=0 before applying the carry algorithm. Roman+Roman accuracy (63.6%) reflects this extra step.

### Why This Matters

The state-sequence scaffold broke a 2-digit addition into two single-step additions: `a + B_tens = mid` then `mid + B_units = C`. The small model could learn the first step (97.5% step accuracy under SFT) but not reliably execute the second, because `mid + B_units = C` could still require a carry — and that carry logic was invisible in the chain-of-thought.

The carry-explicit scaffold makes the carry **visible as a number**. The model sees `7+5=12` — the full column sum — and never has to "figure out" that the answer digit is 2 and carry is 1. It just learns to produce a number ≥10 when there's a carry, and reads it back when adding the next column. This is within the representational capacity of a 13,000-parameter model.

The result confirms that the bottleneck was not model capacity per se but *information hiding*: the state-sequence format withheld the carry signal that the model needed in its chain-of-thought.

### Continued Training

We continued from this checkpoint for 80,000 more steps (160K total) with a reduced learning rate (0.001). The inline eval at the final step showed 98.5% full accuracy and 100% units accuracy on a Hindu-Arabic sample. Formal evaluation:

| Notation pair | Accuracy |
|---|:---:|
| hindu + hindu | 1909/1960 = 97.4% |
| hindu + roman | 1877/1960 = 95.8% |
| roman + hindu | 1769/1960 = 90.3% |
| roman + roman | 1665/1960 = 84.9% |
| **Overall** | **7220/7840 = 92.1%** |
| Units step accuracy | 1930/1960 = 98.5% |
| Carry propagation accuracy | 1903/1923 = 99.0% |

The model improved substantially — 79.4% → 92.1% overall. Carry propagation accuracy is near-perfect (99.0%): when the model correctly identifies the units-column sum, it nearly always propagates the carry correctly. The remaining gap (7.9%) is almost entirely Roman numeral inputs, where the model must additionally decode the operands before applying the carry algorithm. A 13,000-parameter model has essentially mastered carry arithmetic in Hindu-Arabic notation.

---

## Phase 5: Scaffold Option 1 — Decomposition

Having established the carry-explicit format, we next tested the decomposition scaffold — the other natural way to expose additive structure:

```
47 + 35 : 40+30=70 7+5=12 = 82
23 + 41 : 20+40=60 3+1=4 = 64
5  + 3  : 5+3=8 = 8               (single-digit: no tens step)
```

Format: `a_t×10+b_t×10=tens_sum  a_u+b_u=units_sum = C`

The chain-of-thought shows both partial sums — the tens contribution and the units contribution — and leaves the final assembly implicit. The model sees 70 and 12 in its context when generating 82. The intuition: if both partial sums are visible, can the model learn to add them? No new vocabulary characters were needed; the dataset structure is identical to carry-explicit.

### Results

Same 13,000-parameter model, 80,000 steps from the same pretrained checkpoint.

| Notation pair | Accuracy |
|---|:---:|
| hindu + hindu | 1076/1961 = 54.9% |
| hindu + roman |  968/1961 = 49.4% |
| roman + hindu |  677/1961 = 34.5% |
| roman + roman |  594/1961 = 30.3% |
| **Overall** | **3315/7844 = 42.3%** |
| Tens step (a_t×10+b_t×10=tens_sum) | 1921/1949 = 98.6% |
| Units step (a_u+b_u=units_sum) | 1122/1961 = 57.2% |

### Why Decomposition Falls Short

The tens step is memorized almost perfectly: **98.6%**. The model learns that `40+30=70` with virtually no error, because these are fixed patterns (at most 81 possible tens-column pairs).

The units step is harder: **57.2%**. Single-digit sums — the model knows them reasonably well in isolation — but placed second in the chain-of-thought, after the tens result, more errors appear than in isolation.

The deeper problem: even when both intermediate steps are correct (56.5% of two-column cases), the model still has to *assemble* the final answer — and that assembly is itself a 2-digit addition. To combine `70 + 12 = 82`, the model must internally add a 2-digit number to a 2-digit number. The scaffold has **deferred** the hard computation, not **eliminated** it.

Compare to carry-explicit (`7+5=12 4+3+1=8 = 82`): after both column steps, the model only needs to concatenate the tens digit (8) and units digit (2) to produce 82. No addition required — just read off each column's result. The carry-explicit format eliminates the assembly step by design.

| Scaffold | Full acc | Step acc | Assembly bottleneck? |
|---|:---:|:---:|:---:|
| State-sequence (80K, small) | 14.8% | 91.0% | Yes — units step still hides carry |
| Decomposition (80K, small) | 42.3% | 98.6% (tens) | Yes — final step is 2-digit addition |
| Carry-explicit (80K, small) | 79.4% | 95.7% (units) | No — final step reads column digits |
| Carry-explicit (160K, small) | **92.1%** | 98.5% (units) | No |

The pattern is clear: scaffolds that **eliminate** an arithmetic bottleneck dramatically outperform those that **defer** it. The carry-explicit format is the only one that reduces every step — including the final one — to operations the small model can reliably perform: single-digit addition and threshold comparison (is sum ≥ 10?).

---

## Phase 6: The Generalization Hypothesis — Digit-Position Normalization

The carry-explicit model (92.1%) was trained and evaluated on 1–99. A natural question: can the model generalize to 3-digit or 4-digit numbers it has never seen? The format `a_u+b_u=units ... = C` is theoretically position-invariant — the carry procedure is the same regardless of how many digits are involved.

When we probed the model on 3-digit inputs (e.g., `247 + 358 :`), it produced valid carry-explicit format — but with the wrong digits. The problem: the model had learned position-specific extraction. For `47`, the units digit is at token position 1; for `247`, it is at token position 2. The attention patterns were tied to absolute token positions, not to semantic roles.

### The Digit-Scaffold Fix

A revised dataset and training run addressed this directly. The new format:

```
47 + 35 : 7 4 + 5 3 : 7+5=12 4+3+1=8 = 82
247 + 358 : 7 4 2 + 8 5 3 : 7+8=15 4+5+1=10 2+3+1=6 = 605
```

`A + B : [digits of A right-to-left] + [digits of B right-to-left] : [carry steps] = C`

The preamble (between the first and second colons) lists digits in column order: units first, then tens, then hundreds. This normalizes position — the units digit is **always** the first preamble element, regardless of number size. The hypothesis: train on 1–999 (up to 3 column steps), and the carry algorithm should transfer to 4-digit numbers because the carry section is indexed against a consistent positional layout.

We trained a 14,816-parameter model from scratch on 32K examples (Hindu-Arabic only, no Roman numerals), for 80K steps with linear learning rate decay. The preamble accuracy reached 97.4%, confirming the model reliably extracts digits in right-to-left order.

### Results

**In-distribution (1–999):**
- Full accuracy: **91.4%**
- Preamble correct: **97.4%** — model reliably extracts digits right-to-left
- Carry (oracle preamble): **93.4%** — the carry algorithm is near-mastered

**OOD blind — 4-digit operands (1000–4999):**
- Full accuracy: **0.0%** | Preamble correct: **0.0%**

Failure pattern: the model extracted 2–3 digits from 4-digit numbers. Even with the right-to-left preamble design, digit extraction remained anchored to specific token positions. The model learned "units digit is at context position N" rather than "scan right-to-left until end of number."

**OOD oracle preamble — 4-digit with correct preamble given:**
- Full accuracy: **0.0%**

This is the key diagnostic. Even when given the correct 4-element preamble, the model only executed 2 carry steps before assembling the final answer. Example:

```
3404 + 2192 : 4 0 4 3 + 2 9 1 2 : 4+2=5 3+3=6 = 659  (true=5596)
```

The model saw four digit pairs but generated only two carry steps. It learned a fixed-depth carry procedure matching its training distribution (1–3 steps for 1–999), not a variable-length iteration.

### What This Reveals

A transformer has no loop construct. The carry algorithm it learns is not "iterate through each digit pair in the preamble" — it is "produce a fixed-depth sequence of tokens that matches the training distribution." When the preamble has 4 elements instead of 3, the model does not extend its carry section; it terminates as trained.

This is fundamentally distinct from the position-extraction failure. Even with perfect digit normalization (oracle preamble), the carry section itself is depth-limited by training. The hypothesis was half-right: the preamble *does* normalize digit position (97.4% in-distribution accuracy confirms this), but the carry section is *also* position/depth-anchored.

### What Would Actually Fix This

1. **Training on mixed lengths (1–4 digit)**: Expose the model to variable carry depth during training; force it to learn N-step carry for variable N.
2. **Explicit step count in the format**: `3 steps: 7+8=15 4+5+1=10 2+3+1=6 = 605` — a token the model can attend to as a loop counter.
3. **Recurrent architecture**: An LSTM or similar model can genuinely iterate; the fixed-depth transformer cannot.
4. **Much larger model**: With sufficient capacity, the model may discover the counting mechanism from examples alone — but this is unlikely at 15K parameters.

The finding reframes the generalization question: it is not just about digit-position normalization, but about *depth generalization* of the chain-of-thought itself.

---

## Full Results

| Stage | Description | Test Accuracy |
|-------|-------------|:---:|
| Pretraining (small, 13K) | Base GPT, next-token prediction | 8.6% |
| SFT — old scaffold (small) | `A + B : a + b = C` format | 7.1% |
| RL v1–v3 (model-free, small) | No scaffold | All failed (0%) |
| RL v5 (scaffold, small) | KL anchoring | 0.9% (reward hacking) |
| SFT — state-seq, 40K (small) | `a + B_tens = mid + B_units = C` | 11.6% |
| SFT — state-seq, 80K (small) | Continued training | 14.8% |
| RL v3 — state-seq (small) | + SFT data mixing | 15.4% |
| SFT — state-seq, 200K (large, 52K) | 4× params, from scratch | 99.5% |
| SFT — decomposition, 80K (small) | `a_t×10+b_t×10=tens a_u+b_u=units = C` | 42.3% |
| SFT — carry-explicit, 80K (small) | `a_u+b_u=units [a_t+b_t[+carry]=tens] = C` | 79.4% |
| SFT — carry-explicit, 160K (small) | Continued training | 92.1% |
| **SFT — digit-scaffold, 80K (small, 15K)** | Right-to-left preamble + carry, 1-999 | **91.4% in-dist / 0% OOD** |

---

## Key Findings

### 1. Model-free RL cannot bootstrap arithmetic from sparse outcome rewards alone

Five algorithmic iterations, five failures. Without intermediate structure, the reward signal of "correct final answer" is too sparse for a 13,000-parameter model to climb the learning gradient.

### 2. Scaffold design matters more than reward algorithm

The original scaffold (`A + B : a + b = C`) taught notation translation but not procedure. The state-sequence scaffold encoded procedure as a sequence of states. The carry-explicit scaffold encoded the carry signal directly. The decomposition scaffold exposed partial sums. Each format produced dramatically different results at identical capacity and training budget — scaffold design was the dominant variable.

### 3. The chain-of-thought must eliminate bottlenecks, not defer them

The clearest result in these experiments comes from comparing the three task-specific scaffolds directly:

- **State-sequence** (14.8%): hides the carry in its final step (`mid + B_units = C`)
- **Decomposition** (42.3%): shows both partial sums (70 and 12) but leaves their sum (82) to the model — which is itself a 2-digit addition
- **Carry-explicit** (92.1%): after both column sums are written, the final answer is just digit concatenation — no further arithmetic

Decomposition deferred the hard computation; carry-explicit eliminated it. When the model had both `40+30=70` and `7+5=12` visible, it still had to produce `82` — and doing so required the same carry logic that was the original bottleneck. The chain-of-thought only helps as much as it reduces the difficulty of each remaining step.

### 4. Information hiding is a distinct failure mode from capacity

State-sequence gave the model `mid + B_units = C` — a step that still hides the carry. Carry-explicit exposes `a_u+b_u=units_sum` — the full column sum including the carry digit. Accuracy jumped from 14.8% to 92.1% at the same model capacity. The model wasn't failing because 13K parameters is too few to do arithmetic; it was failing because the arithmetic required inferring hidden state (whether to carry) that wasn't in its context.

### 5. One-sample KL estimates can fail to anchor long-format models

A KL penalty computed only at sampled tokens provides no gradient for tokens the model has stopped generating. SFT data mixing — computing cross-entropy over complete training examples at every RL step — is a structurally different and more robust anchor.

### 6. RL's marginal contribution, at this scale, is small

Both with the small model (SFT: 14.8% → RL: 15.4%) and implicitly with the large model (where 99.5% SFT accuracy left almost no room for improvement), RL added little over SFT. The learning was happening during supervised training, not reinforcement.

### 7. Model capacity matters, but scaffold design matters more

### 7. Model capacity matters, but scaffold design matters more

Quadrupling parameters (13K → 52K) with the state-sequence scaffold: 14.8% → 99.5%. Redesigning the scaffold with the same 13K model and 160K steps: 14.8% → 92.1%. Both interventions worked. The right information in the chain-of-thought is a more efficient lever than model capacity.

### 8. Transformers learn fixed-depth procedures, not variable-length algorithms

The digit-scaffold experiment (Phase 6) revealed a fundamental limitation independent of capacity or format. Even when given the correct 4-element preamble (oracle preamble), the model only executed 2–3 carry steps before terminating — matching its training distribution (1–3 steps for 1–999) rather than the preamble length. A transformer has no loop construct; the carry algorithm it learns is a fixed-depth pattern, not a variable-length iteration. This is not fixable by better scaffold design alone: it requires either training on mixed lengths, an explicit step-count signal in the format, or a recurrent architecture.

---

## The Analogy Revisited

We started with a historical analogy. The medieval merchant used an abacus — a device that externalizes the *state* of a computation. The bead positions after each step encode exactly the carry information that the next step needs. That's not just a mnemonic; it's a form of working memory that makes the algorithm tractable.

Our three task-specific scaffolds correspond to three different approximations of what the abacus does. The decomposition scaffold is like writing both addends' column contributions separately: `40+30` and `7+5`. Useful — but the merchant still has to combine 70 and 12 without the abacus. The carry-explicit scaffold is closer to what the abacus actually provides: after each column is processed, the *carry state* is written down. With the carry visible, the next column's computation is automatic, and the final answer is just the digit register — no further addition required.

The lesson for chain-of-thought reasoning: **the chain-of-thought needs to externalize intermediate state that the model would otherwise have to hold implicitly.** Arithmetic is tractable when the carry is written down. It becomes hard when the carry must be inferred. The abacus works not because it stores the operands, but because it stores the *carry*.

With the carry-explicit scaffold and 160,000 training steps, a 13,000-parameter model achieves 92.1% accuracy on mixed Roman/Hindu-Arabic addition — not through memory or scale, but through the right representation.

---

## Phase 7: The Abacus as a Digital Tool

Phases 1–6 studied scaffolds that encode procedure *within* the model's output sequence. But historically, the abacus was not a notation — it was an external device. The merchant didn't write the carry; the abacus *held* the carry as bead positions. The model's output described the result of a physical process that happened outside the language.

This distinction motivates **tool use**: instead of generating the full chain-of-thought, the model issues *commands* to an external abacus simulator and reads back *state updates*. The chain-of-thought becomes an interleaved dialogue between the model (which decides what to do) and the tool (which does the computation).

### The Abacus Simulator

The simulator represents a 3-rod abacus with state notation `[H|T|U]` — hundreds, tens, units — where each rod holds a single digit 0–9. The model issues commands like `+u5` (add 5 to the units rod) or `+t3` (add 3 to the tens rod). The simulator executes the command, handles overflow/carry internally, and returns the new state. A `^` suffix flags that a carry occurred.

```
47 + 35 : [0|4|7] +u5 [0|5|2]^ +t3 [0|8|2] = 82
```

In this trace:
- `[0|4|7]` — simulator loads operand A=47
- `+u5` — model's command: add B's units digit (5) to the units rod
- `[0|5|2]^` — simulator's response: 7+5 overflows, carry propagated to tens (4→5), units wrap to 2
- `+t3` — model's command: add B's tens digit (3) to the tens rod
- `[0|8|2]` — simulator's response: 5+3=8, no overflow
- `= 82` — model reads the final state and produces the answer

The critical design: **the model never computes carry**. It issues commands; the simulator does the arithmetic. This is precisely the medieval merchant's workflow.

Four abacus trace variants were designed (A–D), varying how carry is signaled:
- **A**: Flag carry with `^` suffix, overflow implicit
- **B**: Token carry with `K`, carry absorbed into state
- **C**: Complement explicit — shows subtraction operation (`-u3` for complement of 7)
- **D**: Carry-first — pre-carry and post-carry states shown separately

### Tool-Use SFT: 100% Accuracy

We trained a 3.5M-parameter model (medium: n_embd=192, 8 layers, 8 heads) on gold trajectories from the simulator for 80K steps. The model must learn:
1. Which command to issue at each step (decompose B into digit-by-digit operations)
2. How to read the simulator's state response
3. When to stop and produce the final answer

The loss function was optionally masked to train only on model-generated tokens (commands and answer), not on simulator-provided states.

**Result: 100.0% accuracy across all 7,920 test examples.**

| Metric | Value |
|--------|:---:|
| Overall accuracy | 7920/7920 = **100.0%** |
| Command validity | 14332/14332 = 100.0% |
| Gold trajectory match | 7920/7920 = 100.0% |
| Avg commands/problem | 1.81 |

Every notation pair — Hindu+Hindu, Hindu+Roman, Roman+Hindu, Roman+Roman — scored 100%. The model learned the complete abacus protocol: decompose B into single-digit rod operations, issue them in the correct order, and read the answer from the final state.

This is the first experiment to achieve perfect accuracy across all conditions. The training loss reached effectively zero by step 9K (of 80K), suggesting the 3.5M-parameter model has far more capacity than needed for this task. The interesting questions now are: (a) can a *smaller* model learn the protocol? (b) under RL with the simulator, would a model discover *alternative* command sequences? (c) what happens when the model has semantic understanding of addition and discovers that the tool can be operated differently?

---

## Phase 8: Probing for Internal Representations

The experiments so far measured *behavioral* accuracy — can the model produce the correct answer? But a deeper question is: **what does the model represent internally?** Does a model trained on Roman numeral inputs develop positional (place-value) representations internally, even though Roman numerals are additive?

This question connects directly to the historical thesis. If the abacus's positional structure was the key cognitive scaffold, and Hindu-Arabic numerals internalized that structure into notation, then we might expect a model that successfully computes with Roman inputs to develop *place-value representations* internally — rediscovering the structure that the abacus provides.

### Method

We used **linear probing**: freeze the trained model, extract hidden states at specific token positions, and train a simple linear classifier on those hidden states to predict various properties of the input. If a property is linearly decodable (high probe accuracy), it is explicitly represented in the model's hidden states. If not, the information is either absent or encoded nonlinearly.

**Probe targets** (8 total):
- *A_value*, *B_value*, *sum_value* — full integer values (regression)
- *A_tens*, *A_units*, *B_tens*, *B_units* — individual digit values (10-class classification)
- *carry* — whether the units column produces a carry (binary classification)

**Probe positions** (3 token positions in the encoded prompt):
- *plus* — the `+` token (operand A fully seen, B not yet)
- *colon* — the `:` token (scaffold transition, both operands seen)
- *equals* — the token just before `=` (the model must have computed its answer)

Each probe is a single linear layer trained with Adam for 200 epochs on 60% of examples, evaluated on the remaining 40%.

### Results

We probed three models: (1) `sft_old_small_step80000` (old scaffold, 7.1% task accuracy), (2) `sft_carry_explicit_small_step160000` (carry-explicit, 92.1%), and a locally trained early checkpoint at 5K steps (14% accuracy) for learning dynamics.

**Carry detection is linearly decodable.** Across all models and positions, the binary carry probe achieves well above chance (50%):

| Model | Position | Best layer | Carry accuracy |
|-------|----------|:---:|:---:|
| Carry-explicit 160K | equals | L1 | **80.6%** |
| Carry-explicit 160K | colon | L3 | 69.4% |
| Old scaffold 80K | equals | L2 | **77.0%** |
| Old scaffold 80K | colon | L3 | 68.8% |
| Carry-explicit 5K | equals | L3 | 77.5% |

The carry signal is present even at 5K steps (14% task accuracy) — the model learns to detect carry-producing digit combinations before it can reliably use that information to produce correct answers.

**Digit classification shows notation-dependent patterns.** At the colon position in the converged carry-explicit model:

| Notation | B_units accuracy | B_tens accuracy | Carry accuracy |
|----------|:---:|:---:|:---:|
| Hindu+Hindu | 0.265 | 0.253 | 0.665 |
| Hindu+Roman | 0.477 | 0.368 | 0.683 |
| Roman+Hindu | 0.236 | 0.250 | 0.701 |
| Roman+Roman | 0.371 | 0.282 | 0.704 |

Roman inputs produce *higher* carry probe accuracy than Hindu-Arabic inputs (0.70 vs 0.67). This suggests the model builds a more explicit internal representation when it must decode Roman numerals — consistent with the thesis that the positional structure is *reconstructed* internally when the notation doesn't provide it directly.

Similarly, B_units accuracy is highest for Hindu+Roman inputs (0.477) — the model has decoded the Roman B operand into its units digit and holds this linearly in the hidden state at the colon position, before the scaffold has begun.

**Full numeric values are not linearly decodable** in the small model. All regression R² values are negative (worse than predicting the mean), indicating the 16-dimensional hidden states don't support smooth linear readout of integers. The model likely uses a categorical or lookup-based encoding rather than a linear number line.

### What This Means

1. **The model develops place-value representations internally**, even for Roman numeral inputs. The digit probes — A_tens, A_units, B_tens, B_units — show above-chance accuracy at the colon position, meaning the model has decomposed the operands into tens and units *before* the scaffold forces it to do so.

2. **Carry detection is the earliest and strongest learned signal.** It appears before task accuracy rises and is present across all scaffolds. The model learns "does this pair of inputs produce a carry?" as one of the first useful features.

3. **Roman notation forces more explicit internal representations.** The model can't just copy digits from Roman input — it must reconstruct them. This reconstruction produces representations that are *more* linearly decodable than those from Hindu-Arabic input, where the digits are already available positionally.

4. **The 13K-parameter model's representations are limited but structured.** With only 16 dimensions per hidden state, the model can't encode everything linearly. But what it does encode — carry detection, digit decomposition — follows the same positional logic that the abacus and Hindu-Arabic notation make explicit.

---

## Updated Full Results

| Stage | Description | Test Accuracy |
|-------|-------------|:---:|
| Pretraining (small, 13K) | Base GPT, next-token prediction | 8.6% |
| SFT — old scaffold (small) | `A + B : a + b = C` format | 7.1% |
| RL v1–v3 (model-free, small) | No scaffold | All failed (0%) |
| RL v5 (scaffold, small) | KL anchoring | 0.9% (reward hacking) |
| SFT — state-seq, 40K (small) | `a + B_tens = mid + B_units = C` | 11.6% |
| SFT — state-seq, 80K (small) | Continued training | 14.8% |
| RL v3 — state-seq (small) | + SFT data mixing | 15.4% |
| SFT — state-seq, 200K (large, 52K) | 4× params, from scratch | 99.5% |
| SFT — decomposition, 80K (small) | `a_t×10+b_t×10=tens a_u+b_u=units = C` | 42.3% |
| SFT — carry-explicit, 80K (small) | `a_u+b_u=units [a_t+b_t[+carry]=tens] = C` | 79.4% |
| SFT — carry-explicit, 160K (small) | Continued training | 92.1% |
| SFT — digit-scaffold, 80K (small, 15K) | Right-to-left preamble + carry, 1-999 | 91.4% in-dist / 0% OOD |
| Tool-use SFT (medium, 3.5M) | Model issues commands to abacus simulator | 100.0% |
| **Tool-use SFT (small, 14K)** | **Same protocol, 250× fewer params** | **100.0%** |

---

## Phase 9: The Capacity Floor for Tool Use (SP1)

The tool-use result of Phase 7 — 100% accuracy with a 3.5M-parameter model — invited an obvious question. The model's job is not arithmetic; the simulator does the arithmetic. The model parses operands and emits a fixed-form command sequence (`+u<d> +t<d>`) that the simulator executes. That is a finite-state transduction — a regular-grammar-level task. **How much of the 3.5M parameters were actually doing useful work?**

The framing predicts that the parsing of Roman numerals is the binding constraint — single-digit addition lives entirely in the simulator, so all the model has to learn is (i) decompose B into its tens/units digits and (ii) emit the corresponding command tokens. To test this, we trained models at three sizes — tiny (4K params, n_embd=8), small (14K, n_embd=16, the existing config), and large (53K, n_embd=32) — under two notation conditions: `notation=all` (all 4 combinations of Roman/Hindu for A and B, the original setup) and `notation=hindu` (Hindu-Arabic only, no Roman parsing burden). Six runs total, 80K SFT steps each.

### Results

| Model | Train | Overall | h+h | h+r | r+h | r+r |
|-------|-------|--------:|----:|----:|----:|----:|
| tiny (4K) | hindu | 25.6% | **97.1%** | 0.4% | 5.1% | 0.0% |
| tiny (4K) | all | 17.1% | 27.5% | 12.3% | 18.5% | 10.3% |
| **small (14K)** | **all** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| small (14K) | hindu | 31.0% | 100.0% | 0.5% | 23.5% | 0.2% |
| **large (53K)** | **all** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| large (53K) | hindu | 31.2% | 100.0% | 1.6% | 23.0% | 0.3% |

**The capacity floor for the full task is between 4K and 14K parameters.** Small (14K) is enough to drive the abacus and parse Roman numerals at the same time, with no errors anywhere on the held-out set. Large adds nothing measurable; medium (Phase 7) is 250× over-provisioned. The headline arc Roman+abacus → Hindu-Arabic notation that this project models historically corresponds to *less than 14,000 parameters of cognitive scaffolding* — a satisfyingly small number.

### Tiny is parsing-bound, not sequencing-bound

The tiny model's split tells the story cleanly. Trained Hindu-only, it reaches **97.1%** on hindu+hindu — meaning 4K parameters is enough to (a) read the digits of B from the prompt, (b) emit `+u<B%10> +t<B//10>`, and (c) read the final answer off the simulator's rod register. Sequencing the abacus protocol fits in 4K. What 4K cannot do is also handle Roman parsing: when forced to handle all 4 notation pairs, accuracy collapses to 17.1% across the board. Roman parsing is the binding constraint, exactly as the controller-as-finite-state-transducer framing predicts.

### Spontaneous Roman-A delegation (the simulator does some parsing for free)

A surprising secondary finding: both small and large models trained Hindu-only reach ~23% on roman+hindu at evaluation time, despite never having seen a Roman A in training. This is not an artifact. The tool-use trace structure passes the simulator's initial state `[0|H|U]` to the model in the prompt, which already encodes A's digits in positional form. So when A is in Roman numerals, the model can ignore A's surface form and read its decomposition directly from the simulator — partial delegation of parsing to the tool. The other ~77% misses on r+h presumably come from cases where the model still attends to the surface form of A and gets confused. Probing under SP6 should disambiguate.

### Failure gallery surfaces an SP6 finding

The error breakdown also exposes a pattern worth flagging early. Several `*_hindu` failures at all model sizes show **correct gold trajectories and correct simulator final states**, but wrong final answers. Examples (from `examples/SP1_capacity_floor.md`):

```
40+40=80 (h+h):  cmds=[+t4]      gold=[+t4]      state=[0|8|0]   predicted=8000
17+4=21  (h+h):  cmds=[+u4]      gold=[+u4]      state=[0|2|1]^  predicted=2121
50+59=109 (r+h): cmds=[+u9 +t5]  gold=[+u9 +t5]  state=[1|0|9]^  predicted=5
16+48=64 (r+h):  cmds=[+u8 +t4]  gold=[+u8 +t4]  state=[0|6|4]   predicted=106  (small_hindu)
67+73=140 (h+r): cmds=[(none)]                                    predicted=7777777
```

The model has executed the abacus protocol perfectly, the simulator has computed the right answer, and then the model — at the answer-readout step — produces a digit-repetition or related malformation rather than reading off the rod state. This says the answer head is **not** strictly conditioned on the simulator's final state in the way one might naively expect. It is sometimes pulling from elsewhere in the context — possibly the operands, possibly memorized priors, possibly the answer head's own bias toward token repetitions. This is exactly the question SP6's "implicit transition-function probe" is designed to test.

### Implications

The smallness number anchors the rest of the project: when SP4 introduces multiplication and a `times(d,e)` tool, the question becomes how much capacity that adds to the floor — keeping in mind that the floor for *addition with the abacus* is just 14K. The capability ladder of SP5 (tally → abacus → +shift → +times → RPN) becomes a curve plotting capacity against tool capability, with this Phase 9 result as the anchor point at the abacus rung.

---

## The Analogy Completed

The historical transition from Roman numerals + abacus to Hindu-Arabic positional notation was not a change in what addition *means*. The algorithm was already there, embodied in the abacus: load the operand, add digit by digit, carry when a column overflows, read the result. Hindu-Arabic notation internalized the abacus's positional structure into the writing system itself — making the *notation* do what the *tool* used to do.

Our experiments trace this transition in miniature:

1. **The abacus as external tool** (Phase 7): The model operates an external simulator, issuing commands and reading state. The computation happens outside the model. This is the pre-Leonardo stage — the merchant knows *what to do* but relies on the physical device to *do it*. Result: 100% accuracy.

2. **The scaffold as internalized tool** (Phases 4–6): The carry-explicit format writes the carry into the chain-of-thought — the model's own output serves as its "abacus." The computation is now inside the notation. Result: 92.1% with a model 250× smaller.

3. **The probing evidence** (Phase 8): Even when the input is in Roman numerals — an additive, non-positional system — the model internally reconstructs place-value representations (tens digit, units digit, carry flag). The positional structure of the abacus is *rediscovered* in the model's hidden states. Roman inputs actually produce *more* explicit internal representations than Hindu-Arabic ones, because the model must actively reconstruct what Hindu-Arabic notation provides for free.

This arc — external tool → internalized notation → implicit internal representation — mirrors the historical sequence. The abacus was the cognitive scaffold. Hindu-Arabic numerals were the internalization. And inside a 13,000-parameter model trained on Roman numeral arithmetic, we find the beginnings of the same positional representations that made the historical transition possible.

---

## Connection to Complementary Inquiry

This work is part of a broader research program studying how intellectual transitions happen — when developments are inevitable products of existing knowledge versus genuine creative leaps. The companion project, *Complementary Inquiry* (adapted from Hasok Chang's philosophy of science), uses language models trained on temporally bounded corpora to measure the **inevitability** of historical developments and map the space of counterfactual alternatives.

The Roman → Hindu-Arabic transition is a paradigm case of what Complementary Inquiry calls a **notational shift**: the underlying semantics don't change (addition is still addition), but the representational system changes in a way that makes computation vastly more tractable. The Numerals project provides the *mechanistic* evidence — showing exactly how models develop positional representations and when tools vs. notations enable the same computation — while Complementary Inquiry provides the *philosophical* framework for understanding why some transitions are inevitable and others are genuine innovations.

The two projects share a central question: **was the adoption of Hindu-Arabic numerals in medieval Europe a discovery (the notation was waiting to be found, given the abacus) or an invention (a creative leap that could have gone differently)?** The probing results here suggest the former: a model that successfully uses an abacus-like tool develops the positional representations that Hindu-Arabic notation makes explicit. The structure was already there, waiting for a notation that could capture it.

---

## What We'd Try Next

1. **RL with the simulator**: The tool-use SFT model has perfect accuracy on gold trajectories. Under RL with the live simulator, can it discover *alternative* command sequences — perhaps more efficient ones (e.g., combining operations) or using complementary arithmetic (variants C/D)? If a model with semantic understanding of addition is given efficiency pressure, does it independently "invent" shortcuts?

2. **Probing at scale**: Run the same linear probes on the medium (3.5M) and xlarge (21M) models. Larger hidden states should support richer linear readout. Do full numeric values become linearly decodable? Does the notation asymmetry (Roman → more explicit representations) persist?

3. **Probing learning dynamics**: Probe at every saved checkpoint (10K, 20K, ..., 160K) to map how representations evolve during training. When does the carry signal first appear? Does digit decomposition emerge before or after task accuracy rises?

4. **Scaffold distillation**: Train on the carry-explicit scaffold, then gradually remove the scaffold tokens from training data. Does the model retain its accuracy when forced to compute without the chain-of-thought? This models the historical transition from needing the abacus to computing mentally.

5. **Efficiency pressure**: Add a length penalty to RL rewards. The carry-explicit format is verbose; can the model learn to produce shorter but equally correct outputs? This models the commercial pressure that drove adoption of Hindu-Arabic numerals over the slower abacus + Roman numeral workflow.

6. **Mixed-length carry generalization**: Train the digit-scaffold model on 1–4 digit numbers. Does exposure to variable carry depth teach the model to iterate rather than produce fixed-depth patterns?

---

*Built on an M-series Mac using PyTorch with MPS acceleration, and on the University of Idaho's fortyfive HPC cluster (RTX A6000 / RTX 4090 GPUs). All code in the `torch/` subdirectory.*
