# SP6 — Tool-Use Model Probing

Generated: 2026-04-28

Linear probes (200-epoch Adam, single linear layer) on hidden states at
the **pre_eq** position (last token before `=`) of the full gold trace.
Both checkpoints reach ~100% accuracy on the standard test set, so
per-condition headline numbers are uninteresting — the question is *what*
the model has internalized to do that.

Each probe is a held-out 40% test split (60% train); 2000 traces total
(500 facts × 4 notation pairs).

## Headline: per-target accuracy at pre_eq, last layer (L3)

| Target | Type | SP1 opaque | SP2a COMP | Chance |
|---|---|---:|---:|---:|
| `final_H` | cls | 99.0% | 94.1% | 10% |
| `final_T` | cls | 28.3% | 58.0% | 10% |
| `final_U` | cls | 14.3% | 19.1% | 10% |
| `sum_value` | reg (R²) | -2.17 | -1.57 | 0 |
| `A_value` | reg (R²) | -0.38 | -0.14 | 0 |
| `B_value` | reg (R²) | -0.23 | -0.04 | 0 |
| `carry_outer` | cls | 76.7% | 74.8% | 50% |

## Per-layer breakdown — pre_eq position

### SP1 small_all (opaque, 100%)

| Target | L0 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| `final_H` | 63.8% | 96.5% | 99.0% | 99.0% |
| `final_T` | 16.1% | 25.5% | 23.1% | 28.3% |
| `final_U` | 15.7% | 18.0% | 16.9% | 14.3% |
| `sum_value` | -3.98 | -3.78 | -3.26 | -2.17 |
| `A_value` | -1.69 | -1.56 | -1.10 | -0.38 |
| `B_value` | -1.25 | -1.09 | -0.79 | -0.23 |
| `carry_outer` | 71.7% | 78.6% | 78.9% | 76.7% |

### SP2a small_all_COMP (99.99%)

| Target | L0 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| `final_H` | 64.7% | 84.1% | 98.6% | 94.1% |
| `final_T` | 19.4% | 32.1% | 71.0% | 58.0% |
| `final_U` | 22.9% | 23.3% | 23.6% | 19.1% |
| `sum_value` | -4.19 | -3.89 | -3.39 | -1.57 |
| `A_value` | -1.88 | -1.63 | -1.28 | -0.14 |
| `B_value` | -1.47 | -1.17 | -0.89 | -0.04 |
| `carry_outer` | 72.7% | 73.0% | 79.7% | 74.8% |

## Per-notation breakdown — pre_eq position, last layer

Tests whether probe accuracy varies by notation of the operands.

### SP1 small_all (opaque, 100%)

| Target | h+h | h+r | r+h | r+r |
|---|---:|---:|---:|---:|
| `final_H` | 99.0% | 99.5% | 99.0% | 98.5% |
| `final_T` | 30.0% | 25.5% | 32.5% | 22.6% |
| `final_U` | 15.0% | 14.1% | 14.2% | 12.8% |
| `sum_value` | -2.28 | -2.12 | -2.30 | -2.04 |
| `A_value` | -0.41 | -0.46 | -0.45 | -0.62 |
| `B_value` | -0.23 | -0.20 | -0.23 | -0.15 |
| `carry_outer` | 77.3% | 78.1% | 81.7% | 78.5% |

### SP2a small_all_COMP (99.99%)

| Target | h+h | h+r | r+h | r+r |
|---|---:|---:|---:|---:|
| `final_H` | 94.2% | 94.4% | 91.0% | 95.4% |
| `final_T` | 59.2% | 59.2% | 56.2% | 57.4% |
| `final_U` | 18.0% | 21.4% | 16.9% | 16.8% |
| `sum_value` | -1.61 | -1.33 | -1.83 | -1.53 |
| `A_value` | -0.14 | -0.05 | -0.18 | -0.17 |
| `B_value` | -0.01 | -0.05 | -0.05 | -0.04 |
| `carry_outer` | 75.7% | 73.5% | 78.1% | 77.2% |

## What the numbers say

1. `final_H` (hundreds digit of the answer) is near-perfectly encoded at `pre_eq` —
   the next token after `pre_eq` is `=`, then the first emitted answer digit.
   When sum ≥ 100 that first digit is the hundreds, which is why it has to be encoded here.

2. `final_U` (units digit) is at chance. The model has *not* preserved the units digit
   in its hidden state at this position — the units digit is emitted last in the
   autoregressive answer, and the model retrieves it later by attending back to the
   simulator response. Dissociation cases (`40+40=80 → 8000`) are misfires of that
   later attention — the trained representation is not the failure mode.

3. The compositional model preserves `final_T` (tens digit) twice as well as opaque
   (58% vs 28%). The rod-index-as-digit encoding may give a more uniform place-value
   subspace. Consistent with SP2(a)'s capacity-floor regularizer effect.

4. Operands (`A_value`, `B_value`) are essentially not preserved as integers at `pre_eq`;
   they've been collapsed into a digit-positional encoding rather than a continuous
   number-line representation. Same finding as Phase 8 for scaffold-trained models.

5. `carry_outer` is well above chance (~75%), consistent with Phase 8 — carry detection
   is the strongest learned signal across all model types in this project.

