# SP2(a) — Compositional command grammar (in-distribution)

Generated: 2026-04-28

Anchors: `examples/anchors.json` (scope: addition_1_to_100, 16 entries)

## Headline accuracy

| Condition | N | Overall | h+h | h+r | r+h | r+r |
|---|---:|---:|---:|---:|---:|---:|
| tiny_all_COMP | 7920 | 9.6% | 10.3% | 9.5% | 9.5% | 8.9% |
| tiny_hindu_COMP | 7920 | 28.9% | 98.9% | 0.4% | 16.2% | 0.0% |
| small_all_COMP | 7920 | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |
| small_hindu_COMP | 7920 | 29.2% | 100.0% | 1.7% | 14.9% | 0.3% |

## Status breakdown

| Condition | correct | invalid_command | wrong_digits | wrong_command_count | wrong_answer_on_gold_path | no_answer |
|---|---:|---:|---:|---:|---:|---:|
| tiny_all_COMP | 757 | 0 | 4736 | 174 | 2253 | 0 |
| tiny_hindu_COMP | 2287 | 1344 | 626 | 796 | 142 | 2725 |
| small_all_COMP | 7919 | 0 | 0 | 0 | 1 | 0 |
| small_hindu_COMP | 2315 | 573 | 854 | 2058 | 164 | 1956 |

## Anchor problems

Each row is one anchor; each cell shows model status across the 4 notation pairs (hindu+hindu, hindu+roman, roman+hindu, roman+roman). ✓ = correct final answer; ✗ = wrong; ? = anchor not present in JSONL.

| Anchor | A+B=C | Category | tiny_all_COMP | tiny_hindu_COMP | small_all_COMP | small_hindu_COMP |
|---|---|---|---|---|---|---|
| single_digit_no_carry | 3+4=7 | single_digit_no_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| boundary_carry | 6+4=10 | single_digit_with_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| single_digit_max_carry | 9+9=18 | single_digit_with_carry | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| mixed_count_small | 2+10=12 | mixed_digit_count | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| no_carry_two_digit | 11+12=23 | two_digit_no_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✓ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✓ ✗ |
| no_carry_max | 76+23=99 | two_digit_no_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| units_carry_round | 13+17=30 | two_digit_units_carry | ✗ ✓ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| units_carry_mid | 29+68=97 | two_digit_units_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| tens_to_hundreds_min | 21+80=101 | tens_to_hundreds_carry | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| tens_to_hundreds_large | 98+91=189 | tens_to_hundreds_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| double_carry_clean_100 | 14+86=100 | double_carry | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| double_carry_38_72 | 38+72=110 | double_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| double_carry_42_79 | 42+79=121 | double_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| double_carry_67_73 | 67+73=140 | double_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| double_carry_max | 99+95=194 | double_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |
| A_equals_100 | 100+50=150 | A_eq_100_edge | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ | ✓ ✓ ✓ ✓ | ✓ ✗ ✗ ✗ |

## Failure gallery

Up to 3 representative misses per (condition, status). Format: `A+B=C (notation): predicted=P cmds=[...] gold=[...]`. When the trace ends at a non-matching final state, the simulator's view of the rod register is shown after `→ state=`.

### tiny_all_COMP

**wrong_digits** (4736 total)

- `38+72=110` (h+h): predicted=**100** cmds=[+02 +18] gold=[+02 +17] → state=[1|2|0]^
- `38+72=110` (h+r): predicted=**102** cmds=[+02 +18] gold=[+02 +17] → state=[1|2|0]^
- `38+72=110` (r+h): predicted=**103** cmds=[+02 +18] gold=[+02 +17] → state=[1|2|0]^

**wrong_command_count** (174 total)

- `24+6=30` (r+h): predicted=**89** cmds=[+05 +16] gold=[+06] → state=[0|8|9]
- `8+5=13` (h+h): predicted=**94** cmds=[+05 +15] gold=[+05] → state=[0|6|3]
- `8+5=13` (r+h): predicted=**61** cmds=[+05 +15] gold=[+05] → state=[0|6|3]

**wrong_answer_on_gold_path** (2253 total)

- `73+62=135` (h+h): predicted=**114** cmds=[+02 +16] gold=[+02 +16] → state=[1|3|5]^
- `73+62=135` (h+r): predicted=**149** cmds=[+02 +16] gold=[+02 +16] → state=[1|3|5]^
- `73+62=135` (r+h): predicted=**149** cmds=[+02 +16] gold=[+02 +16] → state=[1|3|5]^

### tiny_hindu_COMP

**invalid_command** (1344 total)

- `67+73=140` (h+r): predicted=**-1** cmds=[+ +] gold=[+03 +17] → state=[0|6|7]
- `67+73=140` (r+h): predicted=**-1** cmds=[+7] gold=[+03 +17] → state=[0|6|7]
- `73+62=135` (r+h): predicted=**-1** cmds=[+86] gold=[+02 +16] → state=[0|7|3]

**wrong_digits** (626 total)

- `8+61=69` (h+r): predicted=**28** cmds=[+11 +11] gold=[+01 +16] → state=[0|2|8]
- `76+10=86` (h+r): predicted=**84** cmds=[+04] gold=[+11] → state=[0|8|0]^
- `60+25=85` (h+r): predicted=**104** cmds=[+04 +14] gold=[+05 +12] → state=[1|0|4]^

**wrong_command_count** (796 total)

- `38+72=110` (r+r): predicted=**84** cmds=[(none)] gold=[+02 +17] → state=[0|3|8]
- `73+62=135` (r+r): predicted=**1** cmds=[(none)] gold=[+02 +16] → state=[0|7|3]
- `24+6=30` (h+r): predicted=**104** cmds=[+19 +19] gold=[+06] → state=[2|0|4]^

**wrong_answer_on_gold_path** (142 total)

- `41+57=98` (r+h): predicted=**94** cmds=[+07 +15] gold=[+07 +15] → state=[0|9|8]
- `21+67=88` (r+h): predicted=**81** cmds=[+07 +16] gold=[+07 +16] → state=[0|8|8]
- `41+81=122` (r+h): predicted=**123** cmds=[+01 +18] gold=[+01 +18] → state=[1|2|2]^

**no_answer** (2725 total)

- `38+72=110` (h+r): predicted=**-1** cmds=[(none)] gold=[+02 +17] → state=[0|3|8]
- `38+72=110` (r+h): predicted=**-1** cmds=[(none)] gold=[+02 +17] → state=[0|3|8]
- `67+73=140` (r+r): predicted=**-1** cmds=[(none)] gold=[+03 +17] → state=[0|6|7]

### small_all_COMP

**wrong_answer_on_gold_path** (1 total)

- `8+1=9` (r+h): predicted=**99** cmds=[+01] gold=[+01] → state=[0|0|9]

### small_hindu_COMP

**invalid_command** (573 total)

- `38+72=110` (r+h): predicted=**-1** cmds=[155] gold=[+02 +17] → state=[0|3|8]
- `8+61=69` (r+h): predicted=**-1** cmds=[+06 161] gold=[+01 +16] → state=[0|1|4]^
- `38+80=118` (r+h): predicted=**-1** cmds=[155] gold=[+18] → state=[0|3|8]

**wrong_digits** (854 total)

- `8+61=69` (h+r): predicted=**63** cmds=[+05 +15] gold=[+01 +16] → state=[0|6|3]
- `76+10=86` (h+r): predicted=**82** cmds=[+06] gold=[+11] → state=[0|8|2]^
- `76+10=86` (r+h): predicted=**54** cmds=[+15] gold=[+11] → state=[1|2|6]^

**wrong_command_count** (2058 total)

- `38+72=110` (r+r): predicted=**8** cmds=[(none)] gold=[+02 +17] → state=[0|3|8]
- `67+73=140` (r+r): predicted=**7** cmds=[(none)] gold=[+03 +17] → state=[0|6|7]
- `73+62=135` (r+r): predicted=**3** cmds=[(none)] gold=[+02 +16] → state=[0|7|3]

**wrong_answer_on_gold_path** (164 total)

- `33+30=63` (r+h): predicted=**3** cmds=[+13] gold=[+13] → state=[0|6|3]
- `50+59=109` (r+h): predicted=**99** cmds=[+09 +15] gold=[+09 +15] → state=[1|0|9]^
- `48+50=98` (r+h): predicted=**8** cmds=[+15] gold=[+15] → state=[0|9|8]

**no_answer** (1956 total)

- `38+72=110` (h+r): predicted=**-1** cmds=[+15] gold=[+02 +17] → state=[0|8|8]
- `67+73=140` (h+r): predicted=**-1** cmds=[+16] gold=[+03 +17] → state=[1|2|7]^
- `67+73=140` (r+h): predicted=**-1** cmds=[+05] gold=[+03 +17] → state=[0|7|2]^

