# SP1 — Capacity floor (tiny model)

Generated: 2026-04-27

Anchors: `examples/anchors.json` (scope: addition_1_to_100, 16 entries)

## Headline accuracy

| Condition | N | Overall | h+h | h+r | r+h | r+r |
|---|---:|---:|---:|---:|---:|---:|
| tiny_all | 7920 | 17.1% | 27.5% | 12.3% | 18.5% | 10.3% |
| tiny_hindu | 7920 | 25.6% | 97.1% | 0.4% | 5.1% | 0.0% |

## Status breakdown

| Condition | correct | invalid_command | wrong_digits | wrong_command_count | wrong_answer_on_gold_path | no_answer |
|---|---:|---:|---:|---:|---:|---:|
| tiny_all | 1357 | 0 | 5435 | 561 | 567 | 0 |
| tiny_hindu | 2031 | 2239 | 69 | 23 | 95 | 3463 |

## Anchor problems

Each row is one anchor; each cell shows model status across the 4 notation pairs (hindu+hindu, hindu+roman, roman+hindu, roman+roman). ✓ = correct final answer; ✗ = wrong; ? = anchor not present in JSONL.

| Anchor | A+B=C | Category | tiny_all | tiny_hindu |
|---|---|---|---|---|
| single_digit_no_carry | 3+4=7 | single_digit_no_carry | ✗ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ |
| boundary_carry | 6+4=10 | single_digit_with_carry | ✓ ✗ ✗ ✓ | ✓ ✗ ✗ ✗ |
| single_digit_max_carry | 9+9=18 | single_digit_with_carry | ✓ ✗ ✗ ✗ | ✗ ✗ ✗ ✗ |
| mixed_count_small | 2+10=12 | mixed_digit_count | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ |
| no_carry_two_digit | 11+12=23 | two_digit_no_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ |
| no_carry_max | 76+23=99 | two_digit_no_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ |
| units_carry_round | 13+17=30 | two_digit_units_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ |
| units_carry_mid | 29+68=97 | two_digit_units_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ |
| tens_to_hundreds_min | 21+80=101 | tens_to_hundreds_carry | ✓ ✗ ✓ ✗ | ✓ ✗ ✗ ✗ |
| tens_to_hundreds_large | 98+91=189 | tens_to_hundreds_carry | ✗ ✓ ✗ ✗ | ✓ ✗ ✗ ✗ |
| double_carry_clean_100 | 14+86=100 | double_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ |
| double_carry_38_72 | 38+72=110 | double_carry | ✗ ✗ ✗ ✗ | ✓ ✗ ✗ ✗ |
| double_carry_42_79 | 42+79=121 | double_carry | ✓ ✗ ✓ ✗ | ✓ ✗ ✗ ✗ |
| double_carry_67_73 | 67+73=140 | double_carry | ✗ ✗ ✗ ✓ | ✓ ✗ ✗ ✗ |
| double_carry_max | 99+95=194 | double_carry | ✗ ✓ ✗ ✗ | ✓ ✗ ✗ ✗ |
| A_equals_100 | 100+50=150 | A_eq_100_edge | ✗ ✓ ✗ ✗ | ✗ ✗ ✗ ✗ |

## Failure gallery

Up to 3 representative misses per (condition, status). Format: `A+B=C (notation): predicted=P cmds=[...] gold=[...]`. When the trace ends at a non-matching final state, the simulator's view of the rod register is shown after `→ state=`.

### tiny_all

**wrong_digits** (5435 total)

- `38+72=110` (h+h): predicted=**102** cmds=[+u4 +t6] gold=[+u2 +t7] → state=[1|0|2]^
- `38+72=110` (h+r): predicted=**99** cmds=[+u1 +t6] gold=[+u2 +t7] → state=[0|9|9]
- `38+72=110` (r+h): predicted=**118** cmds=[+u8 +t7] gold=[+u2 +t7] → state=[1|1|6]^

**wrong_command_count** (561 total)

- `38+80=118` (r+h): predicted=**117** cmds=[+u8 +t7] gold=[+t8] → state=[1|1|6]^
- `38+80=118` (r+r): predicted=**111** cmds=[+u3 +t7] gold=[+t8] → state=[1|1|1]^
- `76+10=86` (r+r): predicted=**87** cmds=[+u1 +t1] gold=[+t1] → state=[0|8|7]

**wrong_answer_on_gold_path** (567 total)

- `38+80=118` (h+h): predicted=**111** cmds=[+t8] gold=[+t8] → state=[1|1|8]^
- `24+6=30` (h+r): predicted=**20** cmds=[+u6] gold=[+u6] → state=[0|3|0]^
- `8+5=13` (h+h): predicted=**12** cmds=[+u5] gold=[+u5] → state=[0|1|3]^

### tiny_hindu

**invalid_command** (2239 total)

- `38+72=110` (h+r): predicted=**-1** cmds=[117] gold=[+u2 +t7] → state=[0|3|8]
- `38+72=110` (r+r): predicted=**-1** cmds=[655] gold=[+u2 +t7] → state=[0|3|8]
- `67+73=140` (h+r): predicted=**-1** cmds=[765] gold=[+u3 +t7] → state=[0|6|7]

**wrong_digits** (69 total)

- `15+1=16` (r+h): predicted=**17** cmds=[+u2] gold=[+u1] → state=[0|1|7]
- `98+10=108` (h+r): predicted=**103** cmds=[+u5] gold=[+t1] → state=[1|0|3]^
- `8+10=18` (h+r): predicted=**13** cmds=[+u5] gold=[+t1] → state=[0|1|3]^

**wrong_command_count** (23 total)

- `50+34=84` (r+h): predicted=**44** cmds=[+u4] gold=[+u4 +t3] → state=[0|5|4]
- `50+73=123` (r+h): predicted=**53** cmds=[+u3] gold=[+u3 +t7] → state=[0|5|3]
- `100+50=150` (h+h): predicted=**159** cmds=[+u9 +t5] gold=[+t5] → state=[1|5|9]

**wrong_answer_on_gold_path** (95 total)

- `40+40=80` (h+h): predicted=**8000** cmds=[+t4] gold=[+t4] → state=[0|8|0]
- `17+4=21` (h+h): predicted=**2121** cmds=[+u4] gold=[+u4] → state=[0|2|1]^
- `50+59=109` (r+h): predicted=**5** cmds=[+u9 +t5] gold=[+u9 +t5] → state=[1|0|9]^

**no_answer** (3463 total)

- `38+72=110` (r+h): predicted=**-1** cmds=[(none)] gold=[+u2 +t7] → state=[0|3|8]
- `67+73=140` (r+r): predicted=**-1** cmds=[(none)] gold=[+u3 +t7] → state=[0|6|7]
- `73+62=135` (h+r): predicted=**-1** cmds=[(none)] gold=[+u2 +t6] → state=[0|7|3]

