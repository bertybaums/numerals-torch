# SP2(b) — Rod-index generalization (OOD probe)

Generated: 2026-04-28

Anchors: `examples/anchors.json` (scope: addition_1_to_100, 16 entries)

## Headline accuracy

| Condition | N | Overall | h+h | h+r | r+h | r+r |
|---|---:|---:|---:|---:|---:|---:|
| tiny_all_COMP | 2000 | 0.1% | 0.0% | 0.2% | 0.0% | 0.0% |
| tiny_hindu_COMP | 2000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| small_all_COMP | 2000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| small_hindu_COMP | 2000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Status breakdown

| Condition | correct | invalid_command | wrong_digits | wrong_command_count | wrong_answer_on_gold_path | no_answer |
|---|---:|---:|---:|---:|---:|---:|
| tiny_all_COMP | 1 | 0 | 178 | 1821 | 0 | 0 |
| tiny_hindu_COMP | 0 | 437 | 37 | 878 | 0 | 648 |
| small_all_COMP | 0 | 0 | 85 | 1915 | 0 | 0 |
| small_hindu_COMP | 0 | 124 | 41 | 1546 | 0 | 289 |

## Anchor problems

_(Anchor set does not overlap this test set — section suppressed. Anchors are scoped to the standard test distribution; OOD evals should add their own anchor set if longitudinal comparison is desired.)_

## Failure gallery

Up to 4 representative misses per (condition, status). Format: `A+B=C (notation): predicted=P cmds=[...] gold=[...]`. When the trace ends at a non-matching final state, the simulator's view of the rod register is shown after `→ state=`.

### tiny_all_COMP

**wrong_digits** (178 total)

- `36+350=386` (h+h): predicted=**66** cmds=[+11 +12] gold=[+15 +23] → state=[0|6|6]
- `36+350=386` (h+r): predicted=**116** cmds=[+15 +15] gold=[+15 +23] → state=[1|3|6]^
- `36+350=386` (r+r): predicted=**145** cmds=[+15 +15] gold=[+15 +23] → state=[1|3|6]^
- `95+204=299` (h+r): predicted=**109** cmds=[+05 +19] gold=[+04 +22] → state=[1|9|0]

**wrong_command_count** (1821 total)

- `82+214=296` (h+h): predicted=**117** cmds=[+04 +13] gold=[+04 +11 +22] → state=[1|1|6]^
- `82+214=296` (h+r): predicted=**117** cmds=[+05 +15] gold=[+04 +11 +22] → state=[1|3|7]^
- `82+214=296` (r+h): predicted=**117** cmds=[+05 +13] gold=[+04 +11 +22] → state=[1|1|7]^
- `82+214=296` (r+r): predicted=**144** cmds=[+05 +15] gold=[+04 +11 +22] → state=[1|3|7]^

### tiny_hindu_COMP

**invalid_command** (437 total)

- `82+214=296` (r+h): predicted=**-1** cmds=[765] gold=[+04 +11 +22] → state=[0|8|2]
- `82+214=296` (r+r): predicted=**-1** cmds=[+1PAD] gold=[+04 +11 +22] → state=[0|8|2]
- `4+859=863` (h+r): predicted=**-1** cmds=[+1PAD] gold=[+09 +15 +28] → state=[0|0|4]
- `4+859=863` (r+r): predicted=**-1** cmds=[+14 +1PAD] gold=[+09 +15 +28] → state=[0|4|4]

**wrong_digits** (37 total)

- `36+350=386` (h+h): predicted=**96** cmds=[+13 +13] gold=[+15 +23] → state=[0|9|6]
- `5+130=135` (h+r): predicted=**49** cmds=[+04 +14] gold=[+13 +21] → state=[0|4|9]
- `5+130=135` (r+r): predicted=**49** cmds=[+04 +14] gold=[+13 +21] → state=[0|4|9]
- `41+510=551` (h+h): predicted=**141** cmds=[+15 +15] gold=[+11 +25] → state=[1|4|1]^

**wrong_command_count** (878 total)

- `82+214=296` (h+h): predicted=**106** cmds=[+03 +12] gold=[+04 +11 +22] → state=[1|0|5]^
- `4+859=863` (h+h): predicted=**93** cmds=[+09 +18] gold=[+09 +15 +28] → state=[0|9|3]
- `29+242=271` (r+r): predicted=**1** cmds=[(none)] gold=[+02 +14 +22] → state=[0|2|9]
- `87+858=945` (h+r): predicted=**75** cmds=[+18] gold=[+08 +15 +28] → state=[1|6|7]^

**no_answer** (648 total)

- `82+214=296` (h+r): predicted=**-1** cmds=[(none)] gold=[+04 +11 +22] → state=[0|8|2]
- `4+859=863` (r+h): predicted=**-1** cmds=[+07] gold=[+09 +15 +28] → state=[0|1|1]^
- `36+350=386` (h+r): predicted=**-1** cmds=[+04] gold=[+15 +23] → state=[0|4|0]^
- `36+350=386` (r+h): predicted=**-1** cmds=[(none)] gold=[+15 +23] → state=[0|3|6]

### small_all_COMP

**wrong_digits** (85 total)

- `95+204=299` (h+r): predicted=**189** cmds=[+04 +19] gold=[+04 +22] → state=[1|8|9]^
- `95+204=299` (r+r): predicted=**189** cmds=[+04 +19] gold=[+04 +22] → state=[1|8|9]^
- `72+303=375` (h+r): predicted=**165** cmds=[+03 +19] gold=[+03 +23] → state=[1|6|5]^
- `72+303=375` (r+r): predicted=**165** cmds=[+03 +19] gold=[+03 +23] → state=[1|6|5]^

**wrong_command_count** (1915 total)

- `82+214=296` (h+h): predicted=**106** cmds=[+04 +12] gold=[+04 +11 +22] → state=[1|0|6]^
- `82+214=296` (h+r): predicted=**176** cmds=[+04 +19] gold=[+04 +11 +22] → state=[1|7|6]^
- `82+214=296` (r+h): predicted=**106** cmds=[+04 +12] gold=[+04 +11 +22] → state=[1|0|6]^
- `82+214=296` (r+r): predicted=**176** cmds=[+04 +19] gold=[+04 +11 +22] → state=[1|7|6]^

### small_hindu_COMP

**invalid_command** (124 total)

- `29+242=271` (r+h): predicted=**-1** cmds=[+02 +14 170] gold=[+02 +14 +22] → state=[0|7|1]
- `95+204=299` (r+h): predicted=**-1** cmds=[+12 101] gold=[+04 +22] → state=[1|1|5]^
- `70+189=259` (r+h): predicted=**-1** cmds=[+08 +18 186] gold=[+09 +18 +21] → state=[1|5|8]^
- `21+814=835` (r+h): predicted=**-1** cmds=[+09 193] gold=[+04 +11 +28] → state=[0|3|0]^

**wrong_digits** (41 total)

- `4+859=863` (h+h): predicted=**103** cmds=[+09 +15 +15] gold=[+09 +15 +28] → state=[1|1|3]^
- `5+130=135` (h+h): predicted=**55** cmds=[+12 +13] gold=[+13 +21] → state=[0|5|5]
- `5+130=135` (r+h): predicted=**55** cmds=[+12 +13] gold=[+13 +21] → state=[0|5|5]
- `6+847=853` (h+h): predicted=**103** cmds=[+07 +14 +15] gold=[+07 +14 +28] → state=[1|0|3]^

**wrong_command_count** (1546 total)

- `82+214=296` (h+h): predicted=**96** cmds=[+04 +11] gold=[+04 +11 +22] → state=[0|9|6]
- `82+214=296` (h+r): predicted=**533** cmds=[+15] gold=[+04 +11 +22] → state=[1|3|2]^
- `82+214=296` (r+h): predicted=**4** cmds=[(none)] gold=[+04 +11 +22] → state=[0|8|2]
- `82+214=296` (r+r): predicted=**25** cmds=[(none)] gold=[+04 +11 +22] → state=[0|8|2]

**no_answer** (289 total)

- `36+350=386` (r+r): predicted=**-1** cmds=[+15] gold=[+15 +23] → state=[0|8|6]
- `95+204=299` (h+r): predicted=**-1** cmds=[+05] gold=[+04 +22] → state=[1|0|0]^
- `95+204=299` (r+r): predicted=**-1** cmds=[+15] gold=[+04 +22] → state=[1|4|5]^
- `76+532=608` (r+h): predicted=**-1** cmds=[+15] gold=[+02 +13 +25] → state=[1|2|6]^

