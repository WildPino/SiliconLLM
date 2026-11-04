# Stage 2, part 1, test 3: the jump of the carried error at layer 6 of stage 1 (b). RESULT

2026-11-04. Protocol: `S2C_LAYER6_DIAGNOSIS_PROTOCOL_20261103.md`. Records: `benchmarks/stage2/results/s2c_layer6.json` and
`s2c_verdict.json`, from `s2c_verdict.py`. 26,702 held-out tokens, five arms, CPU, about 2.7 h.

## Outcome in one line

**THE JUMP IS A MASSIVE DIMENSION THE NORMS REMOVE.** Restoring the guilty expert cuts the raw carried error at layer 6
by 78%, moves the normalized error by 0.0002% and the head's KL by 0.7%.

## Arms

| Arm | Top-1 at the head | KL (nats) | Carried, layer 6, raw | Normalized, layer 6 | Normalized, layer 19 | Same 8 experts, layer 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tern | 0.768 | 0.555 | 30.44 | 0.0716 | 0.186 | 0.146 |
| ph (no training) | 0.632 | 1.197 | 25.02 | 0.1917 | 0.424 | 0.012 |
| fix5 (layer 5, expert 118 restored) | 0.766 | 0.551 | 6.65 | 0.0716 | 0.186 | 0.146 |
| fix15 (also layer 1, expert 13) | 0.764 | 0.571 | 0.13 | 0.0718 | 0.187 | 0.147 |

Controls: C1 holds (the planted expert and dimension are found) and C2 holds (jump ratio 4.436 against (b)'s 4.444).

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| label MASSIVE DIMENSION, 85% | **hit** |
| tern top-1 0.78-0.83, KL 0.40-0.55 | **miss, worse**: 0.768, 0.555 |
| fix15 KL within 10% of tern | **hit**: +2.8% |
| route_same at layer 10, 0.15-0.30 | **miss, worse**: 0.146 |
| C1, C2 pass | **hit** |

## Read here, not registered

- **fix15 removes almost all the raw error and makes KL slightly worse.** The raw error at layer 6 goes from 30.4 to 0.13,
  and the head does not notice.
- **The training's gain is in the normalized error and the routing.** From ph to tern, the normalized error at layer 6
  falls from 0.192 to 0.072, and routing agreement at layer 10 rises from 0.012 to 0.146.
- **Routing agrees for only 15% of tokens at layer 10.** Those are the same 8 experts in the same set. Most tokens run
  with at least one different expert long before the head.

## What the result changes

- Keeping the massive-activation rows out of ternary is not a lever. Test 3b does not take the guilty experts as its arm:
  it takes the expert with the highest routing weight.
- Part 2's training and logs, if it runs, read the error in normalized space, and its loss sees the routing.

## Thesis line

The test moves no quality number, as the protocol said. It removes one cheap candidate for buying back tern's −8.52.

## Parts specific to Ling

As the protocol's: the experts and dimensions named, and the sigmoid router with group selection. Massive activations
and the normalized reading are not specific.
