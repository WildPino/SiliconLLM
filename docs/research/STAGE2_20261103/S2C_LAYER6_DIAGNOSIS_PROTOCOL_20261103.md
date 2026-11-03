# Stage 2, part 1, test 3: the jump of the carried error at layer 6 of stage 1 (b). PROTOCOL

2026-11-03. A diagnosis on the CPU, before the experts are trained again (part 2).

## Question

Stage 1 (b) logged the mean square of the error carried into each layer of the ternary student (tern): 7.5 at layers
2-5, 33.5 at layer 6, 66.5 at layer 19 (`../STAGE1_20261030/S1D_TERNARY_EXPERTS_RESULT_20261103.md`). What makes the
jump, and does it carry the quality loss?

## Thesis line (rule 4)

Part 2 has to buy back most of tern's −8.52 points.
- If the jump comes from a few rows and carries the loss, keeping those rows out of ternary costs kilobytes and changes
  the design.
- If it does not carry the loss, the training has to fix an error that is spread, and the error in the raw residual
  stream is the wrong thing to watch.

The test moves no quality number by itself.

## Literature (read in the papers)

| Source | What it gives here | Known ways to fail |
| --- | --- | --- |
| Sun et al., Massive Activations in LLMs, arXiv 2402.17762, §2-3, Table 3 | a few residual dimensions carry values orders of magnitude above the rest; they appear abruptly after one layer (LLaMA2-7B layer 2) and stay nearly constant; set to their mean, perplexity is unchanged (5.47 → 5.47); set to zero, it is infinite | they act as biases: an error in their magnitude reads large in raw mean squares while the model barely depends on the exact value |
| Yu et al., The Super Weight in LLMs, arXiv 2411.07191, §3, Table 1 | one down_proj weight in an early layer creates the massive activation; pruning it takes Llama-7B's zero-shot average from 70.11 to 35.14; round-to-nearest with larger blocks degrades unless that weight is held out | finding it needs the activation spikes at down_proj's input and output |

## Found before registration (two dry runs, 3,158 and 1,313 held-out tokens)

- **Layer 5:** 99.9% of tern's own error at layer 5 comes from one expert, 118, and 78% of it lies in one dimension,
  848. The teacher's mean square at layer 5's output goes from 61 to 151: layer 5 writes a massive activation.
  - The rows: that expert's down_proj row 848 has weights up to 1.61, and its two group scales are 0.89. (b)'s outlier
    rule did not keep them at their max.
- **Layer 1:** 99.5% of the own error comes from expert 13, 96% in dimension 1149. The super weight's row is kept at
  its max (2.515625).
- **Restoring expert 118 of layer 5** (fix5, 1,313 tokens):
  - the raw carried error at layer 6 goes from 51.8 to 11.1;
  - in normalized space it does not move: 0.0691 against 0.0691;
  - at the head, KL goes from 0.452 to 0.446 and top-1 from 0.800 to 0.800.
- **Normalized space:** the error that the later norms read grows smoothly, 0.025, 0.042, 0.051, 0.060, 0.069 at
  layers 2-6. It shows no jump.
- **Routing:** from layer 2 on, only 44% of tokens, and from layer 5 on about 20%, pick the same 8 experts at the
  student's input as at the teacher's.
- **The raw jump ratio:** carried at layer 6 over layer 5 is 4.44 and 4.51 in the dry runs, against (b)'s 4.44 over
  1.58 M tokens. The absolute values move with the token sample (7.5 in (b) against 8.9 and 11.5 here), because a few
  tokens hold the massive activations.

## Design (`benchmarks/stage2/s2c_layer6.py`, CPU, fp32)

- **Streams:** the teacher (the original bf16 weights, in fp32) and a student. Both run over the first 25,000 of stage
  1's held-out tokens, layers 0-19, then the final norm and the head.
- **Per layer:**
  - the carried and output errors, raw and in normalized space (x / rms(x), what each later RMSNorm reads before its
    gain);
  - at layers 1-7: the own error (the student's layer on the teacher's input) and the carried error through the
    original layer;
  - the own error's split by expert (shares that add to 1) and by dimension;
  - the down_proj row of the top expert against (b)'s outlier rule;
  - the fraction of tokens with the same 8 experts.
- **At the head:** top-1 agreement and KL(teacher ‖ student) in nats.

| Arm | Student |
| --- | --- |
| plant | the teacher with one down_proj row multiplied by 3 at layer 3: the most selected expert, its row of largest Σ\|w\| |
| tern | (b)'s trained experts, layers 1-19 |
| ph | (b)'s post-hoc experts, no training |
| fix5 | tern with layer 5's top expert, this run's, restored to the original weights |
| fix15 | fix5 with layer 1's top expert restored too |

## Controls (`s2c_verdict.py`, selftest PASS)

| Control | Rule |
| --- | --- |
| C1 planted | at layer 3, plant's top expert and top dimension of the own error are the planted ones, each with share ≥ 0.9 |
| C2 replay | tern's raw jump ratio, carried at layer 6 over layer 5, within 15% of (b)'s 4.44 |

A failed control makes the run VOID.

## Decision rule (registered)

On fix5 against tern:
- **"THE JUMP IS A MASSIVE DIMENSION THE NORMS REMOVE":** fix5 cuts the raw carried error at layer 6 by ≥ 50%, while
  it moves the normalized carried error at layer 6 and the head's KL by < 10% each;
- **"THE JUMP CARRIES THE LOSS":** fix5 cuts the head's KL by ≥ 25%;
- **"PARTLY"** otherwise.

**Reported, not judged:**
- fix15 (the same with layer 1's source);
- ph against tern (what training changed);
- each layer's own error in normalized space;
- the routing agreement.

## Predictions (registered after the dry runs)

| | Predicted |
| --- | --- |
| label | THE JUMP IS A MASSIVE DIMENSION THE NORMS REMOVE, 85% |
| tern head | top-1 0.78-0.83, KL 0.40-0.55 |
| fix15 against tern, KL | within 10% |
| route_same at layer 10 | 0.15-0.30 |
| C1, C2 | pass |

## What a result changes (rule 5)

- **MASSIVE DIMENSION:**
  - part 2's training and its logs read the error in normalized space, not the raw mean square;
  - keeping the massive-activation rows out of ternary is not a lever;
  - part 2's loss sees the routing, which flips for most tokens long before layer 6.
- **CARRIES THE LOSS:** part 2 keeps the named rows at higher precision, which costs kilobytes, and measures what that
  buys at the panel before any long training.

## Parts specific to Ling

- **Specific:**
  - the experts and dimensions named (13 and 1149 at layer 1, 118 and 848 at layer 5);
  - the sigmoid router with group selection, whose routing agreement is measured.
- **Not specific:** massive activations and super weights are reported across LLMs (Sun et al., Yu et al.). So is the
  normalized-space reading.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| (b)'s 1.58 M tokens | 25,000 held-out tokens | CPU time; the replay is on the jump ratio, which a token sample keeps |
| the GGUF panel's numbers | fp32 PyTorch on the bf16 weights, top-1 and KL at the head | a diagnosis of the stack, not a panel score |
| bf16 streams | fp32 streams | the CPU's bf16 matmuls are slow; the rounding is below the errors measured |

## Cost

About 3-4 h of CPU (6 threads). It runs after test 2's chain, alone.
