# Stage 2, part 1, test 2c: the organs without a router, ternary on every layer. PROTOCOL

2026-11-04. Ling-mini, one RTX 3060, block-wise training as stage 1 (b). It runs beside tests 2b and 3b, and stops during
the idle session.

## Question

Attention, the shared expert and the dense FFN are read by every token, so no router can choose where to read them exact.
Test 2 trained one layer of each. **What do they cost stacked, on the whole panel?** Three numbers: attention alone, shared
expert plus dense FFN, and the two together. Routed experts and head stay as saved in P0.

## Thesis line (rule 4)

Ling-flash on the desk, k = 4, head in two passes, q8_0 KV (`s2s_ceiling.py`, `results/s2s_ceiling.json`):

| Left as stored (Q4_K_M) | ms | tok/s |
| --- | ---: | ---: |
| nothing | 39.37 | 25.40 |
| dense FFN | 40.94 | 24.43 |
| shared expert | 41.73 | 23.97 |
| both | 43.29 | 23.10 |
| attention | 55.54 | 18.00 |

The room under 24.7 is 1.12 ms, and x15 of test 3b would spend 1.03 of it. So none of these organs can stay as stored, and
attention is 16 ms away. They hold 300 M parameters on mini against 15.3 B in the routed experts. A training of these organs
alone keeps optimizer state for 2% of the weights, though it still runs forward and backward through the frozen experts.

## Literature (read in the papers)

| Source | What it gives here | Known ways to fail |
| --- | --- | --- |
| QEP, arXiv 2504.09629, §4-5, Fig. 2, Table 1 | error grows about exponentially through quantized blocks and keeps growing after them; aiming each layer at the full-precision output from the quantized input (Eq. 5) is what (b) does; at INT2 it takes QuIP on Llama-2-7B from perplexity 65.6 to 12.0 | full correction overfits a small calibration set, mostly in MLP blocks (§5.3): default α = 1/2 |
| CBQ, arXiv 2312.07950, Table 3(c), Table 7 | blocks trained in overlapping windows: W4A4 Wiki perplexity 11.98 with one block, 9.45 with 4 blocks and overlap 3 | the paper shows no error curve; its W2 rows keep part of the first and last blocks at 4 bits |
| EfficientQAT, arXiv 2407.11062 (test 2) | block-wise training of all weights and steps | overfitting with few samples |
| ParetoQ, arXiv 2502.02631, §2.2, Fig. 3-4 | ternary QAT saturates near 30 B tokens, against 10 B at 3-4 bits; at this width QAT rebuilds the weights (about 40% changed), it does not adjust them | this test sees 1.5 M tokens per layer, 20,000 times fewer |
| BitDistill, arXiv 2510.13998 (test 2) | SubLN steadies the conversion to 1.58 bits | unstable without it; changes the graph |
| SliderQuant, arXiv 2603.25284, §1, Fig. 1 | first and last layers cost most, one layer at a time and as a growing prefix, at W4A4 on Llama and Qwen | no split of attention from FFN; 4 bits, not ternary |
| Stage 1 (b), test 2, test 3 (this repo) | one layer of experts is free, 19 cost −8.52; KL recovered by training 17%; attention and shared expert lose about twice what experts lose before training (0.60, 0.69 against 0.35) | deep layers overfit (0.651 against 0.732 at layer 19) |

No paper found measures attention against FFN at ternary on a converted LLM.

## Design

`s2s_train.py`, `s2s_export.py`, `s2s_run.py`. Teacher stream written once to disk (21 × 6.5 GB).

| Arm | Ternary | Training | Panels | Role |
| --- | --- | --- | --- | --- |
| st_both | attention 0-19, dense FFN 0, shared expert 1-19 | attention and FFN of a layer together | all 7 | **judged** |
| st_attn | attention 0-19 | block by block | all 7 | reported |
| st_ffn | dense FFN 0, shared expert 1-19 | block by block | all 7 | reported |
| st_attn_lo | attention 0-9 | st_attn's layers, no new training | documents | depth |
| st_attn_hi | attention 10-19 | on the original stream up to layer 10 | documents | depth |
| ph_both | as st_both | none: initial scales | documents | what training buys; plant |

- **Per layer, in order, on the student stream:** the organ's tensors take test 2's quantizer (codes {−1, 0, 1}, one scale per
  row, Lloyd start, no outlier rule, STE, LSQ). They train on the block's output against the teacher's, in the space the
  later norms read. The rest of the block is the original bf16, routing k = 8. The target is the teacher's stream, so each
  block also absorbs the error carried in.
- **Data:** stage 1's corpus, 1.5 M tokens, 75 k held out, one epoch, Adafactor, about 4,096 tokens per step.
- **lr:** test 2's picks per tensor (attention 1e-3, shared expert 3e-3, dense FFN 1e-3) times one scale, picked on a stacked
  layer: st_both's layers 0-1 at the picks, then layer 2 from that stream at scales 0.3, 1 and 3. The least held-out block
  loss sets the scale for every arm, before any other layer. If it is not 1, layers 0-1 are trained again.
- **Logged per layer, held out:** block loss with the original weights on the student stream (the carried error alone),
  post-hoc, trained, on training sequences; the student stream's error and cosine against the teacher's.
- **Export and panels:** test 2's path with amendment 1; k = 8, mode none.

## Line (registered; `s2s_verdict.py`, selftest PASS, 9 cases)

Stage 1's labels and chat gate, 10,000 paired draws, against ref's 55.19. On st_both:
- **THE ORGANS WITHOUT A ROUTER FIT THE SHARE:** CI95 lower end ≥ −2.5, chat gate;
- **THEY FIT THE BUDGET ALONE:** lower end ≥ −5, chat gate;
- **THEY DO NOT FIT:** otherwise.

**Part 2's first test.** Margin = Δ(st_both) − Δ(x15), x15 being what test 3b judged, the experts at the speed that fits:
- **STAYS ON THE EXPERTS:** st_both fits, or the margin's CI95 reaches 0 or above. A tie stays on the stage's plan.
- **THE ORGANS COST MORE THAN THE EXPERTS:** otherwise. The label does not move anything by itself: the delivery may
  recommend moving the first test, after test 4's page says the training fits on flash.

Reported, not judged: st_attn and st_ffn with labels, top-1 and KL of every arm beside its points, per-panel accuracy, the
depth arms (no line: documents only, read as a direction). Δ(st_both) − Δ(st_attn) − Δ(st_ffn) with its CI95: it mixes the
interaction with joint training, because st_both's weights are not the other arms'.

**THEY DO NOT FIT means with this training.** It does not say an organ cannot be ternary, and it does not stop the stage.

## Controls (a failure is VOID)

| Control | Rule |
| --- | --- |
| C1 floor | each arm's first layer: original weights on the student stream give block loss at most 1e-3 of its post-hoc loss (dry run: 0.0 on the dense layer, 2.5e-11 on layer 10, whose sum is not deterministic on the GPU) |
| C2 exports | layout against the HF checkpoint, tensors TQ2_0 and bit-exact, every other tensor equal to P0's, the exact tensor count (40, 60, 100, 20, 20, 100), every layer logged, each panel's model the export's sha256 |
| C3 plant | ph_both, a known damage: document top-1 < 0.70, KL > 0.10, and it fails the chat gate |

Test 2's planted arm is not repeated: it checked the panel on one layer, and ph_both does it here with chat.

## Predictions (before any full training)

| | Predicted |
| --- | --- |
| st_attn | top-1 0.60-0.78, KL 0.7-1.6, Δ −6 to −20 |
| st_ffn | top-1 0.72-0.85, KL 0.3-0.9, Δ −3 to −12 |
| st_both | top-1 0.50-0.70, Δ −10 to −30; DO NOT FIT 85%, BUDGET 10%, SHARE 5% |
| excess over the sum | negative |
| part 2 | THE ORGANS COST MORE 70% |
| st_attn_lo, st_attn_hi | top-1 0.70-0.85 and 0.75-0.88: early layers cost more |
| ph_both | top-1 under 0.45 |

## What a result changes (rule 5)

The prediction is against the organs, and the decision that changes is this one:
- **STAYS:** part 2 opens as the stage says, on the experts' −8.52. These organs keep this training.
- **THE ORGANS COST MORE:** the delivery recommends where part 2's first mechanism test goes, once test 4's page is read. The
  label ranks cost under one trainer, not what a better training recovers, and a result on these organs would say little
  about the experts: their gradients are dense, and each routed expert sees a thirty-second of the tokens.
- **Attention over, FFN within (or the reverse):** the delivery names the one organ a training must take.
- **Depth arms:** a direction only. Half of attention stored costs about 8 ms on the desk: priced, not proposed.

## Parts specific to Ling

- **Specific:** the fused QKV with q/k norms and RoPE on half of each head; one dense layer; a shared expert of 512 beside
  256 routed experts; 20 layers; which half of the depth costs more.
- **Not specific:** the quantizer, the loss and the sequential training need only RMSNorm before each block. Any MoE has
  organs without a router, and they are read by every token.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| Whole model ternary | experts and head as saved | the question is these organs' share; the whole stack is part 2's |
| k = 4 | k = 8 | to compare with −8.52 |
| IQ2_BN's kernel | TQ2_0, same values | the panel's llama.cpp has no IQ2_BN |
| Ling-flash | Ling-mini | 32 layers on flash: stacking there is not measured, and more depth should cost more |
| Overlapping windows (CBQ) | one block at a time | (b)'s training, as the stage asks; windows belong to part 2 |
| α < 1 (QEP) | full correction | (b)'s target; the train/held-out gap is logged per layer |
| lr grid per layer | one scale, picked on layer 2 of the stack | the pick is on one layer of 20; the deep layers' best lr is not known |
| Experts at one scale per row | x15 keeps (b)'s group 256 and its 5,588 outlier groups | (b)'s weights are what test 3b reads; the margin compares a format that fits with one that holds more, and leans against these organs |
| Experts and organs ternary together | not measured | part 2's; the loss need not add up |
| SubLN | not used | it changes the graph the engine runs |
| 30 B tokens (ParetoQ) | 1.5 M | the trainer of (b); no outcome here says what more tokens would buy |

## Cost

- **GPU, RTX 3060:** teacher about 2 h (measured beside the panels); about 13 min per layer from test 2's rates, so st_attn,
  st_ffn and st_both about 4 h each and st_attn_hi 2 h; lr pick about 1 h. About 17 h alone, more beside the panels. Over
  10 h: independent reviewer below.
- **CPU, after 3b:** six exports of about 45 min, three whole panels of 2-3 h, three document panels of 37 min, one chat.
- **RAM:** 13 GB for the two streams, measured beside the panels.

## Review (rule 6)

A fresh reviewer read the thesis, this protocol and the code. Objections and answers:

| # | Objection | Answer |
| --- | --- | --- |
| 1 | STAYS tested the share, so organs that fit the budget could still move training | it tests FITS; selftest case added |
| 2 | A tie moved for a reason that counts twice: x15 already spends the room, and "2%" is a constant | a tie stays on the plan |
| 3 | The margin ranks cost under a weak trainer, not recovery; a stack-seeing loss still crosses the experts; test 4's page does not exist | the label moves nothing by itself: a recommendation, after test 4 |
| 4 | A crash at layer 12 or a dry run left in the directory passes C2 | exact tensor counts and logged layers in C2 |
| 5 | Panels' models and x15's status are not checked | both checked in the verdict |
| 6 | C1 at 1e-9 can VOID on a MoE layer: `index_add` is not deterministic | tolerance relative to post-hoc, measured on layer 10 |
| 7 | lr picks came from clean inputs, and the reason for no grid is cell-scale | a scale picked on a stacked layer, about 1 h |
| 8 | C3 barely fires and skips the judged path | line 0.70, ph_both runs chat and must fail the gate |
| 9 | The margin is not like for like, and experts plus organs are never measured | stated in rule 1's table |
| 10 | The excess, the halves and DO NOT FIT cannot carry what was claimed | the excess is marked mixed, the halves are a direction, DO NOT FIT is scoped to this training |
