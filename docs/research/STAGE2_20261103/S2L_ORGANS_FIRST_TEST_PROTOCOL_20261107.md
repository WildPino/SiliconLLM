# Stage 2, part 2, first test: the organs without a router. A test of mechanism. PROTOCOL

2026-11-07. Ling-mini. GPU for two trainings, CPU beside it. Code: `s2s_train.py --form`, `s2y_export.py`, `s2y_ks.py`, `s2y_run.py`, `s2y_chain.py`, `s2y_verdict.py`, `s2y_dry.py`.

## Question

The organs cost 20 ms when stored and −13.36 points when ternary. Five readings, one document: does a four-level grid move them? How large is one seed's noise? What does the first real stack score? Why does st_ffn collapse? Does an existing 2-bit format without training beat our training?

**Judged on top-1 and panels, never on block loss** (test 2c showed it does not stand in for them).

## Thesis line (rule 4)

No arm reaches the 24.7 line alone. The stack's row with ik's attention is 41.29 + 0.25 + 3.92 = **45.46 ms, 22.00 tok/s**, a sum of cells, attention ternary and FFN organs stored. Ik's flash path stays a stage 3 matter. Mini at 1.5 M positions is not flash: this is a mechanism budget.

## Arms

| Arm | What | Where | Panels |
| --- | --- | --- | --- |
| **as_both, judged** | st_both, same seed, data, lr scale 3, in {−1, 0, 1, 2} with a sign per row | GPU, about 5.5 h | all seven, k = 8 |
| st_both2 | st_both ternary, seed 20261107 | GPU, about 5.5 h | all seven, k = 8 |
| stk4 | bal's experts, st_attn's attention, shared expert and dense FFN as stored, no training | CPU | all seven, generation at k = 4 |
| ff_nodense | st_ffn, dense FFN back as stored | CPU | documents, chat |
| ff_noearly | st_ffn, shared expert of layers 1-9 back | CPU | documents, chat |
| ff_nolate | st_ffn, shared expert of layers 10-19 back | CPU | documents, chat |
| ff_keepout | st_ffn, outlier groups back to the original values | CPU | documents, chat |
| ff_keeprnd | as ff_keepout, the same number of groups per tensor, at random | CPU | documents, chat |
| ks_org | attention, shared expert, dense FFN at IQ2_KS's values with the imatrix, no training | CPU | documents |

- **Asym:** Lloyd on both signs, the better one per row (S2I's `q_asym`), STE and LSQ as ternary. The panels read Q4_0 holding the same values, written by bytes. Quality only.
- **The seed** sets the order of the data and nothing else: the start is deterministic.
- **Outliers:** the organs have no rule, the experts have one (group 256, maximum over 8 times the TWN scale). On the checkpoint it marks **2.1% of the shared expert's groups (4,993 of 233,472)**, 0.04% of attention's (322) and 0.03% of dense FFN's (31), most on gate and up of layers 2-10. Record: `results/s2y_outliers.json`.
- **IQ2_KS:** a scale per row, block scales and non-linear levels, 2.19 bits per weight; 3.73 ms per 4096² against IQ2_BN's 3.36 (cell 3). llama.cpp does not read it, so ik's quantizer writes it and requantizes the file to BF16, and the panels read those values in a BF16 container.

## Lines (registered; `s2y_verdict.py`, selftest PASS, 13 cases)

**Noise:** gap = |Δ(st_both2) − Δ(st_both)|; top-1 gap likewise, on the documents.

**Grid, judged:** margin = Δ(as_both) − Δ(st_both), paired CI95.
- **THE GRID MOVES THE ORGANS:** CI95 lower > 0, margin > gap, top-1 gain > top-1 gap.
- **THE GRID HURTS THE ORGANS:** the mirror.
- **WITHIN THE NOISE:** otherwise.

**Stack:** stage 1's label and chat gate on stk4, beside the sum of parts, −4.74 − 6.25 = −10.99.

**st_ffn:** recovered = (top-1 − top-1 of st_ffn) / (1 − top-1 of st_ffn).
- **ONE GROUP CARRIES IT:** the best of the three groups recovers ≥ 0.5, the next at most half of that. Otherwise **IT IS SPREAD**.
- **OUTLIERS WEIGH:** recovered(ff_keepout) − recovered(ff_keeprnd) ≥ 0.15. Otherwise **OUTLIERS DO NOT CARRY IT**.

**IQ2_KS:** **BEATS THE TRAINING** if top-1 is above st_both's by more than the top-1 gap and KL is under st_both's; **SITS BETWEEN** if top-1 is at least ph_both's + 0.10; else **NO BETTER THAN POST-HOC**.

## Controls

| Control | Rule | Failing means |
| --- | --- | --- |
| Exports | replaced tensors read back to the intended values bit for bit; every other tensor equals the reference's bytes (P0, or bal-q4 for stk4) | VOID |
| Tool | st_both's file through these binaries gives test 2c's answers on 30 nq items | VOID |
| Training | 20 layers logged, trained under post-hoc on each | VOID |
| Imatrix | on every organ tensor, weighted error with the imatrix under the one without | VOID |
| Weight error | IQ2_KS's relative error per tensor, reported: a decoding fault reads near 1 | |
| Planted, dry | `s2y_dry.py`: grid back at 5% noise, both signs, estimators, Q4_0 exact, a 1% scale error fails at 0.0100 against 0.0039: PASS | |
| Trivial | ff_keeprnd for outliers; ph_both for IQ2_KS; the seed for the grid | |

## Predictions

| | Predicted |
| --- | --- |
| as_both | Δ −12 to −7, top-1 0.65-0.74, chat 16-24 |
| Grid | MOVES 45%, NOISE 50%, HURTS 5% |
| Gap, top-1 gap | 0.3-2.5 points, at most 0.015 |
| stk4 | Δ −13 to −8, chat 18-24; OVER BUDGET 70%, UNDECIDED 28%, WITHIN 2% |
| st_ffn | SPREAD 55%, ONE GROUP 45%; ff_noearly recovers most |
| Outliers | WEIGH 60%; ff_keepout recovers 0.2-0.6 |
| ks_org | top-1 0.70-0.85, KL 0.4-1.2, dBPB +0.10 to +0.50; BEATS 60%, BETWEEN 39% |

The grid's NOISE at 50% is not a predicted failure of the mechanism: step 1's organ gains were 0.15-0.19 against the experts' 0.38, so the expected real gain is about 2 points, near the panels' reach.

## What a result changes (rule 5)

- **Grid MOVES:** part 2 trains the organs in asym, then buys tokens. **NOISE or HURTS:** the grid is dropped on the organs; part 2 buys tokens and downstream blocks.
- **Gap:** every one-seed margin of this stage, bal's +7.22 included, is read against it.
- **Stack WITHIN BUDGET:** it becomes the configuration to beat, at 22.00 tok/s. **Otherwise:** attention trains with bal's experts in place, not apart.
- **ONE GROUP:** that group stays stored or takes its own training, at a known price. **OUTLIERS WEIGH:** the organs take the rule in part 2. **Neither:** the cause is in the training, and the next suspect is the lr below.
- **IQ2_KS BEATS:** organs start from IQ2_KS at 11% more time, and the format enters the plan. **BETWEEN:** a start for training, no more. **NO BETTER:** it closes.

## Literature (rule 2)

Asym, ParetoQ's reading and its way to fail: S2I and S2K. STE and LSQ: stage 1 (b). The outlier rule is stage 1 (b)'s own. IQ2_KS has no paper: read in ik_llama.cpp's `iqk_quantize.cpp` and `ggml-common.h`. It is a post-training format, so its known way to fail is the imatrix's text, which here is fit24, the one P0 used.

## What this takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| Noise over seeds | one pair, data order only | two trainings fit the budget; a gap is not a spread, and it may understate noise from the corpus |
| lr pick for asym | st_both's scale 3 | one variable changes; the grid's best lr is not known |
| Row sign in the format | a bit per row, Q4_0 in the panels | same values; bytes and speed are not what panels read |
| Stack trained together | st_attn trained beside stored experts, the head stored | it is the first stack, and no GPU is spent on it |
| Outlier rule in training | groups restored, not retrained | the whole group at its original value: an upper bound on what the rule can buy |
| Restoring as stored | ff_keepout's groups in BF16, not Q4_K | a tensor cannot mix types; the difference is smaller than the effect sought |
| lr of the shared expert | not tested | st_ffn ran it at 9e-3, three times test 2's pick, chosen on st_both's layer 2. Testing it needs a GPU arm not in this plan; the result says if it is left as the suspect |
| IQ2_KS on the panels | documents, in BF16 | a container with the same values; this arm reads documents only |
| Bal under an exact slot | not run | not part of this test |

## Rule 6

Two trainings add up to about 11 h of GPU, past the 10 h. No independent reviewer ran: subagents are not in use for now. The rule is not met, and this says so.

## Parts specific to Ling

The numbers, the outlier counts, one dense layer, the shared expert beside routed experts, the fused QKV. Not specific: the grid, the method, the restoration, IQ2_KS.

## Cost

GPU about 11 h, one trainer at a time. CPU about 14 h, beside it: stk4 3 h, five st_ffn arms 4 h, ks_org 1 h, the two trained arms 3 h each. No timing runs here.
