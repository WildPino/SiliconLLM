# Stage 2, part 1, test 2b: the head in two passes, without new training. PROTOCOL

2026-11-04. Ling-mini, quality only. No training runs in this test. No timing runs in it.

## Question

Test 2 left the head as the one organ that does not stay ternary (top-1 0.820, KL 0.251), and no route to 24.7 tok/s
keeps it at Q6_K. A cheap preview over the whole vocabulary picks K candidates. Their rows are read from the stored Q6_K
head, so their logits are exact. Does the original's choice fall among the candidates, and do the panels stay on the
line?

## Thesis line (rule 4)

On Ling-flash the head costs 16.5 ms as stored and 4.14 ms ternary. The desk ceiling (`s2t_ceiling.py`, DESK-MODEL, every
other organ ternary, full q8_0 KV) is 19.44 tok/s as stored. With the two cells below it is 25.40 and 26.18, against
24.7. A PASS with cell 2 means no head training at all; a PASS with cell 1 alone means the head's training stays as test
2 did it, and flash pays for it once. Neither says anything about the experts. The test runs on mini: flash's numbers are
a desk reading, and the flash sweep below is the first check of that reading.

## Literature (read in the papers)

| Source | What it gives here | Known ways to fail |
| --- | --- | --- |
| Shim et al., SVD-softmax, NIPS 2017, Alg. 1, Tables 1-3 | preview window W, N exact rows, tail keeps preview outputs (Eq. 4); W = 256, D = 2,048, 80K words, N = 4,096: NLL 3.589 against 3.605, top-100 99.85%; "5% to 10%" of the vocabulary sufficed | window proportional to D, 1/8 of D adequate; too short a window "affects the performance badly" |
| Chen et al., Learning to Screen, ICLR 2019, Table 1 | candidates by cluster of the context vector, tail at zero; on PTB, P@1 0.996-0.998 at 10-45x | the clustering is trained (Gumbel); spherical k-means alone is weaker. Tail zero gives a target outside the list no probability |
| Samaga B L et al., HiRE, arXiv 2402.09360, Table 4 | int4 preview with exact refinement: top-1 55.94% without it, 56.99% with k′ = 128, 31.30 of the top-32 found; low-rank r = 25%: 56.92%, 29.93 at k′ = 512 | "a k′ that is much larger than k is critical"; the low-rank factors need training; the tail is not kept |
| mini-sglang PR #169 (draft) | INT8 head, exact top-64 in BF16, Qwen3-0.6B +5.82% at B = 1 | misses depend on the prompt: 480/512 tokens match at 0.6B, B = 8; the sampling tail stays approximate |
| Zhao et al., FR-Spec, arXiv 2502.14856, Table 1 | a frequency-ranked subset for a draft's head: 75% less head compute | a fixed list loses the rare tokens: the basis of the trivial control |
| Wang et al., SVD-LLM, arXiv 2403.07378 | truncation by whitened SVD (Cholesky of the input covariance), so singular values rank the output's error | needs calibration data |

**Engines:** llama.cpp 5b335f4's `bailingmoe2.cpp` multiplies `result_norm` by the whole `model.output`. A grep of
ik_llama.cpp's `src/*.cpp` finds no partial head, and a web search found nothing. Neither does this today.

## Design (`s2t_head.py`, `s2t_run.py`, `s2t_two_pass.h`)

**Previews**, from the stored head:
- trained ternary (test 2) and untrained ternary (test 1);
- sign, per-row int4;
- SVD at ranks 32-512 and whitened SVD at the same ranks, the factor in int8 per row;
- two lists that read no weights: frequency, and 64 or 256 k-means clusters of the hidden state.

**Composition:** the candidates are the K largest preview logits. They take the exact logits. The tail keeps the preview
plus the offset mean(exact − preview) over the candidates (`--tail-offset 1`; the offset is ours, SVD-softmax has none).
The token is chosen among the candidates only.

**Recall** is counted against the panel tool's own top-1 and top-32, from a run's records, never a float recomputation.
Each position's hidden state comes from the same process that wrote its records.

**Panels:** the preview's logits come from a second llama context on a GGUF that differs from P0 only in `output.weight`,
so both heads run on llama.cpp's kernels. Each file is pinned by sha in `s2t_run.py`. Tensor by tensor, `org_head` and
`wsvd256` differ from P0 only there (`results/s2t_materialize_wsvd256.json`).

**Sampling:** the engine samples from the composed distribution, so a tail token can be drawn at a temperature, with the
preview's logit. Greedy chooses among the candidates. The tail's mass, exact and composed, is reported.

## Found on dev, before registration

fit is 150,005 tokens of the head trainer's training split, used for whitening, frequency and clusters. dev is its
held-out 74,867 positions (`results/s2t_sweep_dev.json`). Miss = the original's top-1 outside the candidates, per 1,000.

| Preview | MB (mini) | K=256 | K=1024 | K=4096 | Hard, K=4096 | Top-32 covered, K=4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tern_tr | 81.1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.0000 |
| tern_im | 81.1 | 3.54 | 2.94 | 0.40 | 1.71 | 0.9954 |
| int4 | 161.6 | 0.00 | 0.00 | 0.00 | 0.00 | 1.0000 |
| sign | 40.9 | 14.51 | 5.82 | 2.83 | 13.45 | 0.9925 |
| wsvd512 | 83.2 | 4.82 | 0.64 | 0.07 | 0.19 | 0.9984 |
| wsvd256 | 41.9 | 21.56 | 4.22 | 0.48 | 1.08 | 0.9905 |
| wsvd128 | 21.3 | 54.90 | 15.61 | 2.34 | 3.94 | 0.9699 |
| svd512 | 83.2 | 93.93 | 33.21 | 5.76 | 12.57 | 0.9692 |
| svd256 | 41.9 | 130.12 | 51.44 | 13.81 | 37.76 | 0.9406 |
| clus256 | 1.0 | 238.37 | 135.99 | 62.20 | 86.12 | 0.8569 |
| freq | 0 | 353.72 | 266.59 | 165.09 | 220.47 | 0.6061 |
| shuf (planted) | 81.1 | 998.36 | 991.76 | 966.94 | 960.08 | 0.0271 |

Random candidates miss 973.9 at K = 4,096. The float head against the tool's records (exact) misses 0.00 at K = 16.

Composed against the exact head in float, over the whole vocabulary, tail offset on:

| Cell | Top-1 | KL (nats) |
| --- | ---: | ---: |
| tern_tr K=256 / 1024 / 4096 | 1.0000 / 1.0000 / 1.0000 | 0.0090 / 0.0043 / 0.0019 |
| wsvd256 K=1024 / 4096 | 0.9958 / 0.9995 | 0.0794 / 0.0127 |
| wsvd512 K=1024 / 4096 | 0.9993 / 0.9999 | 0.0167 / 0.0025 |
| tern_im K=4096 (offset off: 0.0169) | 0.9996 | 0.0257 |
| int4 K=256 | 1.0000 | 0.0009 |

Reading: at equal rank, whitening cuts the misses at K = 4,096 by a factor of 3 at rank 32 and 80 at rank 512. The lists
do not reach the line. The offset helps the
trained preview and hurts the untrained ternary one.

## Registered cells

**Cell 1, the best preview: tern_tr, K = 1,024, tail offset on.** It alone has no miss on dev down to K = 256. K is the
smallest with KL ≤ 0.005. Desk ceiling 25.40 tok/s.

**Cell 2, the untrained preview: wsvd256, K = 4,096, tail offset on.** Cell 1's preview was trained on mini, and flash
would have to pay for that training. Whitened SVD needs only a dump of hidden states. Among untrained previews with a
ceiling above 24.7 it has the highest ceiling, 26.18 tok/s, against tern_im's 25.19; wsvd512 at K = 4,096 reads 24.48.
Cell 2 decides whether flash's head needs any training. It cannot change cell 1's outcome.

In the panels wsvd256 is a dense f16 head, the product of its factors, rounded through bf16; the sweep reads the factors
in float. The miss that counts on the documents is the panel tool's own.

## Line (registered; `s2t_verdict.py`, selftest PASS, 23 cases)

Judged on data that took no part in the choice. Each cell is read at its own K and at 4,096.

| Check | Line |
| --- | --- |
| Miss on the panel's 72 documents (165,858 positions), the panel tool's own count at the cell's K | ≤ 1 per 1,000 |
| Miss on the documents at K = 4,096, float sweep | ≤ 1 per 1,000 |
| Miss on the generated positions of ref's answers (nq, tqa, gsm8k, chat), cell's K and 4,096, float sweep | ≤ 1 per 1,000 |
| Miss on the documents with stage 1 (b)'s ternary experts, against that model's own records, cell's K and 4,096, float sweep | ≤ 1 per 1,000 |
| KL top-32, documents, panel tool | ≤ 0.02 nats |
| KL over the whole vocabulary, documents, float | ≤ 0.02 nats |
| dBPB, CI95 upper bound | ≤ +0.005 |
| Chat, 40 items | ≥ 27 passes, ≤ 1 degenerate (ref: 28 and 0) |

Top-1 agreement is reported, not judged: with exact candidates it equals 1 − the tool's miss, and the verdict checks this
identity. The KL and BPB lines are tight because this head is meant to be near exact and must leave the 5-point budget
to the experts: 0.02 is a fifth of an organ's line in test 2. They were set after dev, and they clear both cells there,
so the miss lines are the ones with teeth. Chat is a gate only: test 2's 0.82 head scored 26.

**Items touched.** Candidates are exact, so a greedy answer can change only after a miss on its own path. An answer with
no miss is ref's, byte for byte. For each panel, the items with at least one miss bound the change of its task score.
This replaces running nq, tqa and gsm8k. If the bound exceeds 1 point on a panel, the touched items are generated again
in two passes and scored.

Outcomes for cell 1: **THE WORD STAYS OUT OF THE CANDIDATES**; **THE PANELS LEAVE THE LINE**; **THE HEAD READS IN TWO
PASSES**; or **PASS, AND A WEAKER LIST PASSES TOO: RECALL DOES NOT DECIDE ALONE**. For cell 2: **THE UNTRAINED PREVIEW
PASSES TOO** or **ONLY THE TRAINED PREVIEW IS WITHIN THE LINE**.

**Reported, not judged:** every preview's curve K = 16-4,096 and bytes per token on each set, with the CI95 by document;
hard positions (p(top-1) < 0.5), rare choices, each generated panel, the worst document; the top-32 cover; the tail's
mass; the gap between the tool's count and the float sweep on the documents.

**Flash sweep, reported, after the panels:** the existing Ling-flash Q4_K_M GGUF, no conversion. Hidden states and records
on about 60,000 fit and 20,000 dev positions, alone on the machine's CPU, then wsvd at ranks 256 and 512 and sign. It
reads the assumption that the rank doubles. It does not change this test's outcome.

## Controls

| Control | Rule | Failing means |
| --- | --- | --- |
| C1 planted | shuf, tern_tr's rows permuted, misses ≥ 900 per 1,000 at K = 4,096 | VOID |
| C2 trivial | freq misses > 1 per 1,000 at K = 4,096 | recall does not decide alone |
| C2 near the line | wsvd128, 2.34 on dev, misses > 1 per 1,000 on the documents at K = 4,096 | recall does not decide alone |
| C3 parity | the float head misses ≤ 0.1 per 1,000 at K = 16 against the tool's records, on documents, generated answers and ternary-expert documents | VOID |
| C4 replay | on the first 2 documents, K = the vocabulary gives records equal to ref byte for byte | VOID |
| C4 plant | on the same documents, test 1's shuffled head as preview at K = 1,024 gives top-1 < 0.5 | VOID |
| Identity | panel top-1 = 1 − the tool's miss, and the records' misses equal the tool's count | VOID |

C1 and C4 catch a broken instrument, not a marginal one: C2's second row is the one near the line. C4's replay does not
exercise the tail; C4's plant does. The plant also differs from P0 in `output_norm.weight`, which it may.

Dump check: the documents' records written with the hidden states must equal ref's byte for byte, or the run stops.

## Predictions

| | Predicted |
| --- | --- |
| Cell 1 outcome | THE HEAD READS IN TWO PASSES, 85% |
| Cell 1 miss at K = 4,096 | documents 0-0.2; generated 0-0.5 |
| Cell 1 panels | top-1 ≥ 0.999; KL top-32 0.001-0.006; \|dBPB\| ≤ 0.002; chat 27-29, 0 degenerate |
| Cell 2 label | THE UNTRAINED PREVIEW PASSES TOO, 45% |
| Cell 2 miss at K = 4,096 | documents 0.4-1.5; generated 0.5-2 |
| Cell 2 panels | top-1 ≥ 0.998; KL top-32 0.003-0.02; chat 26-29 |
| Ternary-expert states | cell 1 under 1 per 1,000; cell 2 1-4, and the likeliest reason it fails |
| Items touched, cell 1 | 0-1 per panel |

## What a result changes (rule 5)

- **Cell 1 fails:** the head does not read cheaply, and the decision moves to the speed line.
- **Cell 1 passes, cell 2 passes:** no head training on flash.
- **Cell 1 passes, cell 2 fails:** flash needs a head training of test 2's kind, or tern_im's cell, at 25.19 on the desk,
  which would need its own panels. The result says which.
- **Prediction against the goal (rule 5):** cell 2 is more likely to fail than not, on ternary-expert states. It runs
  anyway because the outcome decides whether flash's head needs a training, and the panels cost hours, not training.

## Parts specific to Ling

Only the vocabulary, 157,184 tokens, which sets the head's weight and how far K sits from V. The mechanism needs a
softmax head and the hidden state before it. The whitening needs a dump of hidden states from the model at hand.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| Judged on flash | mini only, flash pending; the ceiling doubles the rank for flash's width (SVD-softmax Table 1) | the panels and a trained preview on flash are outside this stage; the flash sweep reads recall only |
| Speed of the two passes | desk: the gather of K rows at 32 GB/s, the selection a guess of 0.2 ms, thread barriers not counted. Cell 1 clears 24.7 by 1.12 ms | the panels run two whole contexts and time nothing. Selection and gather are timed in the idle session; until then 25.40 and 26.18 are desk numbers |
| The low-rank preview as two small products | a dense f16 head in the panels | llama.cpp's graph has no such head; the numbers differ by rounding only |
| Other previews in the panels | two cells only | the rest is settled by recall on dev |
| Sampling at a temperature | greedy panels only | the tail matters more under sampling; KL, BPB and the tail's mass are the proxy |
| Task scores on nq, tqa, gsm8k | the items-touched bound | an answer without a miss is ref's byte for byte |
| Experts ternary in the panels | P0 with exact experts; ternary-expert states judged by recall only | test 3b composes them |

## Independent review (rule 6)

A fresh reviewer read the thesis, this protocol, the code and the dev sweep. Ten objections, with what was done:

| Objection | Answer |
| --- | --- |
| Recall is judged at a K the cell does not run | miss is read at each cell's K and at 4,096 |
| The top-1 line cannot fire, and the real-kernel miss is not judged | the tool's count is judged, with a CI by document; top-1 is reported and the identity is checked |
| Judged on states the design will not have | ternary-expert states are on the line, for both cells |
| Cell 2 decides flash from mini, and no conversion is needed | true: a flash GGUF exists. The wording now says "flash pending", and the flash sweep is added |
| "A PASS removes the head" is false for cell 1 | reworded |
| The mean hides where the misses land; task scores are not run | per-panel misses and items touched; the bound replaces the task runs, and it triggers them if over 1 point |
| Files and arms are not enforced | sha pins, the runner refuses anything else, the verdict checks each cell, replay and plant are checked for K and length |
| C1 and C2 cannot fire near the line; C3 covers documents only | wsvd128 added to C2; C3 runs on all three sets |
| Lines set after dev; KL top-32 understates; chat barely discriminates | full-vocabulary KL added to the line; the protocol says the miss lines carry the weight and chat is a gate |
| The tail contradicts itself; the desk margin is thin | sampling stated; tail mass reported; selection and gather timed in the idle session, the margin stated |

## Cost

Dumps of documents, generated answers and ternary-expert documents: about 3 h. Sweeps: 15 min. Each cell's panels run two
contexts: about 3 h. All on the CPU, one at a time, about 9 h in all. The flash sweep comes after, about 6 h of CPU. No
GPU beyond the sweeps.
