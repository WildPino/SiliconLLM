# C50-A4 result: P0 against BF16, NOT PASSED; the planted positive passes too

17 October 2026. The result of the [A4 protocol](C50_A4_QUALITY_PROTOCOL_20261017.md), the cell that
qualifies T0 = P0 (ROADMAP §5 S0).

## Verdict, as registered

- **A4: NOT PASSED.** G6 (chat) is INCONCLUSIVE. Every other gate passes.
  - On chat, BF16 itself scores 27/40 (0.675), below the panel's ability floor of 0.75.
  - The protocol reads an INCONCLUSIVE gate as "not passed", never as a pass.
- **Planted positive: the registered expectation is NOT MET.** The planted positive PC (the same
  source and imatrix quantized to Q2_K) was expected to fail G1 and at least one of G2-G7. It
  fails nothing:
  - it passes G1 with an upper CI95 of +0.0017;
  - it passes G2-G5 and G7;
  - G6 is INCONCLUSIVE for it too.
  The registered consequence: a P0 PASS would be uninformative, and a design review follows
  (rule 3).
- **Registered prediction: MISSED.** It said P0's fresh dBPB lies in +0.005..+0.020. The measured
  dBPB is −0.0063.
- **The "if A4 fails" branch of ROADMAP §5 S0 does not apply mechanically.** That branch moves T0 to
  Q5_K_M, then Q6_K. No gate measured a P0 loss. The two defects below (a chat floor that BF16
  misses, and a G1 that cannot see Q2_K) would block any candidate in the same way. A higher
  precision would not change the outcome, so T0 is not moved on this result.

## Run

- **Session:** one session of `c50_a4_run.py`, 09:17:55-21:57:34. Arms in the order P0, BF16, PC;
  21 panels, all COMPLETE, none VOID.
- **Wall time per arm:**
  - P0 12,714 s;
  - BF16 20,310 s;
  - PC 11,620 s.
- **Disk contention, disclosed:** the Ling-flash download ran beside the BF16 and PC arms. It was
  paused during the BF16 cold load. It changes wall time only: the outputs are deterministic
  (greedy decoding, fixed IDs).
- **Records** in `benchmarks/chatbot50/results/` (sha256 prefixes):

  | File | sha256 |
  | --- | --- |
  | `c50_a4_run.json` | `e77e1384…` |
  | `c50_a4_scores_{p0,bf16,pc}.json` | `bfc5722f…`, `71c242c4…`, `1023bae9…` |
  | `c50_a4_verdict_{p0,pc}.json` | `fddab1c7…`, `6d5cf174…` |

## Gates

**Notation:** b = items BF16 gets right and the candidate wrong; c = the reverse.

**P0 against BF16:**

| Gate | BF16 | P0 | b / c | Result |
| --- | ---: | ---: | ---: | --- |
| G1 documents (72), BPB | 0.7749 | 0.7686 | dBPB −0.0063, CI95 [−0.0097, −0.0032] | PASS; preferred bound met |
| G2 PIQA (1,838) | 1,458 | 1,461 | 30 / 33 | PASS (STRAT-01 rule ≥ 1,429 met) |
| G3 ARC-C (600) | 316 | 319 | 17 / 20 | PASS |
| G4 GSM8K (100) | 83 | 81 | 5 / 3 | PASS |
| G5 IFEval subset (173) | 134 | 135 | 9 / 10 | PASS |
| G6 chat (40) | 27 | 28 | 0 / 1 | INCONCLUSIVE (BF16 0.675 < floor 0.75) |
| G7 rollout (72), degenerate | 2 | 2 | 1 candidate-only | PASS |

- **P0's G1 by category:** code +0.0003, technical −0.0014, prose −0.0178. (Erratum, 18 October: the technical value was first transcribed here as +0.0014; `c50_a4_verdict_p0.json` records −0.00141.)

**PC (planted positive) against BF16:**

| Gate | BF16 | PC | b / c | Result |
| --- | ---: | ---: | ---: | --- |
| G1 documents | 0.7749 | 0.7673 | dBPB −0.0076, CI95 [−0.0168, +0.0017] | PASS |
| G2 PIQA | 1,458 | 1,464 | 75 / 81 | PASS |
| G3 ARC-C | 316 | 332 | 36 / 52 | PASS |
| G4 GSM8K | 83 | 80 | 9 / 6 | PASS |
| G5 IFEval subset | 134 | 137 | 10 / 13 | PASS |
| G6 chat | 27 | 29 | 1 / 3 | INCONCLUSIVE |
| G7 rollout | 2 | 1 | 0 candidate-only | PASS |

- **PC's G1 by category:** code +0.0072, technical +0.0142, prose −0.0442.
- **What the task gates do see:** PC disagrees with BF16 on about 2.5× as many items as P0 does
  (b + c: PIQA 156 against 63; ARC 88 against 37). The disagreements are symmetric, so no net
  loss appears at these panel sizes.

## Diagnostic 1 (post hoc; it decides nothing)

**What was run:** `c50_a4_diag_tail.py`, on the recorded document outputs only, with no new run.
Record: `results/c50_a4_diag1_tail.json`. It asks where the negative dBPB comes from.

**The tail.** There are 1,846 tail positions out of 165,858 (1.1%): positions where BF16 gives
the target a log-probability below −15, while its own top-1 is confident.
- **Medians on the tail, BF16:** target −17.1; top-1 −0.28.
- **By category:** prose 1,048, technical 427, code 371.
- **Examples:**
  - " ordering" after "…before the War of";
  - "inn" in an OCR-split "binn|acles";
  - "aukee" as BF16's top-1 at position 1 of 71 of the 72 documents. The spans start mid-text,
    with no BOS, so position 1 sees a single token. 65 of the 1,846 tail positions are these.

**Results:**

| | P0 | PC |
| --- | ---: | ---: |
| Net Δbits against BF16, all positions | −3,721 | −4,473 |
| … on the tail | −1,116 | −4,575 |
| … outside the tail | −2,605 | **+103** |
| Top-1 agreement with BF16's top-1 | **0.9265** | **0.8485** |
| Top-1 accuracy (BF16 0.6726) | 0.6723 | 0.6664 |
| Mean \|Δ log p(target)\|, nats | 0.226 | 0.495 |

**Reading (labelled as a diagnostic):**
- **PC's whole G1 gain comes from the overconfident tail.** On the tail, Q2_K's flatter distribution
  loses fewer bits (median −15.9 against −17.1). Outside the tail, PC is worse than BF16.
- **On these documents, BPB rewards flattening as much as fidelity,** so G1 cannot see a
  Q2_K-sized loss. This matches the protocol's own caveat: FIT text is probably in pretraining.
- **P0 is lower than BF16 outside the tail as well (−2,605 bits).** It was quantized with an
  imatrix computed on FIT items 1-24 of the same categories. That is a possible imatrix fit to the
  FIT distribution, and it is not established here.
- **The per-position fidelity numbers discriminate:** top-1 agreement 0.93 against 0.85, and a mean
  |Δ log p| 2.2× larger for PC. They are post hoc, and no gate is set from them on these data.

## Design review (rule 3): what changes, before any new number

- **What is wrong:** two instrument defects, not a measured loss.
  - **G6:** the chat floor (0.75) was set before BF16's chat ability was known, and BF16 misses
    it.
  - **G1:** a pooled dBPB on FIT documents cannot see a Q2_K-sized loss, because the overconfident
    tail dominates.
- **Proposed A4-bis** (the protocol is pushed before it runs; it needs the owner's go, rule 7, at
  ~3-4 h):
  - **Fresh documents.** Use items read by no cell and outside the imatrix sample. The source is
    fixed in the protocol.
  - **A fidelity gate against BF16, per position.** Top-1 agreement and the mean |Δ log p|, with
    the threshold derived from an existing measured floor, not from these A4 numbers. BPB is still
    reported.
  - **PC must fail the new gate on the fresh documents** before P0's result counts.
  - **The chat panel:** either its floor is re-based on a pre-registered rule, or the panel becomes
    reported-only. Which one is fixed in the protocol.
- **Until A4-bis:**
  - T0 = P0 stays the working reference, unqualified.
  - S0 (v0 of T0) waits for A4-bis.
  - A1b (process stability, a property of machine and engine) and the S2 builds are not affected
    by A4, and may run.

## Design review, step 1 (18 October; no new model output was read)

This step **supersedes the fidelity-gate proposal above.** No gate is registered yet.

**Two facts from tools run on data already seen:**

- **The fresh documents exist.** `c50_holdout_ext.py --parts fresh` builds FIT ranks 249-272 per
  category: 72 items, 161,506 tokens, read by no cell and outside the imatrix sample.
  - Its proof check passes: the regenerated FIT ranks 1-248 equal `fit.jsonl` plus ext_v1's SEL.
  - Record: `results/c50_holdout_fresh_v1_manifest.json`.
  - The inputs are at `E:/…/a4bis/inputs`; their manifest sha256 is `2c01331f…`.
- **The proposed fidelity gate would fail P0 on A4's own data.**
  - The gate: top-1 disagreement with BF16, upper CI95 ≤ 2 × floor_top1 = 0.0820.
  - The tool: `c50_a4bis_verdict.py`, which is not registered.
  - Its null control (BF16 against itself) gives exact zeros.
  - On the A4 outputs, P0 scores 0.0735, CI [0.0646, 0.0825], which is just over the bound. PC
    scores 0.1515, CI [0.1329, 0.1706].

**Why that gate is the wrong coordinate:**
- floor_top1 measures how often a pure numerics change (another build of llama.cpp) flips the
  greedy choice of one artifact. It is a fidelity bound for an engine. Applied to a quantization, it
  counts flips between near-tied tokens as damage.
- P0's top-1 accuracy is 0.6723 against BF16's 0.6726: P0 flips 7.3% of choices without losing
  accuracy. A gate that fails it would measure difference, not degradation.

**Why the document statistic is blind on this text: evidence from three cells.**
- **A2b's two numerics-only pairs both gave negative point dBPB:** −0.0022 for the build change
  and −0.0014 for the micro-batch change (A2b result table). Their CIs touch zero (upper +0.0003
  and +0.0009). Even a pure numerics perturbation does not read as a loss here.
- **The tail is not the whole story.** Outside BF16's overconfident tail, PC differs from BF16 by
  +103 bits over 589,592 bytes, which is +0.0002 BPB. Q2_K barely moves the likelihood of these
  documents at all.
- **The likely cause is the protocol's own caveat:** FIT text (PG19, The Stack, public markdown) is
  very probably in Ling-mini's pretraining data. On memorised text, likelihood is insensitive to
  quantization, so fresh FIT ranks would very probably share the defect.

**Direction for A4-bis (recommendation; the protocol fixes it before any run):**
- **Text the model cannot have seen,** written after Ling-mini-2.0's release. The source, licence,
  date filter and selection rule are pinned before any of it is scored.
- **The document gate unchanged** (dBPB upper CI95 ≤ +0.02, roadmap §2), so the roadmap's quality
  coordinate stays where it was.
- **The planted Q2_K must fail it first.** If Q2_K still passes on new text, the plant is too weak
  for this model, and a coarser plant is registered before P0's result is read.
- **Reported, not gated:** top-1 disagreement, top-1 accuracy, mean |Δ log p| and the tail.
- **G6 chat:** reported only. BF16's own ability sets that outcome, whatever the candidate.
- **The cost:** a new corpus download (authorized), plus ~3 h of CPU for three arms on 72 documents.

**What runs meanwhile:** S0 still waits. The S2 builds and arms do not depend on T0's qualification,
and they run in the free slot.
