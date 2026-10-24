# C50-T1 part 2a result: either half of c48 carries most of the gain, and both halves lose the same facts

24 October 2026. Protocol: [C50_T1_PROTOCOL_20261024](C50_T1_PROTOCOL_20261024.md), addendum 1 (`1672670`, pushed
before either partial merge was built). Verdict record: `results/c50_t1_part2a.json` (sha256 `7bec0461e0f59289…`,
corrected; the first version, `694c7504…`, is in git at `6de3ad0`), written by `c50_t1_verdict.py part2a` from the
records only. Panel runs: `results/c50_t1_part2a_run_attn.json`
(`916398c4…`) and `results/c50_t1_part2a_run_ffn.json` (`f6b54695…`). The artifacts are under
`E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/t1/`.

## Verdict

**No group is clean, so by the registered rule part 2's first T4 arm is the anchored LoRA.** The rule asks for a
partial merge that keeps the gain (R ≥ 0.559) and is not worse than T0 on nq and on tqa. The attention half keeps
the gain but regresses on nq and tqa. The FFN half regresses on both and does not keep the gain.

| Merge (of c48) | Tensors | R_llama, 36 S4 documents (CI95) | keeps gain | nq: T0 / X right (b, c) | tqa: T0 / X right (b, c) | clean |
| --- | ---: | --- | --- | --- | --- | --- |
| full (MVE-0) | 100 | 0.6970 (0.589..0.824) | yes | 174 / 147 (33, 6), **worse** | 292 / 269 (30, 7), **worse** | no |
| attn only | 40 | **0.6783** (0.574..0.801) | yes | 174 / 158 (24, 8), **worse** | 292 / 270 (28, 6), **worse** | no |
| ffn only | 60 | 0.5332 (0.450..0.631) | **no** | 174 / 157 (25, 8), **worse** | 292 / 272 (24, 4), **worse** | no |

b = T0 right and X wrong, c = the reverse; "worse" is A4's paired rule, as in MVE-0. The nq and tqa counts exclude
both halves of the donor-gap items (502 nq items, 483 tqa items).

**Correction, 24 October.** The first version of this record left out S4's half of the donor-gap items only, so
S5's half sat inside the nq and tqa counts, as in MVE-0's first version (see its correction). Those items can only
add to c. With them in, both halves passed nq near the rule's edge. With both halves out, both regress on nq
clearly: b − c is 16 and 17, against limits of 11.3 and 11.5. The decision does not change: leaving items out can
only lower c, so no variant could have become clean.

## Checks

| Check | attn | ffn |
| --- | --- | --- |
| BF16 merge, ‖bf16(W + ΔW) − W‖ / ‖ΔW‖ and cosine, medians | 1.009, 0.991 | 1.003, 0.997 |
| Q4_K_M sha256 | `91673de5…` | `22a79b4a…` |
| L (`layout --only`): header and layout as P0, data differs only in the group's tensors | **PASS**, 40 of 278 differ | **PASS**, 60 of 278 differ |
| After Q4_K_M against P0, ‖ΔQ‖ / ‖ΔW‖ and cosine, medians | 5.36, 0.187 | 3.75, 0.266 |
| Panels nq, tqa, docs | COMPLETE (522 s, 417 s, 2,362 s) | COMPLETE (514 s, 401 s, 2,366 s) |

Both partial T1s have P0's bytes per token, as the full one does. Along ΔW, Q4_K_M keeps about 5.36 × 0.187 ≈ 1.00
(attn) and 3.75 × 0.266 ≈ 1.00 (ffn) of ‖ΔW‖, as in MVE-0.

## The gain does not add up

R(attn) + R(ffn) = 1.21, against 0.70 for the full merge. **Each half alone recovers most of what the two recover
together.** The attention half alone keeps 97% of the full gain (0.678 of 0.697). c48 was trained with both halves
at once, so the two halves learned largely the same correction through different paths. This says where the effect
sits in c48. It does not say what a LoRA trained on attention alone would reach.

## Post-hoc, not registered: the lost items are the same items

`c50_t1_part2a_posthoc.py` (record `results/c50_t1_part2a_posthoc.json`, `fc6d43cb…`) lists, for each merge, the
items T0 gets right and the merge gets wrong:

| Panel | T0 right | lost: full / attn / ffn | lost by both attn and ffn | expected if independent | full's losses also lost by a half |
| --- | ---: | --- | ---: | ---: | --- |
| nq | 174 | 33 / 24 / 25 | **14** | 3.4 | 27 of 33 |
| tqa | 292 | 30 / 28 / 24 | **16** | 2.3 | 25 of 30 |

Two disjoint sets of tensors break the same facts four to seven times more often than chance. **The loss is
item-level and systematic, not noise from moving the weights.** It is not located in one half of the model either.
A shared set of T0-right facts breaks whichever half carries the correction. That points at the training target,
not at which modules are trained, and it is the case for anchoring the loss to T0. One LoRA, one seed: the overlap
is a description of c48, not a law.

## Registered predictions, scored

| Prediction | Measured | Score |
| --- | --- | --- |
| L passes for both, exactly 40 and 60 tensors differ | 40 and 60, PASS | **RIGHT** |
| R_llama attn 0.35-0.55, ffn 0.25-0.45, both below 0.697 | attn 0.678, ffn 0.533; both below 0.697 | **WRONG** (both above their range: I assumed the gain splits between the halves, and it overlaps) |
| The regression sits mostly in the FFN: ffn regresses on both, attn not worse on at least one | ffn regresses on both; attn regresses on both | **WRONG** (ffn's half is right, attn's is wrong: the loss is not in the FFN) |
| No group is clean, because the gain needs both parts, so part 2 starts with the anchored arm | no group is clean; the anchored arm comes first | **RIGHT on the outcome, WRONG on the reason**: attn keeps the gain alone and fails on nq and tqa |

One of four RIGHT outright, one RIGHT on the outcome only. The decision follows the registered rule, not my
reason for predicting it.

## What this says for part 2

- **The first arm is the anchored LoRA, by the rule:** all of c48's modules, with the loss anchored to T0 where T0
  is confident. Its protocol is pushed before any T4 session.
- **Reported for the design, not a change to the rule:** the attention half alone keeps the gain. So if anchoring
  works, an anchored LoRA on attention alone is the cheaper variant to try next, at 40 tensors instead of 100.
- **It does not say** that an attention-only LoRA trained from scratch keeps R 0.68, or that anchoring fixes tqa.
  Both are questions for part 2.
