# Source coverage capture and adoption result

15 October 2026. **CLOSED / source supervision adopted; goal incomplete.**

The frozen source-only capture completed all432 planned cases:360 FIT and72 DEV,
30/6 per each of12 sources. It produced76,512 new label positions plus274 cached
control positions. Worker and holder exited0 in7588.719/7620.141 seconds.
The873 output files occupy10,086,953,614 bytes on the external results volume.
The two cached controls match every generated ID and BF16 score word.

The separate CPU auditor, frozen at225a5c6, exited0 after485.047 seconds,
with245,190,656 bytes OS peak and zero GPU calls. It verified all5,032,324,082
BF16 score words for finiteness and first-index argmax, complete432-case order,
canonical tokenization/decode, split/group provenance, both cached controls,
all411 current raw source BF16 tensor ranges,873 output files and8390 input
pins before/after. Decision: `INDEPENDENT_STORED_NUMERIC_PASS`.

GPU model identities/versions, MuP, original F32-to-BF16 losslessness, actual
GPU source execution and release before posthash remain producer witnesses;
the CPU audit did not independently replay the source. The completion journal's
`original_posthash_complete=false` is preserved because that journal predates
the later successful posthash. No original failure is replaced by a replay.

Producer OS peak was3,764,826,112 bytes; GPU peak allocated6,515,751,936 and
reserved9,850,322,944 bytes, within frozen limits. No learner forward or optimizer
step occurred. Deferred40 LongAlign FIT/DEV cases and all RESERVED remain
untouched. Falcon scores and token IDs are reusable Falcon supervision, not
labels for a different tokenizer or model. Corpus texts/split groups can be reused.

This adoption establishes stored numeric and custody checks. It admits no
learner transfer, semantic quality, CPU token rate, physical DRAM or useful
large-model capacity. Do not repeat the failed D512 replay unchanged.

- [Producer report](chatbot_source_coverage_result_20261015.json)
- [CPU audit](chatbot_source_coverage_audit_20261015.json) and [frozen audit protocol](CHATBOT_SOURCE_COVERAGE_AUDIT_20261015.md)
- [Session closure and receipt hashes](chatbot_source_coverage_closure_20261015.json)
- [Independent strategy and physical partition correction](../PIPELINE_REASSESSMENT_20261015/INDEPENDENT_REVIEW.md)

Next: qualify exact packed parent-to-child transfer and complete active-work
cost before selecting a conditional-transfer pilot. The family path must address
its protected core/head/router cost at very large scale; a small pilot alone
cannot satisfy the goal. Native MoE within one family is fully acceptable.
