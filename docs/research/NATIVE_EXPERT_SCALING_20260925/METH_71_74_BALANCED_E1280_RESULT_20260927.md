# METH-71–74: balanced E1280 learns broad routes but fails joint promotion

**Decision.** METH-71's fixed 64-update development gates pass for both
E128 and the 10× E1280 bank. On a new external set, E1280 passes
document, prompt, generation and PIQA checks, but its preregistered
factor-pair utility threshold fails by 0.000257 BPB. An additional
blinded semantic diagnosis finds one more severe unsupported claim than
the donor. The E1280 checkpoint is retained as a research artifact;
it is **not** quality-promoted or integrated into the native engine.

## Bound method and data

The [METH-71 protocol](METH_71_BALANCED_PRODUCT_KEY_PROTOCOL_20260927.md)
was committed at `f39da3f` before any training. A pre-inference
exclusion-count assertion was corrected at `dee1c98` without changing
the objective or viewing results. The [trainer](../../../benchmarks/donor_adaptation/s1/meth71_balanced_e128_e1280_smoke.py)
uses the original dense product-key initialization, BF16 donor, FP32
rank-8 independent A/B factors, rank-64 router, top-four routing,
dense GPU AdamW, two raw/two chat microbatches, KL weights 2/4, and
a fixed product-axis balance penalty of 0.02 summed over 24 layers.
Both arms have identical training draws. The 24
[development prompts](meth71_balanced_dev_manifest.json), SHA-256
`4bb5eb221c0754e6769f0faff5f8487d7f20cfa5dac029531ad174cede7840a8`,
exclude previous development and the raw rows sampled by METH-70.

| Fixed update-64 check | E128 | E1280 |
|---|---:|---:|
| Donor-position top-1, new prompts | 3,719/3,813 = 97.535% | 3,750/3,813 = 98.348% |
| Held-out raw ΔBPB | −0.003204 | −0.001785 |
| Minimum changed B slots per layer | 128/128 | 1,219/1,280 |
| Minimum selected slots per layer | 128/128 | 1,076/1,280 |
| Worst maximum/mean route load | 4.06× | 19.64× |
| Peak allocated GPU | 2.582 GB | 10.253 GB |
| Full run including save/hash | 151.0 s | 233.0 s |

All precommitted development gates pass. The
[E128 result](meth71_balanced_e128_result.json) was committed at
`42af643` before the E1280 run; the
[E1280 result](meth71_balanced_e1280_result.json) at `f1208a8`.
The checkpoint SHA-256 values are
`f50a72f1ab86bc304fc6cb23e167a719131b3b79ba8a44201b1e98d716d4c98c`
and
`ead764d1f258819f2ee2d7e6af7995d329a078e360f06f085325c96ac6437fbb`.
The E1280 checkpoint is 5,302,760,956 bytes including optimizer and
resumption state. All evaluated product-key top-four sets match
exhaustive pair-score top-four sets. The development prompt set is
now viewed and cannot serve as a new confirmation set.

## External quality and route utility

The [METH-72 protocol](METH_72_E1280_EXTERNAL_PROTOCOL_20260927.md)
preceded source selection and inference. Its
[manifest](meth72_e1280_external_manifest.json), SHA-256
`11621dce9638aae01cc3bd8c76730de7676e4bbb704271056f164b9ad4d5b437`,
contains 8 code, 8 PG19 prose and 8 technical/general documents,
disjoint by source ID and fragment checks from prior external audits.
All 24 excerpts passed a
[pre-inference answerability screen](meth72_answerability_screen.json).
The exact METH-71 E1280 checkpoint was evaluated on the local RTX 3060.

| External measure | Donor | E1280 | Fixed outcome |
|---|---:|---:|---|
| Pooled document BPB | 1.217692 | 1.216787 | Δ−0.000905; pass |
| Prompt top-1 | — | 4,093/4,284 = 95.542% | pooled and each category pass |
| Greedy EOS, 24 prompts | 21 | 23 | pass; no repeated 8-gram 3× in either arm |
| PIQA accuracy, 1,838 items | 1,291 correct | 1,293 correct | +0.109 points; paired fifth percentile −0.326 points; pass |
| Changed B slots per layer | — | minimum 1,219 | pass |
| BPB after one fixed factor-pair permutation | trained 0.969470 | permuted 0.971213 | +0.001743 < +0.002; **fail** |

External prompt routing selected at least 1,046/1,280 slots per layer;
worst maximum/mean load was 22.04×. These are diagnostics, not new
external load gates. The [raw paired result](meth72_e1280_external_audit_result.json),
SHA-256 `f43658cd8c8e1a791237a7bffa1203f518cc6836a7dab4f725b3d2fb0bef2696`,
holds document nats, per-position counts, all paired responses, every
PIQA outcome and route load. Runtime was 623.6 s, GPU peak 5.139 GB
and process RSS 6.105 GB. METH-72's automatic decision is
`stop_instruct_parent_promotion`; its threshold is not revised.

The [METH-73 protocol](METH_73_ROUTE_UTILITY_DIAGNOSTIC_PROTOCOL_20260927.md)
fixed eight further permutation seeds before rerunning. All eight
permutations worsened BPB on both source sets:

| Diagnostic source | Median ΔBPB, permuted minus trained | Range |
|---|---:|---:|
| 24 external documents | +0.000945 | +0.000812 to +0.001021 |
| Four raw held-out windows | +0.001737 | +0.001504 to +0.002638 |

The [METH-73 result](meth73_route_utility_diagnostic_result.json),
SHA-256 `5c1f8ec56df42c076f2cd04909fdc2a0b097038bcc0ddc39296d36623576d35c`,
supports a small consistent functional signal in the learned
factor-to-route pairing. It also shows that the METH-72 +0.001743
value was near the eight-seed raw median. This diagnosis cannot
replace the failed METH-72 gate.

## Arm-blind semantic diagnosis

The [METH-74 protocol](METH_74_BLIND_SEMANTIC_DIAGNOSTIC_PROTOCOL_20260927.md)
was fixed after the automatic failure, without inspecting labeled
continuations. The [A/B file](meth74_blind_semantic_review.json) was
committed before review. The single-reviewer
[verdict](meth74_blind_semantic_verdict.json) and its
[transcription](../../../benchmarks/donor_adaptation/s1/meth74_blind_verdict_build.py)
were committed at `fbabc31` before unblinding. Each counted claim has
a short limiting or contradicting excerpt citation. The
[unblinded score](meth74_semantic_score.json), SHA-256
`ad5e166953417715a5a734a726fe0a083f4ad2d893499db1553775ecffe38ed3`,
reports:

| Excerpt-grounded finding | Donor | E1280 |
|---|---:|---:|
| Unsupported factual claims | 30 | 26 |
| Severe unsupported named-entity, numerical, comparative or causal claims | 7 | 8 |
| Missing requested specific detail | 0 | 0 |

The severe-claim criterion fails. Examples in the E1280 responses
include treating a fallback `default=0` as an observed maximum,
inventing a menagerie manager, and identifying a producer commit hash
as a protocol ID. This is one reviewer on 24 short excerpts; the
counts are a diagnostic, not an estimated population rate.

## Method consequence

The balance objective has addressed the METH-70 collapse on a new
development set: E1280 top-1 rises from METH-70's 92.865% on its
different prompts to 98.348%, and selected-slot minimum rises from
474 to 1,076. These figures are **not** a paired same-prompt effect
estimate, because the training rules and development prompts differ.
The direct METH-71 comparison at E128/E1280 supports broad routing
without donor top-1 loss at 10× expert count. METH-72/74 show that
this is still too weak to claim useful donor-relative response quality
under all fixed gates. A next training study must increase learned
route utility while retaining the balanced coverage, then use new
development and external sources. CPU LUT and full native accepted
token rate have not been measured for this E1280 checkpoint.
