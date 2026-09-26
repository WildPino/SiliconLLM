# METH-07 result: expert precision has priority, but both organs matter

**Verdict: DIAGNOSTIC_ONLY; experts meet the frozen rescue priority.**
Restoring Q4_K to routed/shared experts while keeping MLA IQ2_XS rescued
**0.123819 BPB** of METH-06's +0.239465 BPB loss. Restoring Q4_K to MLA
alone rescued **0.119356 BPB**, narrowly below the predeclared 0.120
priority threshold. The [machine decision](meth07_organ_rescue_summary.json)
therefore selects experts as the first organ for a new priced representation.
The MLA effect is also large; the two interventions cannot be added as
independent losses. Neither arm is eligible for the native target because
both exceed the 540 MB/token active-byte ceiling. This is no donor-quality
or `engine.c` speed pass.

## Frozen apparatus and conversion

The [protocol](METH_07_ORGAN_PRECISION_ABLATION_PROTOCOL_20260926.md)
fixed both interventions and the 0.120 rescue criterion before scoring.
Both arms used the same bound GigaChat source BF16 SHA-256
`fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`,
the same 231-chunk BF16 imatrix SHA-256
`7a881fc7bb821388bbb019bca3cc2047272e0a1b5985020427a0302b7cfa8cbc`,
and six local Ryzen 5 3600X CPU threads. The [planner](../../../benchmarks/native_expert_scaling/plan_meth07_organ_precision.py)
changed exactly 104 MLA IQ2_XS tensors to Q4_K for one arm, or exactly
75 routed and 75 shared IQ2_XS expert tensors to Q4_K for the other.
The remaining tensors, head and first dense FFN were unchanged.

| Arm | Quantizer time | GGUF bytes / SHA-256 | Actual active bytes/token | 540 MB gate |
|---|---:|---|---:|---|
| MLA Q4_K | 1,303.666 s | 3,371,201,824 / `fc45b530acc1980c2e85a25dee0d8af995bfdbc78f02643bc65d3a23b91314b7` | 696,863,840 | fail by 156,863,840 |
| Experts Q4_K | 347.076 s | 5,829,162,784 / `c5e77242eb60372d556beb42938c4eea29a4bbedbb3f8836c021c61c5c7a3741` | 735,624,800 | fail by 195,624,800 |

Both conversions exited 0 under the 40-minute, 50-GB and 8-GB stops.
Observed resident snapshots were below 22 GB for MLA and 19 GB for
experts, not measured process-wide peaks. The [MLA log](meth07_mla_q4k_quantize.log)
and [expert log](meth07_experts_q4k_quantize.log) show the merged 231-chunk
matrix loaded and no fallback/missing-matrix warning. Independent
[MLA header](meth07_mla_q4k_verify.json) and
[expert header](meth07_experts_q4k_verify.json) audits found all 414
planned types, no missing/extra tensors, and exact stored/active bytes
matching their [MLA](meth07_mla_q4k_plan.json) and
[expert](meth07_experts_q4k_plan.json) descriptor preflights. The GGUFs
remain local under `results/native_expert_scaling/`; hashes and compact
evidence are committed here.

## Paired source-ID quality

Both arms used the same frozen nine heldout documents, 11,091 source IDs
and 36,855 UTF-8 bytes, BOS=1 and KV reset per document. Each new GGUF
was bound by its SHA-256 before scoring. The [MLA scores](meth07_mla_q4k_pilot9_scores.jsonl),
[expert scores](meth07_experts_q4k_pilot9_scores.jsonl),
[source BF16 scores](meth06_source_bf16_pilot9_scores.jsonl) and
[published Q4 control](meth05_prior_q4_pilot9_scores.jsonl) share IDs,
categories and byte/token counts. The per-arm [MLA adjudication](meth07_mla_q4k_pilot9_adjudication.json)
and [expert adjudication](meth07_experts_q4k_pilot9_adjudication.json)
record all category scores and input hashes.

| Arm | Pooled BPB | Delta versus BF16 | Rescued from original IQ2 loss | Code delta | Prose delta | Technical delta |
|---|---:|---:|---:|---:|---:|---:|
| Source BF16 | 0.668119 | 0 | — | 0 | 0 | 0 |
| Published Q4 | 0.680547 | +0.012428 | — | +0.013449 | +0.011328 | +0.012508 |
| METH-06 all IQ2 map | 0.907584 | +0.239465 | 0 | +0.243515 | +0.251165 | +0.223715 |
| MLA Q4_K | 0.788228 | +0.120110 | **0.119356** | +0.099509 | +0.136372 | +0.124449 |
| Experts Q4_K | 0.783765 | +0.115646 | **0.123819** | +0.128238 | +0.123633 | +0.095068 |

The borrowed METH-05 adjudicator labels both Q4 arms
`CONTINUE_TO_FULL_HELDOUT` because neither triggers its old +0.20 gross
failure rule. That label does **not** override the METH-07 protocol:
both arms fail the active-byte gate, so no 96-document, task, rollout or
native-speed escalation was run. The nine-document screen diagnoses an
organ-level effect, not general donor-quality retention. Restoring MLA
helps code more; restoring experts helps the technical/general subset
more. Those category differences and the small nine-document sample limit
any claim of universal organ sensitivity.

## Decision

Prioritize a cost-qualified expert representation or learned correction,
then test the whole artifact against the donor. MLA compression remains
material and must be rechecked after any expert change; the two rescue
values are not additive. METH-08 prices one explicit expert/head/dense
precision reallocation under 540 MB/token and freezes a stricter pilot
quality screen. If it fails, do not spend another large conversion on a
nominal bit-count change alone without a new mechanism or diagnostic.
