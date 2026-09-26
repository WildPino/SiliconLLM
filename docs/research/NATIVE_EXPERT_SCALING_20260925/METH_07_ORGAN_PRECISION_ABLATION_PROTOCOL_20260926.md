# METH-07 frozen protocol: locate GigaChat IQ2 quality loss by organ

## Question and decision

METH-06 passed calibration exposure (minimum 94) and actual active payload
(534,024,800 bytes/token) but lost +0.239465 BPB on the frozen nine-document
pilot. Is most of this loss associated with aggressive IQ2 precision in the
MLA attention matrices, or in the routed/shared experts? This is a bounded
counterfactual diagnosis, **not** a search for a passing target. Both planned
arms exceed the 540 MB/token native payload ceiling and cannot be promoted
even if their pilot quality is good.

The [planner](../../../benchmarks/native_expert_scaling/plan_meth07_organ_precision.py)
starts from the exact [METH-04 map](meth04_gigachat_tensor_types.txt), SHA-256
`a55334bf64f39a506efba1350a3eaae034f6d4453de47d5a2e5e227b7bb44128`,
and the same bound BF16 source SHA-256
`fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`.
It changes only the specified IQ2_XS tensor families to Q4_K. All control,
router, head, embedding, dense first FFN and Q4_0 `attn_k_b` tensors remain
fixed. The same [231-chunk BF16 imatrix](meth06_merged_231chunks_audit.json),
SHA-256
`7a881fc7bb821388bbb019bca3cc2047272e0a1b5985020427a0302b7cfa8cbc`,
is used for both arms. Its original and Cyrillic contexts are distinct and
exclude heldout. Do not recollect calibration for this diagnostic.

| Arm | Changed tensors | Planned stored tensor bytes | Planned active bytes/token | Over 540 MB ceiling |
|---|---:|---:|---:|---:|
| [MLA Q4_K](meth07_mla_q4k_plan.json) | 104 | 3,365,116,160 | 696,863,840 | 156,863,840 |
| [Experts Q4_K](meth07_experts_q4k_plan.json) | 150 | 5,823,077,120 | 735,624,800 | 195,624,800 |

The [MLA map](meth07_mla_q4k_tensor_types.txt) SHA-256 is
`8aa8ed338930b97056ce3b512b556f6330c232767cbfdccd12860670c85f9033`;
the [expert map](meth07_experts_q4k_tensor_types.txt) SHA-256 is
`760b458fb094a355d8103da7ff0e6b5376b8c763f341564a3d945a94b6d41461`.
These are descriptor calculations, not measured conversion or DRAM traffic.
At a favorable 40 GB/s they already imply 17.42 and 18.39 ms/token for
addressed weight payload, leaving too little for a complete 20 ms step.

## Run and stop rules

1. Run the pinned `llama-quantize` from BF16 with the merged imatrix and
   exact map, first MLA Q4_K then experts Q4_K. Use six local CPU threads,
   no T4/GPU. Stop each conversion at **40 minutes wall**, **50 GB resident
   RAM** or **8 GB output**; preserve partial logs and do not score a partial
   GGUF. The prior METH-06 conversion took 22 min 33 s and about 21 GB
   resident; this prices the new arms. Abort the second conversion if the
   first breaches its resource ceiling.
2. Re-read each complete GGUF header against all 414 intended types,
   stored bytes and active bytes in its preflight. Missing tensors, a type
   mismatch, unexplained fallback or byte mismatch stops that arm. Record
   that both arms **fail** the 540 MB cost ceiling regardless of their
   diagnostic header pass.
3. Score each surviving arm with the source-ID GGUF scorer on the
   [same nine-document selection](meth05_pilot9_selection.json), SHA-256
   `38298c84038b6f26a9c40f47347d5cf930c89ad2bc089e64057f5125c79bb2ac`,
   11,091 source IDs/36,855 UTF-8 bytes, BOS=1 and document KV reset.
   Bind each GGUF SHA-256 before scoring. Budget **20 minutes and 25 GB
   resident** per score. Reuse [source BF16 scores](meth06_source_bf16_pilot9_scores.jsonl),
   SHA-256 `c37ed1c7364d0c75d081486769ce413c3ca1eb791d4f56398c2609c28f3f7757`,
   and [Q4 control](meth05_prior_q4_pilot9_scores.jsonl). No 96-document,
   task, rollout or C speed test follows these over-budget diagnostic arms.

## Interpretation fixed before candidate scoring

Compute each arm's pooled and category BPB delta versus the same BF16
reference, plus rescue = `0.23946509328236 - arm_delta`. If one arm rescues
at least **0.120 BPB** (approximately half the original loss), mark that
organ family as a priority for a new cost-qualified representation or
learned correction. If both do, both are priorities; if neither does,
precision loss is distributed or interacting and a single-family fix is
unsupported. This is a prioritization threshold, **not** a quality pass.
Do not infer additive contributions from the two counterfactuals. Any
future target must be priced under the active budget, evaluated on all
96 heldout documents with uncertainty plus generation/tasks, and finally
validated for donor-relative quality and ≥50 accepted tok/s in `engine.c`
on one artifact. Native learned large-E CPU routing/LUT quality remains a
separate requirement.
