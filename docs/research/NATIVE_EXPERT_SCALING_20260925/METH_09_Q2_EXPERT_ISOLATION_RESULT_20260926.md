# METH-09 result: Q2_K experts help modestly; precision payment dominates

**Verdict: Q2_EXPERT_IMPROVES_PILOT, diagnostic only.** Changing only
GigaChat's routed/shared experts from IQ2_XS to Q2_K reduces the frozen
nine-document donor-relative loss by **0.023740 BPB**, from +0.239465
to **+0.215725 BPB**. This meets the [predeclared 0.020 diagnostic
threshold](METH_09_Q2_EXPERT_ISOLATION_PROTOCOL_20260926.md). The
head/first-FFN precision payment added in METH-08 then worsens quality
by **+0.059428 BPB** conditional on Q2_K experts, yielding METH-08's
+0.275153 loss. The [machine summary](meth09_q2_isolation_summary.json)
checks the shared BF16/Q4 references and paired denominators. Neither
arm is a useful or native-speed-validated donor transfer: expert-only
costs 562,824,800 active bytes/token, above the 540 MB ceiling, and
its pilot quality remains far from the +0.02 full-quality gate.

## Bound one-variable run

The [planner](../../../benchmarks/native_expert_scaling/plan_meth08_q2_reallocation.py)
with `--experts-only` changed exactly 75 routed and 75 shared expert
tensors from IQ2_XS to Q2_K. The head remained Q3_K and the three
first-block dense FFN tensors remained Q4_K, as in METH-06. Its
[plan](meth09_q2_experts_only_plan.json) and
[type map](meth09_q2_experts_only_tensor_types.txt), SHA-256
`fbc8501f5abecbd0c01bd28963654f6869de1d3da49166b5d3e59661542b5163`,
were fixed before scoring. The source BF16 SHA-256 is
`fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`;
the 231-chunk BF16 imatrix SHA-256 is
`7a881fc7bb821388bbb019bca3cc2047272e0a1b5985020427a0302b7cfa8cbc`.

Pinned `llama-quantize` with six local Ryzen 5 3600X CPU threads exited
0 after **419.068 s** (6 min 59 s), below the frozen 40-minute/50-GB/5-GB
stops. Observed resident snapshots stayed below 22 GB, not a certified
process-wide peak. The [log](meth09_q2_experts_only_quantize.log)
reports the same 231-chunk matrix loaded, with no fallback or missing
imatrix warning. The GGUF is **3,582,762,784 bytes**, SHA-256
`8427c4e2c986d50216cf26aeb5829a57a0d8128793d65e270b59aa1c66978b0a`.
The [independent header audit](meth09_q2_experts_only_verify.json)
found all 414 planned types, no missing/extra tensors, exactly
3,576,677,120 stored tensor bytes and **562,824,800 active bytes/token**.
It fails the 540 MB native-cost gate by 22,824,800 bytes. The GGUF
remains local under `results/native_expert_scaling/`.

## Same source-ID heldout pilot

The hash-bound model scored the same nine heldout documents, 11,091
source-tokenizer IDs and 36,855 UTF-8 bytes, BOS=1 and document KV
reset. The [per-document scores](meth09_q2_experts_only_pilot9_scores.jsonl)
and [paired adjudication](meth09_q2_experts_only_pilot9_adjudication.json)
retain the input hashes and category results.

| Arm | Pooled BPB | Delta versus BF16 | Code delta | Prose delta | Technical delta |
|---|---:|---:|---:|---:|---:|
| Source BF16 | 0.668119 | 0 | 0 | 0 | 0 |
| METH-06 IQ2 base | 0.907584 | +0.239465 | +0.243515 | +0.251165 | +0.223715 |
| **Q2_K experts only** | **0.883844** | **+0.215725** | **+0.224667** | **+0.224042** | **+0.198466** |
| METH-08 Q2_K experts + head/dense payment | 0.943271 | +0.275153 | +0.271735 | +0.300938 | +0.252785 |

Expert-only minus IQ2 is **−0.023740 BPB** pooled; METH-08 minus
expert-only is **+0.059428 BPB**. These are measured conditional
differences on nine documents. The result does not isolate the head
payment from the dense-FFN payment, and it does not establish statistical
precision or general language quality. The reused METH-05 adjudicator
labels the over-budget arm `CONTINUE_TO_FULL_HELDOUT` because its old
gross-failure rule requires every category to exceed +0.20. METH-09's
frozen protocol explicitly forbids full-quality escalation for this
over-budget diagnostic, so no 96-document, PIQA, rollout or native-speed
test was run.

## Decision for the method

The expert Q2_K swap has a small pilot benefit but is not enough to
preserve donor capability, even before paying its extra active cost.
Paying for it by lowering head and first-FFN precision makes the whole
artifact worse. A further quantization-only reallocation with the same
organs is not justified by these measurements. The next candidate needs
a mechanism that changes the quality/cost frontier, such as an explicitly
trained correction or a different conditional representation, with a
prepriced native export path. Separately, useful learned quality and CPU
LUT/router behavior as expert count grows with RAM remain unproven;
these GGUF tests have 64 routed experts and do not establish 10× E
scaling.
