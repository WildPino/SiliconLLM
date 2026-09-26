# METH-08 result: budgeted Q2_K swap loses more donor quality

**Verdict: STOP_PILOT_QUALITY.** The preregistered expert/head/dense
precision reallocation produced a complete GGUF with all 414 intended
types and **533,140,064 active bytes/token**, under the 540 MB ceiling.
On the same nine-document source-ID pilot, however, it lost **+0.275153
BPB** relative to source BF16, worse than the METH-06 all-IQ2 map's
+0.239465. Every category exceeds both this experiment's +0.10 category
screen and the older +0.20 gross-failure screen. Under the
[frozen protocol](METH_08_Q2_REALLOCATION_PROTOCOL_20260926.md), the
96-document, task, rollout and native C speed gates were not run.

## Bound conversion and actual header

The [planner](../../../benchmarks/native_expert_scaling/plan_meth08_q2_reallocation.py)
changed exactly 150 routed/shared expert tensors from IQ2_XS to Q2_K,
the output head from Q3_K to Q2_K, and the three dense first-FFN tensors
from Q4_K to Q2_K. The [plan](meth08_q2_reallocation_plan.json) and
[414-tensor map](meth08_q2_reallocation_tensor_types.txt), SHA-256
`9d6b10b7b33ba49f499c2b560647773d0470cc937573100002aa60fcf41f1331`,
were frozen before candidate scoring. Source BF16 SHA-256
`fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`
and merged 231-chunk BF16 imatrix SHA-256
`7a881fc7bb821388bbb019bca3cc2047272e0a1b5985020427a0302b7cfa8cbc`
are the same METH-06 inputs.

Pinned `llama-quantize`, six local Ryzen 5 3600X CPU threads, exited 0
after **435.336 s** (7 min 15 s), below the 40-minute/50-GB/5-GB stops.
Observed resident snapshots stayed below 22 GB; they do not certify the
process-wide peak. The [quantizer log](meth08_q2_reallocation_quantize.log)
reports 308 matrix entries from 231 chunks and no fallback or
missing-matrix warning. The complete GGUF is **3,553,078,048 bytes**,
SHA-256
`ffb9c8eaf2b72524a961e9405c90b3ee050b2796300fc469f54541661ee514e6`.
The [independent header audit](meth08_q2_reallocation_verify.json)
found 414/414 planned types, no missing/extra tensors, exactly
3,546,992,384 stored tensor bytes and 533,140,064 active bytes/token.
The 6,859,936-byte margin is an addressed-payload calculation, not
measured DRAM traffic or an `engine.c` throughput result. The large GGUF
remains local under `results/native_expert_scaling/`.

## Frozen source-ID pilot

The hash-bound candidate used the same nine heldout documents, 11,091
source-tokenizer IDs and 36,855 UTF-8 bytes as the source BF16 and Q4
controls, with BOS=1 and KV reset per document. The
[candidate scores](meth08_q2_reallocation_pilot9_scores.jsonl),
[runtime record](meth08_q2_reallocation_pilot9_scores.jsonl.runtime.json)
and [paired adjudication](meth08_q2_reallocation_pilot9_adjudication.json)
retain the exact per-document and aggregate evidence.

| Group | BF16 BPB | METH-06 IQ2 BPB | METH-08 Q2 swap BPB | METH-08−BF16 |
|---|---:|---:|---:|---:|
| Code (3) | 0.363524 | 0.607039 | 0.635259 | **+0.271735** |
| Prose (3) | 0.950031 | 1.201196 | 1.250969 | **+0.300938** |
| Technical/general (3) | 0.690800 | 0.914516 | 0.943586 | **+0.252785** |
| **All (9)** | **0.668119** | **0.907584** | **0.943271** | **+0.275153** |

The protocol's pilot screen required pooled delta ≤+0.05 and each
category ≤+0.10 before a costly full evaluation. All four comparisons
miss by a wide margin. The older +0.20 gross-failure rule also stops the
candidate. This evidence rejects **this combined precision reallocation**,
not Q2_K experts in isolation. The cost payment on the head and dense FFN
could erase any expert benefit, so [METH-09](METH_09_Q2_EXPERT_ISOLATION_PROTOCOL_20260926.md)
freezes a single over-budget expert-only diagnostic. Neither result
establishes donor quality, multilingual capability, learned large-E
routing or native speed.
