# METH-08 frozen protocol: reallocate precision to GigaChat experts

## Question, causal basis and priced candidate

METH-06's IQ2 map met 534,024,800 active bytes/token but lost +0.239465
BPB on the nine-document donor-relative pilot. [METH-07](METH_07_ORGAN_PRECISION_ABLATION_PROTOCOL_20260926.md)
restored one tensor family at a time: expert Q4_K rescued **0.123819 BPB**,
above its frozen 0.120 priority threshold, while MLA Q4_K rescued 0.119356,
just below. Both Q4 diagnostics exceeded the native active-byte limit.
The uncertainty now is whether a smaller expert precision increase, paid for
by lower precision in the output head and first dense FFN, can preserve
enough donor quality **within** the active-byte ceiling. A higher nominal
bit count alone does not establish better quality; Q2_K and IQ2_XS use
different quantizers, so the actual paired score decides.

The [planner](../../../benchmarks/native_expert_scaling/plan_meth08_q2_reallocation.py)
starts from the exact METH-04 414-tensor map, SHA-256
`a55334bf64f39a506efba1350a3eaae034f6d4453de47d5a2e5e227b7bb44128`.
It changes 75 routed and 75 shared expert tensors from IQ2_XS to Q2_K,
the output head from Q3_K to Q2_K, and three first-block dense FFN tensors
from Q4_K to Q2_K. MLA, router, controls and embedding remain identical
to METH-06. The [exact new map](meth08_q2_reallocation_tensor_types.txt)
SHA-256 is
`9d6b10b7b33ba49f499c2b560647773d0470cc937573100002aa60fcf41f1331`.
The [descriptor plan](meth08_q2_reallocation_plan.json) predicts
**533,140,064 active bytes/token**, 6,859,936 below the 540 MB gate, and
3,546,992,384 stored tensor bytes. The implied addressed-payload floor is
13.3285 ms/token at 40 GB/s, leaving about 6.67 ms of a 20 ms full-step
budget before any claim of ≥50 accepted tok/s. This is arithmetic, not a
DRAM or engine.c measurement.

The source BF16 GGUF SHA-256 is
`fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`.
Use the same [231-chunk BF16 imatrix](meth06_merged_231chunks_audit.json),
SHA-256
`7a881fc7bb821388bbb019bca3cc2047272e0a1b5985020427a0302b7cfa8cbc`.
Q2_K does not require an output-head imatrix entry; that entry is absent
because the calibration ran without perplexity. Do not silently substitute
an IQ2 output head or recollect a different matrix. The same source
tokenizer and heldout identities from METH-05/06 apply.

## Ordered gates and resource stops

1. Quantize the bound BF16 source with pinned llama.cpp `5b335f413`, the
   merged BF16 imatrix and the exact map, six local CPU threads, no GPU/T4.
   Stop at **40 minutes**, **50 GB resident RAM** or **5 GB output**;
   preserve any partial log. The METH-06 IQ2 conversion took 22 min 33 s;
   the METH-07 Q4 diagnostics took 21 min 44 s and 5 min 47 s. The
   different Q2_K mix has no measured runtime yet.
2. Independently read the completed GGUF header. Require all 414 planned
   tensor types, no missing/extra tensors, exactly 3,546,992,384 stored
   tensor bytes and 533,140,064 active bytes/token. Any mismatch, fallback
   or missing-matrix warning stops quality scoring. Passing this gate is
   a cost eligibility check, not native C throughput.
3. Bind the GGUF hash and score the frozen nine heldout documents with
   source-tokenizer IDs, BOS=1, document KV reset. Use the same
   [selection](meth05_pilot9_selection.json), 11,091 IDs and 36,855 bytes,
   and [source BF16 scores](meth06_source_bf16_pilot9_scores.jsonl), SHA-256
   `c37ed1c7364d0c75d081486769ce413c3ca1eb791d4f56398c2609c28f3f7757`.
   Score budget: **20 minutes and 25 GB resident**. A pilot with pooled
   candidate−BF16 **>+0.05 BPB** or any category **>+0.10 BPB** stops
   this map. These thresholds are internal escalation screens, stricter
   than METH-05's original +0.20 gross-failure rule and fixed before
   METH-08 scoring. Nine documents cannot certify quality.
4. Only if the pilot screen passes, score all frozen 96 heldout documents
   with paired BF16/candidate source IDs and the existing stratified
   bootstrap (20,000 draws, seed `20260916`). Require one-sided CI95 upper
   candidate−BF16 **≤+0.02 BPB**, plus PIQA correct
   ≥ceil(0.98×BF16 correct) and fewer than three candidate-only
   catastrophic rollouts. Before this larger run, bind its exact expected
   token count and resource ceiling from the local corpus and pilot rate.
   A pass on one internal corpus remains family-specific; external/fresh
   evidence and native `engine.c` fidelity/rate are still required.

## Interpretation

If the compact map fails the pilot, report whether the expert Q2_K swap
failed to recover quality, or whether the head/dense precision payment
erased that recovery. The present bundled map cannot separate those two
mechanisms causally; a further isolated arm would be required. Do not
weaken the active-byte or quality gates after observing the result. If
the map passes quality, it is still only a GGUF donor-format candidate:
GigaChat operators, tokenizer behavior and the exact weight format must
work in the native C engine, with donor-relative quality and ≥50 accepted
batch-1 tok/s verified on the same exported artifact. The CPU LUT/router
and useful quality with much larger learned expert counts remain open.
