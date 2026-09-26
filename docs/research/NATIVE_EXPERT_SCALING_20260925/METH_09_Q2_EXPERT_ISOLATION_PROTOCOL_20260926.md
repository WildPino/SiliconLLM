# METH-09 frozen protocol: isolate the Q2_K expert swap

METH-08's 533,140,064-byte/token GGUF passed its cost/header gate but
lost **+0.275153 BPB** on the frozen nine-document pilot, worse than
METH-06's **+0.239465 BPB** all-IQ2 map. METH-08 changed experts to
Q2_K while lowering the output head and first dense FFN to Q2_K to pay
for them. The bundled result cannot tell whether expert Q2_K helps and
the payment harms, or whether expert Q2_K itself fails. This diagnostic
changes **only** the 75 routed and 75 shared expert tensors from IQ2_XS
to Q2_K; all other types match METH-06 exactly. It is deliberately
over budget and cannot be promoted.

The [planner](../../../benchmarks/native_expert_scaling/plan_meth08_q2_reallocation.py)
with `--experts-only` emitted the [150-tensor map](meth09_q2_experts_only_tensor_types.txt),
SHA-256
`fbc8501f5abecbd0c01bd28963654f6869de1d3da49166b5d3e59661542b5163`,
and the [preflight](meth09_q2_experts_only_plan.json) predicts
**562,824,800 active bytes/token**, 22,824,800 over the native 540 MB
ceiling, and 3,576,677,120 stored tensor bytes. The source BF16 GGUF,
source tokenizer, 231-chunk BF16 imatrix and nine-document heldout are
the exact METH-05/06 bound inputs. Source BF16, Q4 and all-IQ2 pilot
scores are frozen and reused. No T4/GPU.

1. Quantize the bound BF16 source with pinned llama.cpp `5b335f413`, the
   merged imatrix and this exact map, six local CPU threads. Stop at
   **40 minutes wall**, **50 GB resident** or **5 GB GGUF output**.
   Preserve partial logs; do not score a partial GGUF.
2. Independently verify all 414 actual tensor types, no missing/extra
   tensors, exactly 3,576,677,120 stored tensor bytes and 562,824,800
   active bytes/token. Any fallback, missing imatrix or byte/type
   mismatch stops scoring. Record the known native-cost failure.
3. Bind the completed GGUF SHA-256 and score the same nine heldout
   documents (11,091 source IDs, 36,855 UTF-8 bytes, BOS=1, document KV
   reset), with **20 minutes/25 GB** as score stops. The primary
   diagnostic is pooled BPB delta versus source BF16. Define the
   expert-only effect as `METH09_delta - METH06_delta`; an improvement
   of at least **0.020 BPB** (effect ≤−0.020) supports Q2_K experts as
   useful before payment. A deterioration of at least **0.020 BPB**
   (effect ≥+0.020) rejects that expert swap alone on this pilot;
   intermediate effects are inconclusive. Also compute the conditional
   payment effect `METH08_delta - METH09_delta` and category effects.
   This is a diagnostic decision threshold, not a donor-quality pass.

No 96-document, task, rollout or native C speed test follows this
over-budget arm. A new candidate must still satisfy cost, paired quality
with uncertainty and generation/tasks, plus native quality and ≥50
accepted tok/s on one exported artifact. This test does not establish
learned large-E routing quality or CPU LUT scalability.
