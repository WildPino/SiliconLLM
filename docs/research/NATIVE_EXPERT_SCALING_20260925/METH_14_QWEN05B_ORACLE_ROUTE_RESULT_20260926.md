# METH-14 result: local-mass oracle helps but fails the quality gate

**Decision:** stop router-only repair of METH-13's Qwen2.5-0.5B
E128/top-32 donor-channel geometry under the frozen diagnostic. The
[prospective protocol](METH_14_QWEN05B_ORACLE_ROUTE_PROTOCOL_20260926.md)
set the +0.40 BPB decision before the run. The [machine result](meth14_qwen05b_oracle_route.json)
and [runner](../../../benchmarks/donor_adaptation/s1/meth14_qwen05b_oracle_route.py)
bind and reproduce this CPU-only comparison.

## Exact comparison

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth14_qwen05b_oracle_route.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth14_qwen05b_oracle_route.json --assets results/native_expert_scaling/meth13_qwen05b
```

The run rechecked the pinned Qwen2.5-0.5B revision and safetensors hash,
METH-13 label/router NPZ hashes, tokenizer fingerprint and the same
2×512-token, 4,396-byte heldout ID hash. Six CPU threads ran fp32/eager;
no GPU was used. The all-groups wrapper retained donor-logit identity
within **1.72e-5** maximum absolute error. Intact donor and fitted
top-32 BPB reproduced METH-13 exactly within the required 1e-4.

| Arm on identical token IDs | BPB | Delta to donor |
|---|---:|---:|
| Intact pretrained donor | 1.083022 | — |
| Fitted input-only route, 32/128 groups | 2.443496 | +1.360474 |
| Current-activation local-mass oracle, 32/128 groups | 1.858748 | **+0.775726** |

The oracle improves **0.584748 BPB** over the fitted router, so route
error is a material part of the failure. Its donor-relative damage
still exceeds the frozen +0.40 cutoff by **0.375726 BPB**. The sparse
oracle changes first-window logits by up to **19.52** absolute. Runtime
was **19.391 s** and process RSS at exit **4.607 GB**, not a peak-RAM
certificate.

The oracle computes the full post-SwiGLU vector at each sparse
trajectory, scores each group's squared activation mass and keeps
the largest 32 groups. It has no viable native sparse-inference cost.
It maximizes a **local** mass criterion, not whole-model BPB. Therefore
the result rejects a router-only continuation **under this fixed local
oracle and threshold**; it does not prove that no learned routing rule
could do better. The diagnostic slice is reused internal data, not a
fresh quality estimate.

The next transfer design should change the FFN architecture or training
path, start from exact donor behavior, and train distinct conditional
experts with a compact shared contribution. It must verify end-to-end
quality after dense-to-sparse transition and price the head plus
sublinear CPU routing/LUT at increasing E. Neither METH-13 nor METH-14
trained experts, exported a native artifact, measured accepted-token
rate, or established the proposed 10B→100B scaling behavior.
