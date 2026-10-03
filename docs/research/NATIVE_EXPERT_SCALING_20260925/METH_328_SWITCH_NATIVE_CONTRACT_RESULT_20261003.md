# METH-328: complete native Switch/T5 tiny contract PASS

Freeze49539f0, observations2c696f6. [Raw record](meth328_switch_native_contract_result.json)
SHA25608f04429c9081266c8bad29aaa3c7ffe0f2534c4484e39e3498e5f1f1ea9f330.
[Protocol](METH_328_SWITCH_NATIVE_CONTRACT_PROTOCOL_20261003.md); C source
benchmarks/native_expert_scaling/meth328_switch_f32_reference.c SHA256
312536e500635870e3347b187c659d2dacfd2131209cf236f249fa09a5301244.
Driver meth328_switch_native_contract.py; opt-in engine.c prefix
SILICON_SWITCH_F32_REFERENCE, legacy default preserved. Binary
f814b01efaf367677a8f1d4a1cb480564e5351888ce554e10f4668927e317243.

Complete source-compatible encoder/decoder relative attention/RMS/FFN/router/
probability/capacity/cache/cross-prefill/tied full head implemented with readonly
F32 tensor manifest, no missing weights fallback. Source-sized architecture
support exists but this test uses fixed Tiny synthetic weights only.

Both whole Tiny models(seed328, source6tokens/four cached steps) pass ALL gates:

| Capacity | Encoder pooled relativeL2 | Decoder pooled | Full logit pooled | Max per-state | Route probability maxabs | Dropped source routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|1|1.22266e-7|1.44748e-7|1.86116e-7|2.09290e-7|2.38419e-7|4|
|64|1.27427e-7|1.42325e-7|1.95846e-7|2.10221e-7|1.78814e-7|0|

Every source initial/block/final and every decoder initial/block/final state
<=1e-4, full logits/greedy choices/routes/capacity exact within frozen numeric
limits. All30 causal/bidirectional relative bucket boundary IDs EXACT official.
Both actual cached contexts preserve per-call capacity; no saturated full-prefix
equivalence assumption. All9 whole model faults detected, relative logit errors
.03858..1.0 versus1e-4: zero relative bias, zero crossV, omitted probability,
unlimited cap1, zero head. Explicit no performance inference from tiny work.

4.656s main AFTER imports (startup not measured/bounded by driver); max sampled
combined controller+childRSS1,251,962,880B/end1,249,415,168B. Full logits/state/
fault files/compiler/runtime logs retained locally. Exec17984 exit0. No GPU.

Eligible only for separately frozen actual source full-size/native mapping;
real all-bank source hash availability327 alone cannot replace this missing
full-size numerical check. F32 bridge is an exact operator prerequisite,
not final efficient LUT representation or accepted>=50 artifact. Full quality,
useful larger-n, DRAM/routing/head/prefill/accepted-rate remain required.
