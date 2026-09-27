# METH-60: BF16 attention helps, but no single organ passes ranking and bytes

**Decision:** no preregistered single-organ BF16 restore passes both
the ≥95% pooled top-1 gate and ≤560 million ideal active-byte allotment.
BF16 attention is the strongest feasible arm at **3,914/4,125 = 94.885%**
and **548.320 MB/token**; it misses the ranking gate by five positions.
Do not promote it as a quality-valid compact core.

The [protocol](METH_60_R8_CORE_ORGAN_ABLATION_PROTOCOL_20260927.md)
binds the METH-59 packed Instruct artifact, original Instruct source,
METH-56 adapter and METH-57 prompts. The
[runner](../../../benchmarks/donor_adaptation/s1/meth60_r8_core_organ_ablation.py)
restored one organ at a time from the original BF16 source and kept the
other matrices reconstructed from stored R8 codes. The BF16 adapter
control reproduced METH-57's **3,997/4,125** match count. One initial
apparatus attempt stopped before any R8 arm on a summary KeyError;
`54ed097` fixed the baseline tally without changing arms or thresholds.
The [raw result](meth60_r8_core_organ_ablation_result.json), SHA-256
`22bac64fa592f6e56c77111082ea51fc10e7476db2d37ed9b445e0fc01d28df3`,
contains each prompt's counts and the byte ledger.

| Core representation | Top-1 vs BF16 donor | Added active bytes vs all-R8 | Core + router + selected factors | Joint gate |
|---|---:|---:|---:|---|
| All R8 | 3,887/4,125 = 94.230% | 0 | 504.477 MB | fail top-1 |
| BF16 head | 3,884/4,125 = 94.158% | 135.527 MB | 640.004 MB | fail both |
| BF16 attention | **3,914/4,125 = 94.885%** | 43.844 MB | **548.320 MB** | fail top-1 |
| BF16 FFN | 3,878/4,125 = 94.012% | 312.766 MB | 817.243 MB | fail both |

The ideal bytes include core matrix codes/scales and FP32 controls,
5.652 MB of trained product-key projection/keys across all layers, and
2.753 MB of four selected BF16 A/B expert factors per layer. They exclude
token-dependent KV state, activations, allocator/format overhead and
actual DRAM transactions; this is not measured throughput. The attention
restore improves pooled top-1 by 27 positions over all-R8 and has the
largest observed gain per added byte among these arms. It still leaves
only 11.680 MB under the 560 MB design allotment and cannot be called
native-rate or semantic-quality evidence.

The local RTX 3060 run took 11.078 s, peaked at 2.278 GB allocated GPU
memory and ended at 2.884 GB RSS. The next candidate must reduce
quantization damage without paying the BF16 head or FFN cost. Any
precision map selected using these viewed prompts needs a new,
source-disjoint quality/generation/task and semantic audit before promotion.
