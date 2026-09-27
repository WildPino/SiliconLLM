# METH-82: packed grouped-Q4 FFN core passes exact export/readback

**Decision.** The [precommitted export gate](METH_82_Q4_FFN_EXPORT_PROTOCOL_20260927.md)
passes. The bound Qwen2.5-0.5B-Instruct donor is stored as a
527,375,264-byte safetensors core with BF16 tied head and attention,
FP32 controls, and packed signed grouped-Q4 FFN. This core is eligible
for fresh development composition with the METH-56 E128 adapter;
its model quality and native rate are untested.

The exporter/protocol were committed at `931cb49` before reading donor
weights for this run. The [exporter](../../../benchmarks/donor_adaptation/s1/meth82_export_q4_ffn_core.py)
uses one FP16 scale per 64 FFN input weights per row, quantizes against
the stored scale and packs two codes per byte. The generated artifact is
`results/native_expert_scaling/meth82_qwen05b_instruct_q4ffn_core.safetensors`,
SHA-256 `fb6e917e949cb88e63dac276748f844dd94d0c51274a8214ff249bf97989a5bf`.
Its [export ledger](meth82_q4_ffn_core_export.json) binds the donor and
METH-56 E128 checkpoint hashes and records every organ. A separate
file hash read reproduced the stored SHA.

| Stored organ | Matrices/controls | Payload bytes | Largest matrix reconstruction error |
|---|---:|---:|---:|
| BF16 tied head | 1 | 272,269,312 | 0 |
| Grouped-Q4 FFN | 72 | 166,699,008 | relative L2 0.124968; max absolute 0.053711 |
| BF16 attention | 96 | 88,080,384 | 0 |
| FP32 controls | 121 | 286,208 | 0 |

Stored tensor payload totals 527,334,912 bytes; header and alignment
add 40,352 bytes. All 362 stored tensors reload byte-exactly, and
independent unpacking of every saved Q4 FFN matrix reproduces the
exporter's BF16 reconstruction exactly. The run took 10.11 seconds
after device initialization, peaked at 72.3 MB allocated GPU and
3.941 GB process RSS; no T4 was used.

The maximum 12.5% matrix relative error is a reason to test the
composed model, not a quality verdict. The ideal addressed total
including product keys and selected factors remains 535,739,904
bytes/token from METH-81. Physical DRAM traffic, dequantization
cost, CPU LUT operation, full `engine.c` parity, semantic quality
and accepted-token throughput remain open.
