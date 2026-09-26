# METH-24: packed R8 Qwen core exported

Command: `.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth24_export_r8_core.py --out results/native_expert_scaling/meth24_qwen05b_r8_core.safetensors --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth24_r8_core_export.json`.

The packed-only artifact SHA-256 is
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`.
It is **496,122,224 bytes**, with 459 tensors: 169 signed-int8
matrices, their 169 fp32 row-scale vectors, and 121 fp32 norm/bias
vectors. The tied embedding/output head is one matrix. All tensor
contents and metadata passed exact reload equality. The file has no
uncompressed matrix copy. It binds the BF16 donor hash and the separate
factor-0.50 E128 adapter hash in metadata.

The matrix codes occupy **493,961,216 bytes**: tied head 136,134,656,
FFN 313,786,368, attention 44,040,192. Row scales occupy 1,824,256
bytes. This is physical file storage, not a measured DRAM-read rate.
Maximum per-matrix relative L2 reconstruction error was 0.009134
for head, 0.014944 for FFN, and 0.014843 for attention. RTX 3060
export took 9.7 s and 2.863 GB peak allocated GPU memory.

**Decision:** artifact identity and storage pass. Independent quality
must reconstruct from these stored codes/scales; native parity and
accepted-token throughput remain unmeasured.
