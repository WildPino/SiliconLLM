# METH-81: grouped-Q4 FFN core fits the ideal byte allotment

The [precommitted payload screen](METH_81_Q4_CORE_PAYLOAD_PROTOCOL_20260927.md)
was recorded at `39c73b8`. It read the header of the exact
Qwen2.5-0.5B-Instruct donor safetensors, SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
All 169 matrix input widths are divisible by the specified group size
64, so no row padding is needed. The tensor census matches METH-59:
one tied head, 72 FFN matrices, 96 attention matrices and 121 FP32
control tensors. The [machine ledger](meth81_q4_core_payload_preflight.json)
contains exact per-tensor shapes and charges.

| Packed-core map | Core bytes | Core + 5,652,480 router + 2,752,512 selected-factor bytes | ≤560 MB? |
|---|---:|---:|---|
| All matrices grouped-Q4 | 262,703,104 | 271,108,096 | yes |
| BF16 attention, Q4 head/FFN | 327,387,136 | 335,792,128 | yes |
| BF16 tied head, Q4 attention/FFN | 462,650,880 | 471,055,872 | yes |
| **BF16 tied head+attention, Q4 FFN** | **527,334,912** | **535,739,904** | **yes** |

The selected leading candidate is the fourth arm, 24,260,096 bytes
below the design allotment. Its FFN matrix rule is signed symmetric
4-bit codes in two-per-byte storage with one FP16 scale per 64 input
weights per row. This read-only calculation completed in 0.812 seconds
after interpreter initialization at 1.246 GB RSS; no GPU/T4 was used.

The ledger counts ideal addressed bytes, **not** saved-file metadata,
dequantization overhead, KV state, physical DRAM traffic or accepted
token rate. It gives no reconstruction or model-quality evidence.
The next gate is an exact packed export/readback followed by fresh
development quality for the composed METH-56 E128 model. Native C
execution and a larger-E quality-valid bank remain separate work.
