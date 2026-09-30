# METH-211: physical exact-head Q8 core stored and verified

The [protocol](METH_211_EXACT_HEAD_EXPORT_PROTOCOL_20260930.md) and
[exporter](../../../benchmarks/native_expert_scaling/meth211_export_exact_head_core.py)
were frozen at `349ac7e`, after METH-210's fixed candidate passed
development and K64 choice gates. The exporter binds source and
prior-core hashes, copies every METH-193 tensor except the proposal
scales' FP32-to-FP16 conversion, and adds the original BF16 tied matrix.

The local core `results/native_expert_scaling/meth211_qwen05b_exact_head_q8_core.safetensors`
has SHA-256 `4bb34a2ac14bfa3f4fe42490a6755513cf6b08e82e981c86926e374b12353c1a`
and 820,707,464 physical bytes: 820,666,624 array bytes and 40,840
header bytes across 364 tensors. Every tensor and metadata field
passes readback. All 362 unchanged tensors and all 72 FFN code/scale
pairs match METH-193 byte for byte. The exact tied matrix matches
the donor; proposal code/FP16-scale bits match METH-210 hashes.

The [export report](meth211_exact_head_core_export.json) SHA-256 is
`5e48b34dacdf1e3d9f7fb7badb22e63941f064566c115c729fcdd52140a989b5`.
Metadata declares exact embedding lookup, full exact-head likelihood
and the separate K64 greedy proposal/row operation. The proposed
greedy ideal weight ledger remains 559,794,176 bytes/token; this
does not account for future native precision changes, measured
DRAM traffic, activation traffic or a third-level router.

Command:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth211_export_exact_head_core.py --out results/native_expert_scaling/meth211_qwen05b_exact_head_q8_core.safetensors --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth211_exact_head_core_export.json
```

CPU-only export took 17.969 s with six host threads and ending RSS
2.074 GB. No GPU calculation, T4, training or quality text was used.

**Decision:** the compact-core composition is now a real hash-bound
artifact ready for a loader and independent quality/generation audit.
Load all effective weights from this artifact, not from an in-memory
source substitution, and verify donor/BF16 controls before scoring a
new frozen source/task/greedy cohort. Native composition must preserve
this precision map or re-audit the changed map, and measure the same
accepted artifact end to end in the project engine. It has no fresh
quality or >=50 accepted tok/s evidence yet; the distinct large-n
specialist and multi-family transfer requirements remain open.
