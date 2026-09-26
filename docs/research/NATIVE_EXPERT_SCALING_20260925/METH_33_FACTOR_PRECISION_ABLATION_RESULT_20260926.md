# METH-33: one ternary factor still misses top-1 fidelity

The [prospective protocol](METH_33_FACTOR_PRECISION_ABLATION_PROTOCOL_20260926.md)
isolates the METH-32 all-ternary factor failure. It binds the same
stored Qwen R8 core, trained fp32 E128 adapter and packed ternary
factor artifact. Each arm changes only A or only B; the fp32 fine
router, output factor and core remain fixed. The 48 METH-25 documents
and 24 prefixes were already observed, so this is a diagnostic.
The intact baseline reproduced all 48 prior document nats within
0.01 and all 24 top-1 streams exactly before either mixed arm ran.

| Reused diagnostic arm | Selected factor bytes/token | Pooled BPB delta vs intact | Top-1 match vs intact | Same top-4 set on trajectory |
|---|---:|---:|---:|---:|
| A ternary, B fp32 | 4,131,840 (75% of fp32) | −0.000221 | **5,961/6,144 = 97.021%** | 89.147% |
| A fp32, B ternary | 3,440,640 (62.5% of fp32) | −0.000641 | **5,977/6,144 = 97.282%** | 89.553% |

Both arms pass the fixed selected-factor ≤4.2 MB/token, paired
BPB-penalty and donor-relative diagnostic screens. Their
category-stratified 20,000-draw one-sided 95% upper BPB-penalty
bounds are +0.000011 for A-ternary and −0.000348 for B-ternary.
Both **fail** the fixed ≥99% top-1 agreement gate, so the protocol
selects neither. The top-4 figures compare each arm's exact fine
route on its changed hidden-state trajectory with the intact
trajectory; they are not approximate-router recall. The unchanged
router's decisions diverge in roughly 10–11% of input-layer cases.

The [machine result](meth33_factor_precision_ablation.json),
SHA-256 `63c6382d501d0738860e3f442576fa6edfcd5d27d3fd50df9b5f26d9297fb89a`,
contains every paired document score, per-category top-1 counts,
bootstrap result and route count. Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth33_factor_precision_ablation.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth33_factor_precision_ablation.json`.
The local RTX 3060 audit took 69.922 s, peaked at 2.439 GB allocated
GPU memory and ended at 3.586 GB RSS. No T4 was used.

**Decision:** the METH-32 ternary mapping is too coarse even when
only one rank-8 factor uses it under this prospective top-1 gate.
These mixed variants are not stored export candidates. The BPB
improvements on reused documents do not prove general quality or
generation. A distinct next precision format should preserve more
weight levels while fitting the CPU LUT cost envelope, then be
exported and checked on new data if its diagnostic passes. Routing
at large E, actual C parity/rate, the existing generation failure
and donor-to-native family generality remain open.
