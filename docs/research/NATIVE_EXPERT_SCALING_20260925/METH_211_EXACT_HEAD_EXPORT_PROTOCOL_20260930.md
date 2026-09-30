# METH-211: materialize the fixed exact-head Q8 core

METH-210 passes the original development quality gates and the K64
FP16-scale proposal inclusion/row-choice screen. Save the tested
representation before fresh quality or native implementation.

Bind METH-210 result, METH-193 core and donor hashes. Copy all 363
METH-193 tensors unchanged except rounding the tied-head proposal
row scales from FP32 to FP16. Add the original donor BF16 tied
embedding/head matrix under `model.embed_tokens.weight.bf16`.
Verify proposal code/scale bits against METH-210 hashes. Preserve
all 72 FFN code/scale pairs, attention and FP32 controls byte-identically.
Declare exact embedding lookup, full-head likelihood and K64 exact-row
greedy proposal as separate operations in the artifact metadata.

Expected array payload: 548,701,184 original bytes -303,872 scale
bytes +272,269,312 exact-head bytes =820,666,624 bytes, 364 tensors.
Save and verify every tensor and metadata field, original head equality
and all unchanged code/scale pairs. Report physical/header bytes and
whole-file SHA. The proposed greedy active ledger remains 559,794,176
bytes/token; this is not native precision/DRAM accounting. Stored core
identity licenses a quality/native composition experiment, not a rate
or independent quality pass, and does not solve E12800 routing.

CPU-only export, six host threads, <=10 minutes, <=20 GiB RSS,
<1 GB combined output/report. Preserve failure record on binding,
readback, dtype, size or budget failure. No T4, training or quality text.
Runner: `benchmarks/native_expert_scaling/meth211_export_exact_head_core.py`.
