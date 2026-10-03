# METH-320: official Switch active-core applicability screen

Frozen before config/index analysis,2026-10-03.318/319 apparatus repair is
preserved; full-width GigaChat additive decoder fails native14ms/stability.
New variable: select a genuinely pretrained sparse donor with smaller active
decoder core, not another synthetic optimization or a smaller student stand-in.
Exact current official model revisions were located through public repository
API, without loading/scoring weights. No Switch/QMoE experiment found in the
reviewed local native research documentation/source search.

## Fixed candidates and primary sources

- google/switch-base-128 revision86c815ec05361a33a8b49fc717277da9c0a4e711.
- google/switch-base-256 revisioncdac1724c078ea4974b4087c59634799561e7835.
- google/switch-large-128 revisionf5573ca2d15edb5ebc609438693db5bbf2e24f58.

Fetch ONLY immutable official config.json and PyTorch shard index for base128/
256, safetensors index for large128;<=4MiB/file, no tensor/shard/tokenizer or
checkpoint download. Preserve bytes/URL/SHA/config and index names. Total
budget5min/RSS2GiB, no GPU/benchmark/inference job. Stop on request/namespace/
geometry/size mismatch; preserve partial record/files before any repair.

[Official Google base128 repository](https://huggingface.co/google/switch-base-128),
[Switch paper](https://arxiv.org/abs/2101.03961),
[official Google checkpoint list](https://github.com/google-research/t5x/blob/main/docs/models.md)
establish pretrained sparse encoder-decoder family availability, not our CPU
quality/rate. [QMoE paper](https://arxiv.org/abs/2310.16795) and its
[official code](https://github.com/IST-DASLab/qmoe) show a different GPU
compression framework; do not borrow its speed/accuracy or install it here.

## Metadata-only controls and accounting

Built-in installed Transformers Switch config/model on Torch META device:
NO allocated parameter payload, NO trust_remote_code. Record library versions
and source-file hashes. Require official model_type/is_encoder_decoder,
top1,ReLU,ungated,F32router and expert counts128/256. Require ALL index names
match built-in state_dict namespace. Infer shapes from this META graph, never
call them source-tensor-header verified. Require declared index total_size
matches inferred unique/all-index-entry count times2 or4bytes; distinguish
tied serialization aliases/precision uncertainty from unique parameters.

Decoder-active ledger: ALL decoder self q/k/v/o, cross q/o (cross k/v
explicitly precomputed per prompt), selected ONE expert per sparse FFN, every
dense FFN, routers/norm/relative-position biases, full tied vocabulary head
plus embedding lookup row. Do not charge all experts as active or exclude
head/shared dense/core. Store every inferred source tensor shape and every
included/excluded-for-prefill decoder row. Count original full encoder +
cross-KV prefill as missing runtime work, not zero-cost generation.

Three precision SCENARIOS: originalF32 all weights; BF16 large matrices with
F32 router/norm/position bias; row-W8 matrices plus4byte/output-row scale,
F32 router/norm/bias. Tied embedding/head shares stored weight; head reads
full matrix per step, input reads one row. Non-FP32 quality UNVERIFIED.
Report all unique full-source storage bytes and router coefficient count;
source/output128-token FP32 KV-cache bytes separately (no DRAM/cached-rate claim).

Decision screen: BF16 complete decoder addressed weights<=560MB may license
ONLY separately frozen actual source binding/reference-semantics work; failure
closes that unchanged precision geometry. F32/W8 scenarios are diagnostics,
not replacements for quality/latency. No weights fetched/hashed, real function
uniqueness/route quality or native rate proven by official expert labels.

## Goal and family-specific limits

Switch is pretrained for text-span reconstruction. Preserving that donor
requires paired untouched reconstruction/generative behavior and relevant
tasks; it does not automatically produce a useful instruction/chat model.
Original relative-position buckets, encoder/cross attention, weighted-top1
router, expert capacity/token dropping, cached/full-prefix consistency,
source precision and prompt cost require explicit controls before conversion.
No unsupported causal BPB comparison or favorable expert subset. No trained
extra n from metadata. Increased real bank count retains source distinct
weights only after actual tensor/hash binding, full conversion and paired
quality. Full final goal/multiple-family/approximately100B remains unchanged.

Command: `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth320_switch_metadata_screen.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth320_switch_metadata_screen_result.json`.
