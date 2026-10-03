# METH-321: smaller active decoder, original pretrained banks not acquired

Freeze `1e965ae`, exit0,14.734s including imports/endRSS1,315,504,128bytes.
Six original320 config/index bytes reused; no network/weight download.
Both completed320 base case records reproduce EXACTLY. All three META graphs
match every official index key and declared byte count/serialization alias
accounting.12/12/24 actual graph routers are EXACT built-in Top1Router;
large128 missing legacy config field now verified through topology instead.

| Official donor | Inferred UNIQUE parameters | Decoder F32 addressed bytes/token | BF16 scenario | W8+row scales scenario |
| --- | ---: | ---: | ---: | ---: |
|base128|7,415,217,408|497,536,512|250,005,504|126,774,020|
|base256|14,664,154,368|499,895,808|252,364,800|129,133,316|
|large128|26,309,291,008|1,547,479,040|777,035,776|393,024,004|

Base128/256 have SAME123,764,736 decoder-active large-matrix coefficients:
top1 sparse expert, every dense/core/head matrix counted. Doubling actual
source-labelled bank count adds F32 router traffic, not a second active FFN.
Encoder weights and decoder cross-KV projection are prefill-only; full encoder
work is missing runtime evidence, not a free prompt. Attention/cache/context
cost must be added. Shapes/counts are inferred from official names + installed
META model; no actual weight headers/values or distinct function hashes checked.

Unique storage scenarios: base128 F32~29.661GB/BF16~14.833GB/W8~7.443GB;
base256 F32~58.657GB/BF16~29.333GB/W8~14.719GB. Published shard-index total
base12829,956,961,280 andbase25658,952,709,120bytes include serialization
aliases; acquisition must verify actual shard sizes/hashes. Base256 is an
actual approximately10B-scale pretrained candidate, not invented/copy-grown n.
RAM/disk/acquisition/runtime validation and full preserved quality remain missing.

**Decision:** base128/256 eligible ONLY for actual source-binding/reference-
semantics work. Large128 unchangedBF16 geometry fails560MB; W8 is a separate
unqualified precision path, not an override. No source-quality/native50 claim.
Original14ms/560MB yardstick is only a screen; real full pipeline decides.

Installed5.13.1 sparse forward flattens input before invoking a router/expert
dispatcher whose selected-mask permutation expects three dimensions. This
static observation identifies a concrete runtime-contract risk, not yet a
measured failure. A model-free forward/shape/router control must resolve it
BEFORE downloading approximately30/59GB checkpoints or scoring donors.
Source original relative bias, weighted top1, capacity/drop behavior, jitter
evaluation and cached/prefill semantics still need a trustworthy reference.

[Full metadata/ledger record](meth321_switch_metadata_compatibility_result.json)
SHA `31351eae3934228468366501492c291f8a7f8b0a228bf3db80a146d797b9ebb1`.
[Official base256](https://huggingface.co/google/switch-base-256) and
[Switch paper](https://arxiv.org/abs/2101.03961) are source availability/family
references. MLM/span reconstruction and encoder-decoder generation need
appropriate untouched paired tests; they do not establish instruction/chat
quality or substitute for the final pretrained-transfer/large-n/SAMEartifact50
and cross-family/actual100B requirements.
