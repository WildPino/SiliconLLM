# METH-322: model-free Switch forward contract before weights

Frozen before observations,2026-10-03.321 establishes metadata eligibility
for base128/256, not a usable reference. Static installed5.13.1 source reveals
flattened sparse inputs versus a three-dimensional expert-mask permutation.
This concrete remaining risk justifies testing actual runtime BEFORE30/59GB
acquisition. No model weights/datasets/network/download/GPU/native rate work.

Pin321 raw SHA31351eae3934228468366501492c291f8a7f8b0a228bf3db80a146d797b9ebb1,
installed model sourceSHAf1eab2d3e02e70d93d20d6fb07dd3acd20519fe6f88a57d7a1b214bc5b305ecd.
Own committed controller/protocol. Torch1thread/seed322. Tiny synthetic config:
D8/FFN16/KV4/heads2/experts2/layers2 in BOTH stacks/one sparse layer each,
capacity1/dropout0/jitter0/vocab32/pad0/EOS1/start0/F32router. No pretrained
knowledge or performance claims from this artificial graph.

Direct router positive control: classifier rows pick x0 and-x0. InputB1/T4,
x0=(1,2,-1,-2),x1=1, other coordinates0. Expected capacity mask accepts first
expert0 and first expert1 only, shape1x4x2; selected probabilities exactly
F32 softmax max, positive>.5, probability/third output shapes1x4x1. Repeated
eval exactly equal (no jitter). This verifies known router behavior only.

Actual sparse forward on ALL fixed shapes1x1x8,1x4x8,2x2x8 with deterministic
arange/32-.5 values must return matching finite Tensor. Record exception
verbatim per case as scientific FAILED contract (continue other cases).
Separate actual tiny full encoder-decoder call on sourceIDs2,3,4/decoderID0,
use_cache true must return finite1x1x32 logits and a nonempty cache.
Exceptions recorded, no candidate substitution/changed inputs/code patch.

All three gates required for installed reference eligibility. Any sparse or
full-model failure means THIS installed reference unavailable, not a failure
of original pretrained Switch or conditional architectures. Preserve outcome,
then separately qualify an isolated official reference implementation/adapter
before checkpoint work. Do not downgrade the existing project environment.
Actual source weighted-top1/drop-capacity/normalization/relative-position/
prefill/cache/precision, full source hashes/quality/rate remain mandatory.

5min total wall including imports,2GiB observedRSS. No CPU benchmark overlaps.
Command: `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth322_switch_runtime_contract.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth322_switch_runtime_contract_result.json`.
