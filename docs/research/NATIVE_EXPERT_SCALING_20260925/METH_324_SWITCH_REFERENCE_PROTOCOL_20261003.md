# METH-324: isolated unmodified official Switch reference

Prospective protocol. Resolve323 installed5.13.1 capacity/cache failures before
any pretrained checkpoint acquisition. Retain321 metadata and323 complete
observations. New variable: official Transformers4.57.6, Hub0.36.0 in isolated
results/native_expert_scaling/meth324_switch_reference/venv. Existing .venv
is reused through an appended .pth for Torch2.6 and compatible dependencies.
Own site precedes donor site; verify actual import paths and direct dependency
requirements. No package upgrade/downgrade in existing environment.

Installer acquires only exact two pure Python wheels from PyPI version metadata;
pin SHA256/size before installation and preserve metadata/manifest/logs. Bound
all streamed downloads cumulatively1GiB, individual wheels64MiB, metadata4MiB,
official release sources1MiB. Compare installed model/config source byte-for-byte
to official v4.57.6 release URLs. No remote checkpoint code or weight downloads.
Record full resolved distribution metadata/RECORD hashes and direct dependencies.
PyPI TLS and metadata-provided hashes establish acquisition provenance, not an
independent package-signing guarantee.

Freeze driver/protocol before observation. Setup fresh only,20min total guard,
child sampled RSS3GiB including descendants; worker300s/endRSS3GiB plus parent
sampler. Preserve partial results and failed environment before any repair.
Single CPU thread, no GPU; no concurrent performance work. Wall sampling250ms.
Python venv/ensurepip creation is bounded by the surrounding session; no network
or large computation during creation. Existing pip bundled installation only.

Same323 seed322/tiny config/capacity1, zero dropout/jitter; classifier rows+/-x0.
Official tuple API(mask,selected probability,raw logits) explicitly adapted.
Direct inputs[1,4,8] x0=(1,2,-1,-2),x1=1 require exact intended capacity mask,
exact softmax selected probabilities, raw logits and deterministic eval.
All323 sparse shapes(1,1,8),(1,4,8),(2,2,8), arange/32-.5: independent per-sequence
onehot/cumsum cap1, same expert and probability reference; nonzero norm,
relativeL2<=1e-6, dropped output exactlyzero, metadata exact. Omitted-capacity
fault must exceed1e-6 for multi-token shapes; zero-output fault equals1.
Capacity1 actual full encoder/decoder [2,3,4]→[0] requires finite[1,1,32] logits
and nonempty cache, preserving original323 control.

Additional prospective cache qualification: same tiny model weights with all
router capacities explicitly64, unsaturated source[2,3,4] and decoder[0,5,6,7].
Four cached steps with reused encoder versus recomputed full source/full causal
decoder prefix require relative logitL2<=1e-5, exact greedy top1, finite logits,
self-cache lengths1..4 and cross-cache3 at bothlayers. This is a separate
unsaturated apparatus context. Source capacity is per call: cap1 saturated full
prefix is not assumed numerically equivalent to stepwise decode.

All three321 original configs instantiated on META, immutable config/index
hashes verified. Entire tensor-name/shape map, unique parameter count and actual
top1 router count/bank width must exactly equal321. No injected fields.

All five gates must pass to justify separate real-source manifest/header work.
Any failure preserves scope and closes this unchanged reference qualification.
This is no proof of real expert diversity, transferred knowledge, native LUT
cost, quality preservation or accepted token rate. Final>=50 token/s and actual
large-n/cross-family/~100B requirements remain.

Command from repository root:
```
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth324_switch_reference.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth324_switch_reference_result.json
```

Primary sources:
- https://github.com/huggingface/transformers/blob/v4.57.6/setup.py
- https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/switch_transformers/modeling_switch_transformers.py
- https://pypi.org/project/transformers/4.57.6/
- https://pypi.org/project/huggingface-hub/0.36.0/
