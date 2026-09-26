# METH-17 result: internal E128 gain does not transfer to fresh code

**Decision:** this exact METH-16 checkpoint fails the prospective
fresh-document quality gate; stop its native export until the
generalization loss is repaired and a new independent corpus passes.
The [protocol](METH_17_FRESH_DOCUMENT_TRANSFER_PROTOCOL_20260926.md)
fixed the manifest, scoring rule, bootstrap and limits before model
scoring. The [manifest](meth17_fresh_document_manifest.json),
[machine result](meth17_fresh_document_result.json) and
[runner](../../../benchmarks/donor_adaptation/s1/meth17_fresh_transfer_audit.py)
retain every source/text hash and paired document loss.

## Bound run and selection controls

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth17_fresh_transfer_audit.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth17_fresh_document_manifest.json --manifest-sha256 7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8 --checkpoint results/native_expert_scaling/meth16_checkpoints/meth16_update1024.pt --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth17_fresh_document_result.json
```

The run re-created the 60-document selection from its inputs and
matched the frozen manifest SHA-256
`7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8`.
It verified the pinned Qwen2.5-0.5B safetensors/tokenizer and the
unchanged update-1024 adapter SHA-256
`a9e74c9ceeb8a91fd04f9b9e35a981965db9ab824d2d25ea728dd940ea5365a4`.
Each selected span's first, middle and final 256 bytes were absent
from both Qwen calibration and prior heldout corpus files. The
documents had not been used to train or select METH-15/16, although
the external prose/technical source had been used in a different
donor program. This fragment screen cannot prove absence of every
short overlap or pretrained-model contamination.

The evaluator scored all 66,914 Qwen text tokens from 245,700
UTF-8 bytes, with a donor EOS document prefix, 512-token target
strides and up to 512 preceding tokens. Each arm used the same
BF16/SDPA RTX 3060 model and IDs, switching only the trained
residual experts on or off. The run took **50.750 s**, peaked at
**2.439 GB allocated GPU memory**, and ended at **3.252 GB RSS**.

## Paired document quality

| Category | Docs / bytes | Donor BPB | Student BPB | Delta |
|---|---:|---:|---:|---:|
| Code from pre-adapter project commit | 24 / 98,280 | 0.971232 | 1.000358 | **+0.029126** |
| Prose | 24 / 98,280 | 1.065589 | 1.069139 | +0.003550 |
| General technical | 12 / 49,140 | 0.637307 | 0.638030 | +0.000724 |
| **Pooled** | **60 / 245,700** | **0.942190** | **0.955405** | **+0.013215** |

The preregistered 20,000-draw, category-stratified paired bootstrap
(seed 171717) gives a one-sided 95% upper bound of **+0.015317 BPB**.
That is inside the +0.02 upper-bound requirement, and each category
is inside its +0.05 gross-failure limit. The pooled **point delta
exceeds the separately mandatory +0.01 limit**, so the joint gate
fails. All **24/24 code documents** worsen, with per-document changes
from +0.011997 to +0.059309 BPB; this is not one outlier driving
the result. Seventeen of 24 prose and seven of 12 technical documents
worsen, with much smaller pooled effects.

This is a direct contradiction of treating METH-16's −0.016383 BPB
on the earlier internal Qwen windows as general donor-relative
retention. The two evaluations use different source content and
scoring protocols; their deltas should not be subtracted to estimate
a causal domain effect. The uniform code regression indicates that
further diagnosis should focus on when the residual bank should
contribute and how to train it without sacrificing other domains.
These 60 documents have now informed that design question and must
**not** serve as its final tuning or promotion set.

No task benchmark, generative usefulness or C performance was
measured here. The core is still dense BF16 and the router still
scores all E128 rows. A repair must be frozen and trained or
calibrated on separate data, then judged on another disjoint
document set before native export or larger-E claims resume.
