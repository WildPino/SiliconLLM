# Coherent cached FIT final states and complete stored audit

10 October2026. COMPLETE: **PAIRED_CACHED_FIT_PASS**. Goal INCOMPLETE.
No codec fit, compact history recovery, native call, T4 or new DEV observation.

## Why this observation was needed

The [one-case qualified instrument](SOURCE_CACHED_FINAL_RESULT_20261010.md)
showed that stored full-prefill states differ from actual cached-generation
states, despite exact same-call norm/head reconstruction. The original ALL48
alignment FAIL remains immutable. The new variable is the previously missing
raw final state and observed postnorm feature, paired with actual cached logits
on the24 authoritative FIT cases. This is offline supervision custody, not a
replayed recovery dose or a new chatbot-quality admission.

## Frozen record and commands

[Protocol](SOURCE_CACHED_FIT_PROTOCOL_20261010.md) and
[tool](../../../benchmarks/native_expert_scaling/source_cached_fit_capture.py)
freeze `f2f40118792a56d978d048c45889b838f144b2ca`.
[Binding](source_cached_fit_binding_20261010.json), SHA256
`4116baa7da2763753b5516f7db090b3473f12f372dd090ad5781dca524cae248`,
114 input extents/5,530,886,470B. Binder validation about5.38s tool wall, separate
from held capture. Source411 BF16 parameters/1,554,863,488 elements, pinned
Falcon-H1-1.5B source revision and runtime operators in binding.

From repository root, isolated Python3.12:

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_cached_fit_capture.py --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_fit_binding_20261010.json --binding-sha 4116baa7da2763753b5516f7db090b3473f12f372dd090ad5781dca524cae248 --freeze f2f40118792a56d978d048c45889b838f144b2ca --directory results/native_expert_scaling/source_cached_fit_20261010 --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_fit_result_20261010.json
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_cached_fit_capture.py --audit --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_fit_binding_20261010.json --binding-sha 4116baa7da2763753b5516f7db090b3473f12f372dd090ad5781dca524cae248 --freeze f2f40118792a56d978d048c45889b838f144b2ca --source-result docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_fit_result_20261010.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_fit_stored_adjudication_20261010.json
```

These namespaces are now immutable; commands document completed work, not a
request to replay it. Capture session90559/launcher20972 created12:25:49,
worker22192 created12:25:54.6128457+02:00. Audit session6122/launcher13068
created12:36:44, worker26704 created12:36:49.9315364+02:00. Both held sessions
exit0/error=null, CLOSED; both workers/launchers absent at12:37:11 observation.
No first fault in this family; earlier instrument startup fault stays retained.

## Exact scientific gates

24 cases/12 domains/2 cases per domain,4422 labels, batch1. Actual cached
generation has the old BF16/eager/tiled-SSD settings, seed0/TF32off/6threads,
max_new_tokens256/use_cache=true/no sampling/EOS[11,228]. All four gates pass
for EVERY case, without tolerance substitution:

| Gate | Result |
|---|---|
| Actual generated IDs versus original FIT IDs | EXACT4422/4422 |
| Actual same-call logits versus original BF16 logits | All4422x65537 bits EXACT |
| Observed postnorm versus source norm(observed raw final state) | All4422x2048 bits EXACT |
| Observed logits versus source head(observed postnorm) | All4422x65537 bits EXACT |

One source model instance/24 generations;4422 actual base forwards and4422
source LM heads;4422 additional paired head reconstructions/24 batched norm
reconstructions. No optimizer/native/RESERVED query. Hooks removed before
reconstructions. No24-layer history or cache-coordinate trace was repeated.

All411 parameter BF16 bytes SHA256 match the pinned safetensors, before AND
after family; parameter object identity/version also unchanged. This covers
parameters, not all nonparameter buffers. All input extents unchanged through
terminal sealing. Every case state/logit/frame/result is durable before its
scientific verdict. A scientific FAIL would still finish the family and audit.

## Independent complete stored adjudication

[Stored audit](source_cached_fit_stored_adjudication_20261010.json) verifies all
114 inputs and171 namespace outputs, sealed case records, every exact gate,
all4422 input/cache-length/position-ID frames, parameter seals and call/resource
counts. Decoded full-V argmaxes equal generated IDs. Exact old score bits imply
KL=0 against their original same-case logits, without an approximate KL pass.

548 scalar head witnesses using F64 math.fsum: maximum absolute difference
**0.20070657237010892** <= frozen1.0 BF16-rounding allowance. F64 analytic
RMSNorm(gamma,epsilon1e-5) versus actual observed BF16 postnorm: maximum
per-case relative RMS **0.0024441823431146587** <= frozen.01.
These are numerical checks, not a claim of exact F64 equality to BF16 operators.
Audit has zero source history forwards/full-head contractions/optimizer updates.

## Measured cost and seals

| Resource | Capture | Stored audit |
|---|---:|---:|
| Held seconds including input/output seals | 644.500 | 21.281 |
| Worker result timestamp seconds before final write | 624.000 | 9.609 |
| Worker OS peak through exit, B | 3,728,396,288 | 191,070,208 |
| Launcher OS peak, B | 31,260,672 | 30,924,800 |
| Combined OS, B | 3,759,656,960 | 221,995,008 |
| GPU allocated peak, B | 5,558,098,944 | No GPU |
| GPU reserved peak, B | 9,680,453,632 | No GPU |

All fixed caps pass: capture1500s/8GiB OS/10-11GiB GPU/2GiB output;
audit300s/1GiB OS/512KiB result. Forecast10-15min was consistent with observed
10min44.5s. Instrumented offline capture is not accepted CPU engine throughput.
Audit's final log timestamp is10.391s after its final JSON write;9.609s above
is the stored prewrite value, not the complete held duration.

171 namespace files1,214,875,583B; numeric payload exactly**1,213,555,992B**,
matching frozen4422*(3*2048*2+2*65537*2). Result141,583B; namespace+result
1,215,017,166B. Metadata is extra, no omitted norm reconstruction stream.

[Result](source_cached_fit_result_20261010.json) SHA256
`af5a8b640249de5ff84e56e13ca546a015b6fd2b6a04ac68ebbb1582f16cd506`.
[Audit](source_cached_fit_stored_adjudication_20261010.json),12,196B, SHA256
`e5900ec987fae21774afabc47b21d4786786cfaa6ed2a5709e43420ff11313c9`.
Both terminal receipts and worker logs retained beside their JSONs.

## Decision toward the converter

Coherent FIT final features are now qualified for ONE paired output-codec
selection. [Output-weighted rank derivation](OUTPUT_WEIGHTED_RANK_CODEC_DERIVATION_20261010.md)
defines a precise fixed quadratic surrogate and a paired whitened encoder/head,
with255 information coordinates plus an original RMSNorm carrier. It has NOT
been fitted or tested. Actual native state/norm/head are F32, not F16.

Next freeze one algorithm/conditioning/precision/quality protocol and finite
costs; fit on these FIT observations only, freeze the pair before matching DEV
capture, reuse the completed53-label DEV observation if source custody matches.
The old actual51 catastrophic failures remain evidence. This PASS does not
establish compact information preservation, causal reachability, useful experts,
chatbot behavior, same-artifact50token/s, physical DRAM or another family/scale.
