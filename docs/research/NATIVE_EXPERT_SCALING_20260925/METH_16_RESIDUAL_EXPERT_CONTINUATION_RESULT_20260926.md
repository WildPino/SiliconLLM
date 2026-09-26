# METH-16 result: pretrained donor plus trained E128 residual experts

**Decision:** the preregistered **development** joint gate passes on the
local RTX 3060. The student improves heldout BPB against its intact
Qwen2.5-0.5B donor, the trained routing matters against a row-permuted
router, all expert output slots change, and greedy repetition does not
regress relative to donor. The [protocol](METH_16_RESIDUAL_EXPERT_CONTINUATION_PROTOCOL_20260926.md)
was written before the continuation; the [machine record](meth16_residual_expert_continuation.json)
contains all 1,008 new update rows and generated token IDs. The
[runner](../../../benchmarks/donor_adaptation/s1/meth16_residual_expert_continuation.py)
resumes the [METH-15](METH_15_ZERO_RESIDUAL_EXPERT_SMOKE_RESULT_20260926.md)
checkpoint without resetting Adam or RNG.

## Bound run and one invalid launch

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth16_residual_expert_continuation.py --resume results/native_expert_scaling/meth15_zero_residual_expert_smoke_rtx3060.pt --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth16_residual_expert_continuation.json --checkpoint-dir results/native_expert_scaling/meth16_checkpoints
```

The first invocation exited **before loading the checkpoint or applying
an update**: one hex character of the checkpoint SHA-256 was mistyped
in the runner/protocol. The correct local hash was independently
rechecked as
`cff4d2cbe4851d736f76d7f0838c42752a9d7bceec83054c34239d413d9a727e`.
The literal was repaired; source, data, optimizer, schedule, gates and
initial state were unchanged. The successful command above verifies
that hash, donor safetensors, tokenizer and training/evaluation IDs.
The 16×512 heldout ID SHA-256 is
`48bb4258c69a93dcc3e421444213dfa0eca38856dc9bdd24edf96c408310d1aa`,
33,374 scored bytes. This is a new **window set**, not a fresh
document corpus; final generalization remains untested.

The frozen donor is the shared core. Each of 24 layers has 128
**independently updated** rank-8 residual experts with top-4 routing.
The adapter stores 44,040,192 expert factors plus 2,752,512 router
weights. The run continued from update 16 through 1,024, exposing
524,288 training token positions in total, including the smoke.
The new continuation consumed **1,653.984 s** (27 min 34 s), peaked
at **2.863 GB allocated GPU memory**, and ended at **3.412 GB process
RSS**. No T4 was used. These are run-specific process/GPU readings,
not a complete system-memory certificate.

## Paired quality and router utility

| Applied update | Student BPB | Delta to same donor |
|---:|---:|---:|
| 16 | 0.834710 | −0.005869 |
| 256 | 0.835145 | −0.005435 |
| 512 | 0.827808 | −0.012771 |
| 768 | 0.827191 | −0.013389 |
| 1,024 | **0.824196** | **−0.016383** |

The disabled-expert donor is **0.840579 BPB** both before and after
training; its frozen weights did not drift. No intermediate checkpoint
crossed the preregistered +0.05 gross-failure stop. Terminal paired
BPB clears the +0.02 development gate.

Permuting the trained router's expert rows while leaving learned
expert factors fixed raises BPB to **0.866327**, so the correct learned
route beats this frozen null by **0.042131 BPB**, well above the
predeclared 0.002 bar. This demonstrates route/expert correspondence
on this pilot, not optimal routing or a gain from raising E. All 128
output factors differ from zero in each of the 24 layers. That
proves parameter updates, but the run did not log per-expert token
exposure or semantic specialization.

For 16 fixed 128-token prefixes followed by 128 greedy generated
tokens, repeated token 8-grams at least three times occurred in
**12/16 donor** and **11/16 student** continuations. Both arms
generated all 2,048 requested continuation tokens. The relative
nonregression gate passes; the high donor loop rate means this prompt
set **does not establish useful generation**. Generation samples and
flags are in the machine record.

The terminal resumable adapter/optimizer/RNG checkpoint is local at
`results/native_expert_scaling/meth16_checkpoints/meth16_update1024.pt`,
561,603,306 bytes, SHA-256
`a9e74c9ceeb8a91fd04f9b9e35a981965db9ab824d2d25ea728dd940ea5365a4`.
Update-256/512/768 checkpoint hashes and bytes are in the record.

## Cost and boundary of the result

The core is still the whole Qwen0.5B donor. At BF16, its roughly
988 MB source payload already has a ~24.7 ms streaming floor at a
favorable 40 GB/s, before experts, router or overhead. Selected
rank-8 experts add **2.753 MB** of BF16 weights/token across 24
layers; the exhaustive fp32 E128 router adds **11.010 MB/token**.
These are addressed-weight counts, not measured DRAM traffic or C
rate. A quality-preserving one-byte core would lower the arithmetic
floor, but the old one-byte Qwen donor rate is a **different artifact**
and its greedy fidelity was imperfect. No current C export includes
these trained experts.

The rank-8 bank adds only 44M parameters, far below the intended
10B/100B capacity regime. At a hypothetical rank 128, each expert
across 24 layers would store 5,505,024 factors. Approximately 1,700
and 17,000 such experts would add 9.36B and 93.59B factors,
respectively, while top-4 active expert factors would stay fixed.
But an exhaustive fp32 router at E17,000 would address **1.462 GB/token**,
a **36.56 ms** payload floor at 40 GB/s before any expert or core work.
This projection makes sublinear candidate selection and measured CPU
LUT work necessary for the user's large-RAM target; it is not a
trained or timed 10B/100B result.

**Next:** test donor-relative quality and task/generation behavior on
fresh documents, then price a low-bit shared core and export this
exact adapter to native C. In parallel, design E expansion with
distinct trainable capacity and sublinear routing; compare quality
and CPU cost as E grows. The present pass is a credible transfer
mechanism at one tractable scale, not completion of the full goal.
