# METH-11 result: shared 25% FFN and routed residual fail the frozen pilot

**Decision:** reject this full-layer step-zero geometry before training or
native export. This does not reject all shared/residual or jointly trained
conditional models. The [protocol](METH_11_SHARED_RESIDUAL_PROTOCOL_20260926.md)
and [raw plan and scores](meth11_qwen_shared_residual_pilot.json) were fixed
before scoring. The executable is
[`meth11_shared_residual.py`](../../../benchmarks/donor_adaptation/s1/meth11_shared_residual.py).

## Run and source binding

Command from the repository root:

```text
python benchmarks/donor_adaptation/s1/meth11_shared_residual.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth11_qwen_shared_residual_pilot.json
```

Qwen2.5-1.5B revision `8faed761d45a263340a0528343f099c05c9a4323`
was loaded in CPU fp32, six PyTorch threads. The frozen H1 24×512 heldout
slice had ID SHA-256 `a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`;
its full IDs matched the bundled copy. Only the first two windows were
scored (4,396 UTF-8 bytes). Calibration-only group labels SHA-256
`c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c`,
activation statistics SHA-256
`49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe`,
and E37 router SHA-256
`42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a`
were bound by the executable. The donor safetensors SHA-256 was verified in
[METH-10](METH_10_QWEN_LOCAL_READINESS_20260926.md), not rehashed here.
Wall time was 54.01 s. No GPU job was run.

## Observed quality and controls

| Arm on the same first two 512-token windows | BPB |
|---|---:|
| Intact donor, full fp32 | 0.960190 |
| 64 shared groups plus top-16 of 192 residual groups, all 28 layers | 1.666242 |
| Delta versus donor | **+0.706052** |

The all-groups-active transform matched the intact donor's first 16
positions with **0.0 maximum absolute logit difference**, verifying the
weight partition and wrapper. Every layer had exactly 256 groups ×35
neurons. The 64 groups chosen by the calibration proxy covered a mean
51.47% of that proxy's mass across 28 layers (range 34.30–99.71%).
This proxy coverage is not retained model quality. The +0.706052 BPB
loss exceeds the frozen +0.20 step-zero stop by 0.506052 BPB.

The pilot donor BPB differs from the earlier 24-window 0.767595 because
this experiment intentionally scores just the first two windows. It is
the *paired delta on identical tokens* that decides this screen. These
windows were previously used by H1 and cannot serve as a final fresh
generalization test. No task or free-generation claim follows.

## Cost and expert-count implications

From source tensor shapes, the 28 donor FFNs store 1,156,055,040 distinct
weights. The selected 2,800/8,960 neurons/layer address 361,267,200 FFN
weights/token, versus 1,156,055,040 if dense. This is arithmetic, not a
native byte or rate measurement. At one byte/weight, Qwen's tied fp32
embedding/head still addresses about 933.5 MB/token for the output head;
the 361.3 MB selected FFN, 154.1 MB attention matrices and 8.26 MB
E192 int8 router together imply at least **1.457 GB/token** before scales,
norms, KV traffic and code overhead. At the favorable 40 GB/s yardstick,
that is **36.43 ms/token of addressed payload alone**, already above the
20 ms budget for 50 tok/s. The current Qwen donor engine stores that head
fp32 and has no shared FFN kind. Low-bit head/kernel/export work is a
prerequisite even if quality were solved.

At E=1,920 residual experts with the same 35-neuron width, a dense int8
router across 28 layers would address 82.58 MB/token, 10× the E192
router, although top-k remains 16. The *additional* 1,728 distinct
experts would require 7.803 billion new FFN weights (about 7.80 GB at
int8), which this 1.5B donor does not contain. Those bytes could be
resident if RAM permits, but their useful capacity and routing quality
would need training and a direct test. This projection cannot establish
10B→100B scaling. [NES-03](NES_03_INT8_ROUTER_SHORTLIST_RESULT_20260925.md)
provides a provisional CPU shortlist mechanism on synthetic large E;
it supplies neither these learned weights nor their quality.

Extending that arithmetic another 10×, E=19,200 residual experts would
store about 87.38 billion total weights under this fixed small-donor
geometry, but an exhaustive int8 router alone would scan **825.75
MB/token**. Its favorable 40 GB/s floor is **20.64 ms/token**, exceeding
the entire 50 tok/s budget before any expert, core or head reads. Thus a
bounded-candidate or hierarchical router is required for the user's
10B→near-100B expert scaling, and its recall and model quality must be
tested. NES-03's present shortlist still performs an exhaustive int8
first-stage scan; its measured E1280 win cannot remove this E19,200
traffic floor. These are projected weights, not a trained 87B model.

## Decision and next discriminating question

The loss occurs without quantization or attention changes, so the
31.25%-active FFN selection itself is too destructive at step zero in this
configuration. Spending GPU hours to repair a +0.706 BPB loss is not
justified by the earlier H1 S3 result (+0.155895 after about 11 h on
only eight layers). The next candidate must give the shared core a
*trained or donor-preserving approximation* before removing most FFN
channels, and must bind its active-byte plan to a credible low-bit head.
Use calibration or training data to design it, freeze one candidate, and
evaluate on a new untouched split. A 10× expert-count test must measure
learned route quality as well as CPU selection cost.
