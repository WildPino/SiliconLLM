# METH-13 result: E128 Qwen carve routes signal but fails full-model quality

**Decision:** stop this Qwen2.5-0.5B, E128/top-32 donor-channel geometry
before GPU training or native export. The [protocol](METH_13_QWEN05B_JOINT_UPCYCLE_PROTOCOL_20260926.md)
fixed the gates before the run. The [machine record](meth13_qwen05b_preflight.json)
contains all 24 per-layer route rows, data and artifact hashes; the
[executable](../../../benchmarks/donor_adaptation/s1/meth13_qwen05b_preflight.py)
regenerates the labels and routers. This is a failure of the tested
step-zero conversion, not evidence against jointly learned experts or
large-E scaling in general.

## Reproduction and controls

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth13_qwen05b_preflight.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth13_qwen05b_preflight.json --assets results/native_expert_scaling/meth13_qwen05b
```

The first invocation stopped after label construction because the project
`.venv` lacked `scikit-learn`; it produced no router or quality verdict.
Installing `scikit-learn==1.9.1` in that environment allowed the unchanged
command and frozen protocol to finish. Runtime: PyTorch 2.6.0+cu124 on
CPU, NumPy 2.4.6, six CPU threads, fp32/eager donor. No GPU was used.
The donor is pinned Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`; the local
`model.safetensors` SHA-256 is
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The tokenizer fingerprint is `4efeeb9382a77a06`. The runner asserted
all four data-ID hashes in the protocol and exact L24/D896/F4864 geometry.

Each of the 24 FFNs was partitioned into 128 disjoint groups of 38 donor
neurons. The partition and input-only ridge routers were learned on
calibration slices. They are **partitions of pretrained channels**, not
128 independently learned experts. The label and router NPZ artifacts
are locally regenerable at `results/native_expert_scaling/meth13_qwen05b/`;
their SHA-256 hashes are respectively
`1c6840399ace72afee3e7200ad21a5a1519b12a688545d5035143168d8c637b1`
and `55f01e1d5c8c5ef9df5db8a813036d48d889f4ad9286046ca6fefb191e509518`.

The all-groups-active wrapper matched intact-donor logits on the first
16 positions within **1.72e-5** maximum absolute error, below the
frozen 1e-4 identity tolerance. Hard top-32 changed logits by up to
**18.30** in that check. The process ended after **70.891 s** with
**4.143 GB RSS**; this is not a peak-memory certificate.

## Routing and paired quality

On the reserved 4×256-token internal route slice, mean fitted top-32
group-ID recall against the post-activation oracle was **0.5522**, versus
**0.2482** for the seeded random router. The selected groups captured
**0.7917 of the mass captured by oracle top-32**, versus **0.4049** for
random: advantage **0.3868**, above the frozen +0.15 gate, and the
fitted fraction exceeds the 0.60 gate. These numbers are relative to the
oracle's selected mass, **not total FFN activation mass**. Per-layer
fitted mass fraction ranged from **0.7090** (layer 3) to **0.9485**
(layer 23). The route gate passes.

| Identical first two 512-token heldout windows, 4,396 scored bytes | BPB |
|---|---:|
| Intact donor | 1.083022 |
| Hard top-32 of 128 donor-channel groups, all 24 FFNs | 2.443496 |
| Paired change | **+1.360474** |

The paired damage is **0.960474 BPB above** the frozen +0.40 stop.
The sampled heldout corpus has informed earlier Qwen method selection,
so this is an internal rejection pilot, not a final generalization
estimate. No rank, group size, top-k or router objective was retuned on
these windows. There was no training, free-generation test, quantization,
native C export, LUT timing or accepted-token rate measurement.

## Interpretation and next mechanism

The router learns a useful signal, yet recovering roughly 79% of the
oracle top-32 mass does not preserve end-to-end behavior when every
FFN loses its other 96 groups. Sparse inputs to later layers also
depart from those used to fit their routers. This experiment cannot
attribute the BPB gap separately to router errors, truncation even
under oracle routing, or error accumulation; those require a new,
predeclared diagnosis if needed.

At BF16 the planned selected FFN, tied head, attention and exhaustive
router would address **522,747,904 bytes/token** before other overhead,
a **13.07 ms** payload floor at 40 GB/s. That cost arithmetic remains
unvalidated because the quality gate stopped the conversion. A new
transfer cell must change the architecture or training mechanism, retain
an exact-donor warm start while training distinct conditional experts,
measure exposure per expert, and test final sparse quality. At a 10×
larger expert pool, it must also avoid a full dense router scan and
measure actual CPU shortlist/LUT work. Repeating this hard carve with a
nominally different group count would not answer the user's scaling
question. The next cell needs a new frozen protocol and its own heldout
instrument before any GPU spend.
