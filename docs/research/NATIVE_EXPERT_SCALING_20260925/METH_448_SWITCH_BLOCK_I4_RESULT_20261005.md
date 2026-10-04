# METH-448: lower block-I4 KL, more changed argmax positions

**ALL9 apparatus PASS;4of5 local feasibility PASS, argmax FAIL. Recipe CLOSED.
Goal ACTIVE / INCOMPLETE.**

Source/math/[protocol](METH_448_SWITCH_BLOCK_I4_PROTOCOL_20261005.md) frozen
012a5bb BEFORE first import/quantization; resumption78d0bb3 before command.
One session59858, terminal exit0, no rerun/repair/gate change:

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth448_switch_block_i4_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth448_switch_block_i4_result.json

[Raw record](meth448_switch_block_i4_result.json),
SHA256c5d26a3e8192ae6c648567134290783192250ed7a92730cafb25db44440e516c.
All source/helper/protocol/runtime hashes and output inventory in raw.

## Scope and qualification

ONE fixed64-column block scale, symmetric levels-7..7, represented F32 scales,
nearest-even, all original coefficients/neuron order retained.64 was selected as
smallest power of two under60% nominal I8 byte cap, not validation/grid tuning.
All128 REAL original source128 last-bank11 functions; WI3072x768, WO768x3072.
Same336 consumed original teacher prefixes/books18..23, input/core/original p,
native A16 quantizer/ReLU/residual/final RMS/tied I8 head. Zero fitting.

ALL nine apparatus gates PASS: fresh source/parents/archives;336 original
complete native states/A16/full heads byte-exact;256 full matrix pack inverses
and saved bank exact; exhaustive pair/extreme/zero/fault/nearest-even and exact
Fraction dyadic-scale controls;1344 actual I4 projection I32/I64 block sums and
ascending-F64/F32 outputs exact; same original readout/input/p; independent KL
within1e-10; removal complete heads byte-exact443; one fixed unfitted encoding.

Each block |integer sum|<=64*7*32767=14,679,616<2^31. After exact integer
blocks, F64 scale multiplication and ascending block accumulation, activation
scale multiplied once, F32 cast once. This qualifies Python arithmetic/real
conversion, not C SIMD, full quantized model or timing.

## Actual gates: no reinterpretation

| Gate | Required | Observed | Outcome |
| --- | ---: | ---: | --- |
| Correct-ID mean source KL | <=.01 | .000623135892746 | PASS |
| EVERY book mean source KL | <=.05 | max .000782541557081 | PASS |
| Changed argmax fraction | <=.01, at most3/336 | 9/336=.026785714285714 | **FAIL** |
| ID+1 increases mean source KL | >=.01 | .178877453868393 | PASS |
| Nominal bank storage ratio | <=.60 | .560674886437378 | PASS |

Correct-ID median KL .000053514232960; p95 .003804943273706;
maximum .011296769890470. Per-book:

| Book | Positions | Mean KL | Changed argmax |
| --- | ---: | ---: | ---: |
| 18 | 56 | .000689893057818 | 1 |
| 19 | 56 | .000679771822727 | 2 |
| 20 | 56 | .000691117962407 | 0 |
| 21 | 56 | .000782541557081 | 2 |
| 22 | 56 | .000600178981855 | 2 |
| 23 | 56 | .000295311974589 | 2 |

I4B64 ID+1 mean sourceKL .179500589761139/41argmax changes; removal
.032704093152633/21changes. Bank identity intervention remains substantial,
not a count of useful functions.128 distinct packed fingerprints and106 natural
IDs remain descriptive. No useful larger-n/10x capacity or another-family proof.

## What changed the next action

| Same consumed input/readout assay |447 row-I4 |448 block64-I4 |
| --- | ---: | ---: |
| Nominal128 bank bytes |303,955,968 |339,738,624 |
| Mean source KL |.001377877644269 |.000623135892746 |
| Argmax changes |4 |9 |
| WI coefficient Frobenius relative range |.13586..18799 |.10522..12486 |
| WO coefficient Frobenius relative range |.15467..40298 |.10764..15318 |

The smaller matrix/KL distortion does not imply fewer winner changes. Existing
447/448 source inputs/readout are matched, unlike446's forced-function assay.
No attribution to small margins, head quantization or semantic ambiguity has
yet been measured. Both fixed encodings remain failed under their frozen gates.

Next NEW449 should diagnose decision margins and readout error using frozen
original/447/448 states/full logits and source head rows. Prospective formulas:
original winner a, candidate b, original gap m=z0,a-z0,b; perturbation contrast
d=(zc,b-z0,b)-(zc,a-z0,a). Candidate changes winner when d>=m (ties include ID
ordering). Decompose d into true head-input displacement, A16 head-input
quantization displacement and native head output-rounding/arithmetic residual.
Independent selected head-row integer replay must qualify this decomposition.

No new candidate, fit, blocksize/precision sweep or relaxed argmax gate. Freeze
diagnostic source/protocol before new numerical observation. Its result must
change the representation/metric decision, not relabel448 passing. If no
new mechanism emerges, reassess before further scalar-precision pilots.

## Resources and artifacts

124.000s total =14.110s admission +109.890s numeric/conversion/reporting.
Peakprocess2,462,076,928B, fresh hashes12,104,059,913B, zero optimizer updates.
CPU0/Torch1/BLAS1, no GPU, new source/data or engine edits.
Bank coefficient/block scales339,738,624B vs source605,945,856B; actual
bank.npz339,739,646B, SHA8be6b0fc2dc866832cbc4ae78b38e49f7bc33d7da211cac84a5324dc0b869eee.
Nine retained outputs555,100,684B, all SHA/paths/sizes in raw: full bank, source
prefixes/full logits and three counterfactual complete logits/state archives.
Original-prefix/logit and removal archives identical447, checked by inventory.

No C export/kernel cost, whole changed-state/routing/generation/task quality,
accepted rate, actual DRAM or generality claimed. Original engine/source/binaries
intact. Full useful RAM-n/routing/LUT/DRAM/SAMEquality>=50/families/~100B goal open.
