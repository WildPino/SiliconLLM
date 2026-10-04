# METH-450: observed flips cross the state/readout decision plane

**ALL8 apparatus PASS and JSON adapter qualified. Goal ACTIVE / INCOMPLETE.**

Frozen6d5beb2, resumptionee18d2b before first import. ONE session29978, exit0,
unchanged449 math/formulas/data/thresholds; only scalar JSON adapter and numbered
bindings changed. First449 serialization AND failure-writer stop retained
dd6650b as external terminal metadata; no449 scientific result reconstructed.

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth450_switch_margin_diagnostic.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth450_switch_margin_result.json

[Protocol](METH_450_SWITCH_MARGIN_PROTOCOL_20261005.md),
[raw](meth450_switch_margin_result.json) SHA256
492e7328a8efd53aa6c3a79796781ec7a2a2a885eac444b80f463492068e5747.
Source449 math SHA c19cc705279fd49f0f3bf4a9eee31b1df067c56ba23cdcdaa21e559fb422b45e.
All parent/source/helper/runtime hashes and raw first-stop bindings recorded.

## Scope and independently replayed arithmetic

ALL336 consumed original source128 teacher-prefix positions, books18..23,
same original/447row-I4/448block64-I4 logits and saved original/candidate head
inputs. Source tied I8 head unchanged.2688 selected original/candidate head values
replayed through independent I64 dot/F64 scale/F32 cast BYTE-EXACT.
ALL original A16 head-input scales/codes exact. No FFN, complete-head/model
forward, candidate, fit, optimizer or quality-gate change.

All672 pair-margin identities<=1e-12; KL/cumulant identities and state/A16/
cast/order decompositions<=1e-10; independent tiny algebra/manual integer
qualification and saved archive byte-exact. Maximum F64 reduction-order residual
5.7211180e-15(row) and6.9935377e-15(block). No arithmetic discrepancy found.

## Native decision plane and information are distinct constraints

At each position a=original winner; paired b=candidate winner when changed,
otherwise original runner-up. For original logits z0 and candidate zc:

    m=z0[a]-z0[b]
    d=(zc[b]-z0[b])-(zc[a]-z0[a])
    zc[a]-zc[b]=m-d.

Each actual changed candidate in BOTH arms is the original runner-up. ALL4row
events and ALL9block events have a positive smooth original pair gap and a
negative smooth candidate pair gap. They cross that pair's decision plane even
in the source head's linear F64 shadow. Zero events in the other three frozen
classification categories: no source shadow disagreement/tie, shadow candidate
pair preserved, or near-zero candidate smooth pair within1e-10.

This identifies real head-input STATE displacement as sufficient for these
selected-pair crossings; A16 head quantization/F32 output casting are not needed
to produce these particular crossings. They still perturb margins and may matter
in other cases. Paired shadow comparisons do NOT identify the full smooth-head
global argmax or whole model/generation/task quality.

Source original top2 gaps, using bins frozen before observation:

| Native original top2 gap | Positions |447 row flips |448 block flips |
| --- | ---: | ---: | ---: |
| [0,.001) | 0 | 0 | 0 |
| [.001,.01) | 3 | 1 | 3 |
| [.01,.1) | 21 | 3 | 6 |
| [.1,1) | 85 | 0 | 0 |
| [1,infinity) | 227 | 0 | 0 |

All observed changes occur among24 positions with margin below.1, yet there
are no exact/near-zero(<.001) original native ties. No semantic ambiguity or
harmlessness is inferred from a small numeric gap.447/448 remain local FAIL.
Overlap: indices30and217 changed in both;86and298 row-only;
58,90,211,251,257,313,332 block-only.13arm events across11 unique positions.

Two illustrative decompositions, observed raw values:

| Arm/index | Original gap | State contrast | A16 contrast | F32 cast contrast | Candidate gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| row/217 | .0232577324 | .5206414311 | -.0008337417 | -3.5069310e-7 | -.4965496063 |
| block/251 | .0032424927 | .0203251907 | .0030900319 | 3.4295116e-7 | -.0201730728 |

The first is a large state crossing. The second shows that head A16 contributes
to the native contrast but the selected pair also crosses in the smooth shadow
(original.0020184519, candidate-.0183067388). No norm-only attribution used.

KL constrains log E_p exp(delta)-E_p(delta), with leading local term .5Var_p(delta),
whereas argmax compares directional differences against the source gap. Lower
mean KL/Frobenius energy need not make every such directional contrast smaller.
448's improved meanKL but increased winner changes is therefore compatible
with the qualified algebra; further scalar blocksize sweeps are not justified.

## Diagnostic certificates: require the original reference

| Sufficient certificate with frozen tolerances |Row certified unchanged |Block certified unchanged |
| --- | ---: | ---: |
| Constant-centered logit Linf range |261/336 |277/336 |
| Posterior total variation |293/336 |303/336 |
| Pinsker via source KL |287/336 |297/336 |

All exclude every observed flip. These use original logits AND candidate values/
posteriors. They are diagnostic bounds, not a deployed cheap selector, fallback
policy or routing acceleration. Missing certification does not prove a flip.

## Decision and next mechanism

Close head-input A16/F32 readout rounding as the SOLE explanation/repair for
these observed paired flips. Preserve448/447 failed encodings; no gate relaxation,
precision/blocksize/seed grid, validation fitting or reinterpretation as quality.

Next proposed NEW451: change coefficient basis before SAME block64-I4, using
one fixed signed block-Hadamard transform, all coefficients and private ReLU
retained. [Prospective algebra/primary literature](SWITCH_ORTHOGONAL_I4_NEXT_20261005.md)
specifies the new variable and exact-real equivalence. A rotated UNCOMPRESSED
control must separately qualify new activation arithmetic before low-bit quality;
neither real orthogonality nor papers establish native A16/C performance here.
No451 source/protocol/transform/output exists at retention. Freeze before math.

## Resources and retention

9.625s total:8.391s admission +1.234s numeric/reporting. Peak186,388,480B;
8,919,526,847B hashed; CPU0/NumPy2.4.6/BLAS1, no Torch/GPU or new source data.
Zero updates, zero FFN/complete-head forwards,2688 selected rows replayed.
All sessions terminal. Source/engine/original374/389 binaries preserved.
margin_inputs.npz5,690,044B, SHA256
6ddf528f4cdecf8522ab1c3bc694de0eafb4ff70c5d47a7fdafd8fe3a77ef036,
retains original/row/block head inputs/A16/scales and selected original head rows.
Per-position full identities/decompositions/information/classes in raw.

Full goal still requires useful RAM-scaled n, routing/LUT/actual DRAM, whole
source-relative held-out/generation/tasks AND>=50 acceptedIDs/s on SAME new
artifact, additional families and actual~100B applicability as resources permit.
