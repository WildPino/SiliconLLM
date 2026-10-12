# From composed recurrent coordinates to a cheap causal operator

13 October2026. **PROPOSED / UNIMPLEMENTED / UNBOUND.** Native implementation
requires completed independent [composition qualification](RECURRENT_AXIS_COMPOSITION_RESULT_20261013.md).
This is a step toward the chatbot-to-engine pipeline, not a selected complete
student or an hours-long adaptation allocation.

## What the composition can establish

At site23 fixed FIT channel384 and state96 maps produce a36,864-coordinate
recurrent state,versus786,432 source coordinates and49,152 in the original
DN512/N96 engine layer. Variable channel widths retain all48 source head
decays. The source scalar-head decay is representable by repeating the same A
across each head's channel/state coordinates. This establishes compatible
recurrence algebra only;source feature functions and output geometry differ.

Same-time gating and RMS do not permit simply moving the channel map into
fixed output weights. If psi is the384-dimensional decoded recurrence,
Vc is the block-diagonal3072x384 channel decoder,s=D*x,and g=SiLU(gate),

`u = diag(g) (Vc psi+s)`

`o = O diag(gamma) u / sqrt(u^T u/3072+eps)`.

The effective recurrent readout `O diag(gamma*g) Vc` varies with each token.
Rebuilding all its2048x384 entries densely costs2048*64*384=50,331,648
multiply-add terms before applying it,greater than the original6,291,456-term
out projection. Precomputation is therefore not a cheap exact fix for arbitrary
g. Full skip adds its own donor functions;normalization includes recurrent/skip
cross terms,not independent branch norms. Quantization adds another contract.

Source x,B,C are produced after convolution and SiLU. General mixed projections
do not commute with these nonlinearities: `W^T SiLU(z) != SiLU(W^T z)`.
Folding W into the source input/conv weights alone changes the function.
Coordinate selection is an exact special case for selected independent
channels,with a different approximation error than the dual mixed basis.
No gate-independent low-rank decoder or closed compact feature generator is
admitted by the current scan assay.

## Exact retained weight counts, not latency estimates

Direct header inspection of the already hash-bound Falcon-H1 source checkpoint,
`model.layers.23.mamba` (revision80ebc50d), gives:

| Operand | Shape | Distinct coefficients | Stored BF16 bytes |
|---|---|---:|---:|
| in_proj | 6704x2048 | 13,729,792 | 27,459,584 |
| out_proj | 2048x3072 | 6,291,456 | 12,582,912 |
| conv1d | 3584x1x4 | 14,336 | 28,672 |
| gated norm | 3072 | 3,072 | 6,144 |

The convolution includes3072 x plus256 B plus256 C channels;gate3072 and
delta48 are additional in_proj outputs. Reducing the state axes alone leaves
these20,021,248 dense in/out projection coefficients in one source layer.
Numbers are arithmetic from stored shapes,not measured DRAM traffic,latency,
cache residency or a50-token/s impossibility proof. Other layers/attention/MLP/
head are additional costs. Full source weights/config are bound by the
[composition binding](recurrent_axis_composition_binding_20261013.json).

## Selected next bounded experiment

Implement a separate C pilot that consumes the exact admitted site23 maps and
actual captured source operands. Preserve original engine.c and quality corpus.
No new fit,source forward,RESERVED query or native chatbot claim.

1. Export fixed physical maps/A/D and histories with input hashes. Implement
   per-token384x96 state update,decode/full skip,source gate/full joint norm and
   trained output projection. Keep original source-denominator oracle separately.
   Every output depends only on prior state and current supplied operands.
2. Qualify ALL48 histories/every timestep against independent explicit CPU
   equations. Preserve original smooth/controlled-BF16 references. Native F32
   arithmetic must have its own declared numerical controls plus the SAME local
   source-quality gates;previous failed F32 precision bridges remain failed.
3. Measure batch1 recurrence,decode/gate/norm,out projection and total supplied-
   operand operator time,including state traffic and physical weight layout.
   Isolate timing from qualification and record concurrent process/clock state.
   Counts of retained source generators remain explicit;injected-operand rate is
   not an end-to-end useful token rate.
4. Count/measure retained input-generator kernels separately. A synthetic-input
   GEMV timing can inform that kernel cost only;it cannot qualify source history
   or conversion quality. Include map projection overhead,conv buffers and
   dynamic gate/fullskip readout before considering a cheap native extension.

Proposed caps:prepare/qualify/timing each<=900s,audit<=1200s,OS6GiB,
artifact output<=2GiB on E,no GPU/T4. Actual source/compiler/runtime binding,
gate tolerances,timing protocol and launch must be frozen before measurement.
Full audit even numerical/quality failure. If unexpectedly hours-long,follow
the user's announce/bind/launch/PAUSE instruction rather than delegate.

## Decisions and open routes

- Cheap recurrence plus expensive retained projections:target the source
  feature/gate/skip functions with conditional nonlinear transfer,not another
  indefinite covariance/gradient refinement of the same state map.
- F32 numerical failure:retain failed artifact,locate reduction/quantization/
  conditioning error before using its timing for any accepted component claim.
- Useful local cost+quality:compose with compact hidden-state/generator maps,
  source attention/FFN transfer,then validate autonomous histories and all layers.
  Site0/12 remain unresolved;site23 cannot substitute for them.

An approximate generator must learn its OWN causal update and conditional
functions under donor distribution/output supervision. A representation fit
with supplied x/B/C/g cannot certify that closure. Multiple families,useful n,
CPU IDs AND masses,actual expert DRAM traffic,conversion masters/gradientsO(n),
and>=50 useful batch1 tokens/s on the SAME complete artifact remain required.
