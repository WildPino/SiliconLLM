# METH-222: conditional maps align with state but tenfold capacity worsens validation

The [protocol](METH_222_CONDITIONAL_FUNCTION_PROTOCOL_20261001.md) and
runner were frozen at `51b2bf4` before source capture or fitting. This
changes geometry, learning actual pinned donor layer12 FFN output from
its actual input: an affine common, shared whitened PCA64 feature and
hierarchical affine residual cells. It does not preserve a full donor
MLP as the active core, select donor neurons, or duplicate weak experts.

All 640 selected raw training rows are distinct, with 512 fit and 128
validation windows, 128 tokens each. They exclude the same 352 old
chat/audit rows. These sets are raw-row disjoint, not document-disjoint
external quality. Capture and all eight stored parameter tensors read
back exactly. All160 cells are occupied and >=90% have >=65 fit states,
passing the frozen fit screen. Counts range **2–1154**, mean409.6;
state count is not independent exposure. PCA64 retains only **48.607%**
of centered fit input variance, not teacher output or residual variance.

| Validation arm | SSE / actual teacher output energy |
| --- | ---: |
| Affine common only | 0.466232462 |
| E16, one cell active | **0.439157627** |
| E160, one cell active | **0.469383645** |
| E160, selected leaf rotated within parent | 0.963465526 |

E160 is aligned to inputs/functions relative to its rotated control,
but has **6.88% higher SSE than E16**, and slightly exceeds common-only
error. It fails <=0.01 absolute normalized SSE and >=10% improvement
over E16. Paired validation-window normalized SSE gain over E16 has
bootstrap P05/P95 **-0.035109477 / -0.025487908**, both negative.
The function/route alignment gate passes; it cannot override quality
or useful increased-capacity failure. Do not declare arbitrary-n
quality degradation, or usefulness from fitted slots alone.

**Decision:** stop this fixed affine-common/PCA64 hierarchical-affine
geometry. No full-model substitution, native kernel, fresh quality,
generation/task or larger-n training is licensed. Fit versus validation
error has not yet been measured; overfitting or insufficient nonlinear
representation are hypotheses, not established explanations. Quantify
that gap and support strata from saved states/weights before choosing
the changed nonlinear shared/specialist training mechanism.

[Raw result](meth222_conditional_function_pilot_result.json), SHA256
`41858ede5147f7c98821987892096acb0aa7abf3611db4a1b849663e1d0ddb34`.
Capture:293,935,332 local bytes, SHA256
`2ba4b548f189f548a954bbaf595e192e982da6498c0d230fbce1f11597773ef5`.
Physical parameter artifact:22,270,608 bytes, SHA256
`1fcb16d51e9bbe47d55f5df648ba89c027018a95da95d0577408021b2550e929`.
Runtime18.907 s after imports, ending RSS2.364 GB, GPU peak2.005 GB,
local RTX3060, six threads. Session96000 exited0; no T4 or remaining
project inference job. METHOD's independent full quality, actual C/LUT/
DRAM at large n, multiple families/approximately10B and >=50 accepted
tok/s requirements remain unverified.
