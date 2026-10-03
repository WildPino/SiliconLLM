# METH-303: compact I8 cost fails before teacher collection/training

**Stable native cost FAIL; numeric/address/router controls PASS. No student
training, new capture or quality/rate promotion.** Frozen protocol/source
commit `c7439d8`; terminal exit0. This is a source-bound input-ready operator
fixture in actual phase60, not causal inference or accepted tokens.

## Actual resident bank and executed work

64 original GigaChat macro parents, ten synthetic128-wide nonlinear children
per parent:640 functions/layer. All9,437,184,000 routed coefficients physically
allocated/populated; total I8 matrices store9,651,773,440 coefficients. Active
273,571,840 coefficients and1,902,656B row scales reconcile all308 matrices.
Original Q6_K full128256-row head, F32 routers/biases/norms freshly read/hash
checked in Python and C:171,818,240 component bytes. Complete descriptor
weight447,357,600B/token passes560MB; physical DRAM traffic is unmeasured.

Timing includes integer projections/scales, activation conversion, source
norms, source macro routing,16D child query/ten-key selection, selected bank
addressing, complete dense/routed/shared SwiGLU/down/mixture and actual Q6
full head/Q8_K input conversion. Attention contexts are input-ready: causal
attention/RoPE/cache/embedding and layer-to-layer residual model composition
are absent. Synthetic children establish no knowledge transfer/useful n.

## Frozen gates

| Gate | Observed | Result |
| --- | ---: | --- |
| Complete addressed weight<=560MB |447.358MB |PASS |
| Exact scalarI64/AVX2I32 and scaledFP32 rows |7,176 exact |PASS |
| Every stored bank first/last code edge |97,194 checks |PASS |
| Decoded I8 relativeL2<=1e-6 |3.630782203605679e-8 |PASS |
| Real Q6 head64-row relativeL2<=1e-5 |1.742025356166587e-7 |PASS |
| Source macro/child ranking, probabilities/gates/address binding |25 layers exact |PASS |
| Signed extrema/zero/ties/NaN/bad-bank/packed-head fault |all pass;sentinel row0 byte0 |PASS |
| All30 complete output/route hashes, three same-input sequences |exact |PASS |
| Median repeatability<=1.10 |1.0213557332828023 |PASS |
| EACH operator median<=14ms |31.267950/31.822250/31.935700ms |**FAIL** |

10 fixtures×3 repetitions, each two warmup/eight measured. Selected-child
union37–40/640 per layer across ten inputs; these are actual addresses, not
learned exposure evidence. Allocation9,932,176,512B; peakRSS9,953,144,832B.
Initialization2.537s, native3.627s, controller11.172s including compilation.
Local Ryzen5 3600X/six threads, no overlapping model job/GPU/T4. Independent
recomputation of three medians/all30 hashes/spec/log bindings passes.

The14ms gate remains unchanged: full target also needs omitted model costs.
Do not collect or train this unchanged native implementation. Cost attribution
of the unchanged math is the next useful action before a new implementation
or geometry is selected. This result does not refute every compact nonlinear
geometry, integer kernel or scheduling policy; no source quality was tested.

## Bindings and reproducibility

- [raw result](meth303_compact_i8_preflight_result.json), SHA256
  `396edaee9c201324ffcb35eb57cadfb915a4ea6899694ca9e1055d8c29dd968b`.
- Controller `3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765`;
  C `eee66947fec515b1f7805a20bd7de2be7b92e665c8a0c4602d1a7a80e2fcdcf5`;
  phase60 `332a198a293d0575f430d1ccdd38559be776ae67c61b3ad42e01bfec0196c2ea`.
- Executable `8a4f0968950cd81651188ce987a89aa0cf663c0b582dedeb7e63a233e517aca8`;
  spec2000B `f11170ce458f9ddce2bcebd977bb0c8dd4feaeb3b7260948e9640b461b5692c5`.
- Stdout `results/native_expert_scaling/meth303_native_run1.jsonl`, SHA256
  `9c799d3b9343561f9052b9857cd8b354a2df924e049d21cbfd7d7f0028f1a832`;
  stderr empty. Toolchain/compiler/runtime pins retained in raw result.

Frozen working bytes match filtered HEAD. Reversing only the new compile
selector restores preceding `engine.c` exactly. Old helpers are unchanged.
Reproduction command and resource gates are in the [protocol](METH_303_COMPACT_I8_PREFLIGHT_PROTOCOL_20261003.md).
Useful RAM-scale capacity, independent donor-relative composed quality,
SAMEartifact accepted>=50token/s and multiple-family/scale transfer remain open.
