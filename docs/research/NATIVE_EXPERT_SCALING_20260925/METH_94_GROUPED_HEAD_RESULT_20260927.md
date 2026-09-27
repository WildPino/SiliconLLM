# METH-91/94: grouped-head refinement fails development composition

The [frozen protocol](METH_91_GROUPED_HEAD_PROTOCOL_20260927.md) spends part of the METH-85 core's byte margin on group-128 FP16 scales for the tied R8 embedding/output head, while retaining group-64 R8 FFN, BF16 attention, FP32 controls and the same product-key E128 geometry. The original Instruct donor source SHA256 is `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

The [METH-91 export](meth91_grouped_head_core_export.json) binds a stored core SHA256 `6f994c9ddf047adc789d3e1c841dd3be156cc578ed0d6c3f4045f97212e2accb`. All 363 tensors reload exactly, and BF16 head/FFN reconstructions from reloaded codes/scales equal their pre-save versions. The file is 550,260,576 bytes. Ideal addressed payload with E128 router and selected factors is **558,625,536 bytes/token**, 1,374,464 below the 560 MB design cap. This is byte accounting, not DRAM or native-rate measurement. Export took 13.67 s and ended at 4.78 GB process RSS on the local RTX 3060.

The frozen METH-88 B-only distillation recipe was repeated from the original METH-56 update-512 E128 bank on this exact core. The [METH-93 report](meth93_grouped_head_b_adaptation_result.json) binds all 128 raw/chat draws and 64 updates; the resulting checkpoint SHA256 is `c11b67438b95c22e80a716f62f79b8fa0d8d73a9ed7fdd1421be04f8bbb6e174`. A factors and product-key router remain unchanged. Training took 61.59 s and peaked at 3.47 GB allocated GPU memory.

The [METH-92 manifest](meth92_grouped_head_dev_manifest.json), SHA256 `b07a40ce4739c943b22cdb984238a09ef9142910c83ed0369e2945e4b3262556`, fixes 24 new source-disjoint documents and prompts, excluding all prior IDs/fragments through METH-90. The [raw METH-94 score](meth94_grouped_head_composition_result.json) compares seven arms on the same 98,280 bytes and 4,504 prompt positions:

| Arm | BPB | Donor top-1 agreement |
|---|---:|---:|
| BF16 donor | 1.238344 | 100.000% |
| BF16 + original E128 | 1.233654 | 94.760% |
| METH-85 old core, experts disabled | 1.238548 | 93.206% |
| METH-85 old core + original E128 | 1.233761 | 92.873% |
| METH-85 old core + METH-88 adapted E128 | 1.233861 | **94.161%** |
| METH-91 grouped-head core, experts disabled | 1.238381 | 93.739% |
| METH-91 grouped-head core + METH-93 adapted E128 | 1.233819 | **93.384%** |

The head refinement improves the expert-disabled core by 0.533 point, but the recomposed adapted model loses 0.777 point versus the old-core adapted comparator. Its pooled document delta is −0.004525 BPB and its expert utility on document loss passes. Three of seven frozen gates fail: pooled top-1 93.384% <95%, pooled top-1 1.376 points below BF16+E128 rather than at most one point, and top-1 gain versus old-core adapted is negative rather than at least +0.5 point. Every category prompt floor and document gate passes. **Stop this grouped-head variant before external quality or native integration.** The result is an interaction between head precision and B-only adaptation under one recipe, not evidence that finer head scales are intrinsically harmful.

No METH-91 C decoder, LUT timing, full-model `engine.c` execution, semantic pass or ≥50 accepted tok/s result exists. The METH-92 prompts are now viewed. A future compact-core candidate needs a new adaptation mechanism and independent source set; the separate large-E learned-quality/routing constraint remains unresolved.
