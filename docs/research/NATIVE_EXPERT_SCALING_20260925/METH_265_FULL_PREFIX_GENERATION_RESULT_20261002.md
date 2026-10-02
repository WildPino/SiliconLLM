# METH-265: same-archive full-prefix generation health passes

Frozen at6c21c66, no repair or candidate/data/threshold changes. Same259
complete archive and26124 sources, original BF16 donor/E1280 controls,
full-prefix recomputation with no KV cache. All72 responses completed.

| Arm | EOS/24 | Early non-EOS<16 | Repeated8gram3x |
| --- | --- | --- | --- |
| Original BF16 donor |22|0|0|
| Original BF16 E1280 |21|0|2|
| Actual saved unique E1280 |22|0|1|

All frozen pooled/category health gates pass. Every candidate generated
state retains full-head argmax in fixedK64 and exact-row reranking agrees;
no omission/mismatch. Raw full head always chose the actual token. This
finite result is not a universal shortlist guarantee or semantic pass.

Raw [result](meth265_full_prefix_generation_result.json), SHA256
`f3581cefc4a7d758b141fc4b4fa62ca45f43d00fe6cc00a10af57895b4a5ec93`.
Session43037 exited0,531.656s after imports, endRSS2,543,906,816 bytes,
peakGPU3,081,579,520 bytes. Stored artifact SHA remains
`3c0949fb2b7f99dc887c6aec1041aa162b35801939349966d18e52b8807a6f71`.

Proceed with pre-frozen266 full PIQA regression and267 anonymous semantic
assessment. Labeled continuation texts have not been read for review.
Keep264 cached-reference stop closed. Native cache arithmetic, CPU
whole-model quality/routing/LUT/alias lookup/real DRAM/accepted>=50tok/s,
useful RAM-scale n and multiple donor families/scales are still required.
