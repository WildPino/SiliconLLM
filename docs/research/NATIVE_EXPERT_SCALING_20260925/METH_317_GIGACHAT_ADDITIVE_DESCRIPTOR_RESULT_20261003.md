# METH-317: full-width additive books pass descriptor only

Freeze `4c6e8ae`, exit0,1.562s/endRSS421,822,464bytes (imports included).
Both actual GGUF header hashes/shapes/types match; no weight payload read.
Original widths32 MLA heads/8960 dense/1280 routed/shared preserved.
Two256x8 I8 palettes per bank, two U8 indices/eight weights (2bits/weight),
F32 row scale; headQ6_K, router/normF32, embeddingBF16 retained.

| Addressed component at actual64 parents | Bytes/token |
| --- | ---: |
| Encoded codes | 357,154,816 |
| Row scales | 5,327,360 |
| Independent palettes, including all selected MLA banks | 8,683,520 |
| Unchanged head/router/norm/one embedding row | 171,821,312 |
| Total | **542,987,008** |

This passes the prospective560MB descriptor yardstick. Decoder still performs
1,428,619,264 encoded integer coefficient products plus full original head/
router per input-ready fixture.357,154,816 logical palette-vector loads are
not assumed cached or equated to physical DRAM. Passing bytes is insufficient.

| Parents/layer | Flat addressed bytes/token | Flat stored bytes | Hypothetical hierarchy addressed / stored bytes |
| --- | ---: | ---: | ---: |
|64 actual source geometry |542,987,008|3,191,834,368|542,987,008 /3,191,834,368|
|640 ANALYTICAL ONLY |631,518,208|24,926,906,368|543,019,008 /24,838,887,168|
|6400 ANALYTICAL ONLY |1,516,830,208|242,277,626,368|543,307,008 /241,308,903,168|

Post-result arithmetic audit: the immutable raw317 larger FLAT scenarios
kept the original64-element correction bias. A true flat n-way router also
requires25*n F32 bias values. The table above adds6400*(n/64-1) bytes to
raw flat active/stored totals:57,600bytes at640,633,600bytes at6400. These
are explicit derived corrections, not a rerun or changed gate. Actual64
source accounting and hierarchical coarse64 bias remain unchanged.

Flat routing alone fails560MB at640 and6400. Hierarchical64 coarse groups,
four selected local16D BF16 key scans can bound additional addressed key bytes
in this calculation, but no hierarchy has been trained or executed. Larger
banks contain no sourced additional weights/capacity.6400 cannot fit the local
approximately80GiB RAM budget;640 may fit, without establishing useful experts.

[Raw descriptor](meth317_gigachat_additive_descriptor_result.json)
SHA `93e778d724490138ce1b187f9f887c8e86d78778d6d0c67a15d1ace71edc2e83`.
**Decision:** eligible only for a NEW complete native decoder preflight at
actual64 geometry; no book training until numeric/cost/stability gates pass.
No physical DRAM/native latency/whole quality/accepted rate evidence here.
Bounded-I8/S7 format is a proposal inspired by [AQLM](https://arxiv.org/abs/2401.06118),
not its implementation or a transfer of its reported results. Old IQ2/Q2
quality failures and301/302 activation-LUT cost failures remain closed.
