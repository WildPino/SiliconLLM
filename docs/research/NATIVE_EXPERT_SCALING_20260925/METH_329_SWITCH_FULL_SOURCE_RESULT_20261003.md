# METH-329: actual full-source native byte mapping PASS, probability gate FAIL

Freeze793352a. [Raw complete record](meth329_switch_full_source_result.json) SHA256
e76373359a81c00ffcbe4aa6d8684c83e17972c4eccbf0440d9c354c55cb419d.
[Protocol](METH_329_SWITCH_FULL_SOURCE_PROTOCOL_20261003.md), driver
benchmarks/native_expert_scaling/meth329_switch_full_source.py and source
meth329_switch_source_binding.c wrapping unchanged328 full arithmetic.

Original acquired full14,664,154,368 architecture-unique pretrained source,
all12x256 banks preserved, actual complete official4.57.6 model loaded from
META then Torch2.6 weights_only/mmap per shard, all6392names/F32/CPU/no META
remainder. Confirmed aliases tied; bound original fast tokenizer/sentinels.
No smaller student/approximated weights/missing banks or injected configs.

Small readonly native manifest584,405B maps original6ZIP storage records,
zero weight-copy bytes. Native Windows BCrypt SHA256 of EVERY6392 actual
mapped tensor EXACT327 expected bytes/name/shape/length, including all
unselected experts and four physical tied copies. No fallback weights.

## Complete actual source numerical controls

Both fixed consumed engineering contexts execute full native encoder and
cached12-layer decoder/all32128 logits, with original routing/probability/
capacity semantics. Results versus official source:

| Case | Encoder pooledL2 | Decoder pooledL2 | Full logit pooledL2 | Max probability abs | Exact choices/greedy |
| --- | ---: | ---: | ---: | ---: | --- |
|0 France|1.62082e-6|2.07474e-6|6.63773e-7|**1.28149986e-6 FAIL**|all PASS|
|1 experiment|4.82248e-6|3.38734e-7|3.32643e-7|9.08970833e-7 PASS|all PASS|

All pooled and every initial/block/final state satisfy original1e-4, all
route choices/capacity acceptance and greedy IDs identical. Both zero-head
faults detected. ORIGINAL selected-probability1e-6 guard remains FAIL for
case0. **Unchanged full-size bridge not qualified for precision/rate promotion.**
No threshold adjustment or claim that other passes override this stop.

113.203s main excluding imports, max checked controller+childRSS6,598,373,376B,
end controller4,601,769,984B. Native audit trims working set every64tensors
solely for resident-memory bound; no cache locality/rate inference. All native
case/state/fault/audit files and compiler/runtime logs saved in results/
native_expert_scaling/meth329_switch_full_source. Exec7607 exit0; all cases
preserved (normal diagnostic completion, not successful native qualification).

New motivated numerical variable: sourceD768 RMS variance currently naive
left-to-right F32 sum, while official CPU reduction order differs. A separately
frozen stable sum of SAME F32-rounded squares can reduce normalization drift;
do not modify329/328 or waive guard. Cause is a hypothesis until measured.
Different accumulation is exact representation/F32-output numerical semantics,
not learned precision quality evidence. If it fails, preserve before choosing
another justified variable. Native quality/accepted>=50/useful n scaling and
cross-family/~100B remain missing.
