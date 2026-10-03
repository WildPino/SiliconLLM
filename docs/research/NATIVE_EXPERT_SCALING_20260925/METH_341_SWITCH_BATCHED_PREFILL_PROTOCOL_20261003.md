# METH-341: exact integer encoder/cross-KV batching before unchanged cost gates

Prospective after340 original cost failure; preserve raw SHA
`a93863da56b79cc2d4ed94ed8712d712a42456c1ac2cced27f0091e53830ee32`.
3406-thread medians decode6.604/8.223ms for source9/64 pass20ms, but64 full
20.173ms fails and decode repeat ratio1.1547 fails1.10. Original candidate
remains unpromoted. Separate profile64t6: prefill core projections206.922ms,
experts63.832ms/router13.585ms; clocks included, not primary decomposition.

New variable ONLY execution schedule for encoder self Q/K/V and decoder
cross-K/V prefill. Normalize original encoder token states first (independent
per token), quantize actual F32 activation once per matrix/vector with SAME
336 rule, one row-parallel team per matrix; for each row/token call unchanged
exact AVX2 integer_dot and unchanged F64 scaling/F32 return. Reuse weight row
over all prefill vectors. No BLAS replacement, accumulation or precision change.
All decoder step projections, attention, routing/capacity, expert/feed, core,
head, weights, source n and context limits unchanged. No fitting/GPU/download.

Logical counter calls remain vector-projection counts, code/scale bytes count
logical per-vector addressed matrices; batching may change physical reads,
which are NOT measured by these counters. No physical DRAM saving claim.
One matrix profile clock per batch now includes all vectors and temporary
quantization/codes allocation. Profile outputs/counters keep previous scope.

Reuse frozen339 protocol EXACT fixture, order, threads, compiler/runtime,
timing/warmup/repetitions/budget and all numeric/cost/repeatability thresholds.
Fresh whole target hash; all source bytes/manifest SAME338/336. Mandatory
original336 two-case exact full output bridges for threads1/6/profile0/1.
Additionally BEFORE primary repetitions, BOTH complete340 source9/64,
32-position encoder/cache/decoder/fullhead/route outputs MUST have SHA exactly
equal340 at threads1/6. ALL subsequent repetitions/profiles ALSO exact340.
No selected subset, gate relaxation or attribution timing substituted for
primary wall. Main30min/combined32GiB, compiler120s, no other model jobs.

Both thread6 fixtures: median decode<=20ms, median full including prefill/32
<=20ms, decode repetition max/min<=1.10; all exact numeric/profile/descriptor
bridges mandatory. PASS only licenses separately frozen NEW untouched original
donor quality then accepted generation rate on SAME artifact. FAIL retains
all measurements and closes unchanged batched execution before quality work;
next variable must come from concrete remaining profile/evidence.
Useful higher n/LUT/real DRAM, donor quality and>=50 accepted generation remain
unverified. Forced32 continuation positions are a cost screen only.

Default engine body byte-exact; opt-in341 entry only. All code/protocol committed
and physical-byte audited BEFORE observations.

Command:
```
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth341_switch_batched_prefill.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth341_switch_batched_prefill_result.json
```
