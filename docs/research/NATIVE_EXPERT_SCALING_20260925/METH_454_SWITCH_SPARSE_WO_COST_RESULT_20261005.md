# METH-454 result: native arithmetic exact, both cost recipes fail

5 October 2026. Goal ACTIVE/INCOMPLETE. Scientific freeze2ba2a3c before first
import/compile; operational resumption8a9a376. ONE session70388, terminal exit0.
Raw `meth454_switch_sparse_wo_cost_result.json`,64,085B, SHA256
`efc087e32c2f88631654efaa4d0f30bf9b23288f66243deb51ed5f07a947d277`.
Scientific sources/protocol unchanged after observation. No454 rerun.

## Native representation/primal is now real

ALL eight apparatus gates PASS. Actual C bank472,257,280B, all128 real functions,
each coefficient/scale/sign recovered byte-exact from453, all fields/WO columns
64 aligned. Expert stride3,689,472B, expert0 offset4864; no new rounding/encoding.
Whole original source manifest/model mapped using the qualified unchanged loader.
Compiler/runtime/original C primitive/engine/source payload hashes verified.

All336 original C functions match raw/down/basis/A16 scales/codes captured source.
Both new candidate kernels at correct/ID+1 match453 at ALL672 functions each:
1,344 candidate records plus336 original =1,680 full local records byte-exact.
Independent actual basis reference336 calls, explicit Walsh F64 relative error0.
Source `mv`/A16/I8 dot are included unchanged from374; source is not a slow surrogate.

Tiny C qualification covers2025 signed/zero pair dots, all98,304 LUT entries/
sentinels, full3072 sparse signed-extrema/alternating/zero/single cases against
scalar I64, scale casts and explicit Walsh. Native reserved -8 fault exits2 as
declared, reason captured; this successful negative control is not a run failure.

## Actual cost overturns the logical-footprint prediction

One physical CPU0, actual group0/mask1/native thread verified for both processes.
Two warm/five measured WHOLE336 trace sweeps per arm; three-arm order rotates.
Every arm completes the trace before the next, preventing immediate same-query
direct/LUT weight warming.106 actual selected IDs, no copied bank or hot-function
shortcut. All7056 query outputs exact qualified primal; ordered timings retained.

| Kernel | Mean microseconds | P95 microseconds | Mean/source | P95/source |
| --- | ---: | ---: | ---: | ---: |
| Original I8 WI /I8 WO |237.68434523571425|255.41499795|1|1|
| Direct I4 WI /sparse original-I8 WO |234.427559525|279.22999805|.9862978535356021|1.0932404138016283|
| Pair-LUT I4 WI /same sparse WO |718.8211904738095|785.925|3.0242681307468793|3.0770510984396173|

Both candidates FAIL ALL three prospective cost gates: mean<=.80, every book
mean<=1.00, p95<=1.00 relative to original. Direct has measured mean about1.37%
lower, insufficient for the declared20% reduction; books18/19 are higher and
p95 is9.32%higher. No significance claim from the small mean difference. LUT's
book ratios are2.978..3.069; no outliers removed or extra repetitions/tuning.

Direct five336-call totals.078849400/.078849400/.078550200/.078953800/.078635500s;
original.079322000/.081047200/.079281400/.079798800/.079860300s;
LUT.241378900/.242094600/.241279200/.241695200/.241171700s.
Raw has all per-query/book/median/max/sweep values and exact times.

453's logical mean34.29%unique addressed footprint DID NOT produce the required
native speedup here. Reduced coefficient bytes and lost arithmetic costs are
different quantities. No hardware DRAM/cache-hit counters were measured, so
this result identifies aggregate kernel cost, not its sole physical cause.

## Algebraic constraints for the next diagnosis

WI still computes D*M=2,359,296 products, now decoding nibbles and reducing/scaling
12 blocks per row. Block scales/horizontal partial reductions are36,864 versus
3,072 full row outputs in source WI. Source and candidate WI both require147,456
AVX2 pair-multiply instructions at this shape in the written loops; precision
reduction saves coefficient bytes without eliminating the product count.
Exact instruction throughput/actual compiled bottleneck remains unmeasured.

LUT adds98,304 entry builds and1,179,648 indexed I32 reads each query. WO has fewer
actual nonzero-column products; aggregate timing cannot say whether WI decode/
scales, LUT build/gathers, original A16 quantizers, scan or WO implementation
consumes that gain. Measure components before selecting a different physical format.

Do not extrapolate the237.7us one-worker selected-FFN mean into391 whole3worker
accepted rate or376 source256 six-worker rate; prefixes/contexts/workers and
upstream/core/head/prefill differ. Matched complete-model cost fraction remains
necessary for deciding how much a FFN optimization can improve the full target.

## Resource, artifacts and decision

25.062s =7.828admission+17.234numerical, compile3.328s. Native primal1.3194862s,
native timing3.2098459s; both exit0, compiler no warnings/errors. Parent peak
1,362,747,392B; native peak983,953,408B; conservative sum2,346,700,800B.
Fourteen outputs519,350,747B, streamed hashing9,037,707,531B, zero updates/new
roundings/GPU/download/engine edits. Workspace465,424B includes393,216B LUT;
allocated direct/source shared test workspace and active reads are distinguished.

Bank SHA1e75f45a996833cdbd282a5f9285faf9d5e67cf36ed7fbaebf626a3125f2ef4c;
binary SHA c79ed58b859668dfce070e21a11092ed86779ad408c2e11e099f99efe97d1c7a;
primal43,915,224B SHAee57b04ae81c68e96e75d9f961445cdab8a733e0e12497734b8efab0744e1a9d;
timings571,977B SHAbcd9d5baa76de5b2a3bc0b9461d0ade23b138035bec0afe05bca7ec0efac4045.
Full paths/sizes/hashes/logs retained in raw. Engine/source/original binaries intact.

Close THESE direct/full-pair-LUT recipes before all-bank export. Arithmetic and
physical bank export are reusable qualified components, not a deployable candidate.
Next NEW component/whole-cost diagnosis, no threshold/kernel sweep. Full goal:
fresh donor-relative whole quality AND>=50 acceptedIDs/s SAME artifact, useful
RAM-scaled n/LUT/routing/winner AND mass/actual DRAM, multiple families/~100B.
