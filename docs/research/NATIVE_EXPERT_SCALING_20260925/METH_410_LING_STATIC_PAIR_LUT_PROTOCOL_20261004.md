# M410: Ling256+16 static pair LUT and Q4-head complete cost gate

Freeze C/controller/engine-prefix/protocol BEFORE compilation/native outcomes.
408 first cost failure d49c951;409 attribution retained9bced64, raw SHA256
c54a20209b156c7ac8b1e4bf9c8dbda06972003535d28338d145dc083f7ea36a.
Six-worker coded12.452-14.396ms/head6.695-7.733ms/router1.516-1.580ms;
remaining controls2.224-2.682ms. No universal one-component exclusion. New
joint representation variable, not a repeated unchanged cost qualification.

## Frozen hypothesis and source scope

ALL values synthetic. Actual407 Ling header shapes remain:20 layers/D2048,
19 banks256/top8/FF512/shared512, first dense5120, fused GQA3072x2048,
attention output2048x2048, full untied157184x2048 head/embedding. No actual Ling
values/original reference/source fitting or quality. This is a necessary-cost
screen for a target representation before32.5GB source acquisition, not transfer.

Each bank has first256x8 and second16x8 I8 book, entries[-63,63], sums[-126,126],
row F32 scales. For16 coefficients store first indices A0,A1 and byte B0|(B1<<4),
3bytes/16. Static pair table4096x8 I8, index A+(B<<8),32768B/bank, precomputed
once for ALL14692 stored bank-matrix instances. No per-token table builder.
Reduced65536->4096 representable vector pairs, NOT exact408 coefficients.
Activation[-63,63] nearest-even, I32 AVX2 unsigned offset correction; new direct
packed two-book mode and pair-table mode share identical math/scales/fixtures.
Q6 head replaced by FULL Q4_K/Q8_K AVX2 head, source precision effect unknown.
Synthetic Q4 block has random code/packed scales, finite fixed F16 d=2^-9 and
dmin=2^-10; no random nonfinite half encodings. All full head rows filled/read.

407 pinned actual geometry and storage counts reused/freshly hashed; code fixes
one format, no grid/book-cardinality/head-precision tuning after observations.
All synthetic bank codes/books differ by deterministic seed, but distinct
synthetic values/labels are not trained useful experts or source function identity.

## Complete accounting and unchanged operators

| Conservative active component | Bytes |
| --- | ---: |
|Codes for779091968 coded coefficients|146079744|
|Row F32 scales|2560000|
|Original active books, charged in addition to pair tables|1209856|
|ALL556 active pair tables|18219008|
|Full Q4_K vocabulary head|181075968|
|Full F32 router/bias|39865344|
|Other F32 control weights|356352|
|One BF16 embedding row|4096|
|Total conservative descriptor|389370368|

Charge both books and tables in both modes although direct mode does not read
pair tables during execute, and pair mode does not read original books there.
This conservative format budget is NOT actual per-mode addressed/DRAM traffic.
The raw field addressed_descriptor_bytes carries this declared conservative
complete descriptor for continuity; do not call it measured physical traffic.

Stored coefficients15601762304, codes2925330432B, scales60461056B,
books31969792B, ALL14692 tables481427456B, Q4 head181075968B,
router/controls40221696B, embedding643825664B: complete4364312064B.
Allocated dynamic payload/output4326650368B (stored less static router/controls
plus matrix outputs2560000B); runtime/static arrays/RSS separately recorded.
Counters EVERY execution must reconcile coded coeff/scales/books/tables/router
and conservative descriptor. All stored pools allocated and filled before ready.

Source bridge requires quantizer, norm, F32 grouped router, exact complete
execute operator ordering and full initialization geometry/source serialization
byte-identical408 after explicit new table-counter/Q4-name/magic reversal.
Head weights and matrix packing/books/decode have NEW independent controls.
New engine opt-in prefix only, old409 engine reversal45c2386/40002afce0/default
0ff9705. Qualified original Switch374/389 binaries stay authoritative.

Include ALL attention projection/shared/dense/routed matrices/QK norms/full
router/mixtures/head. Independent layer/context fixtures, NO causal attention/
RoPE/KV/cache/composition/prefill/tokenization/generation/task. FULL operator
stream lower-bound screen, not accepted50/s. Fixture prepare/setup/table filling/
cryptoSHA/archive/printing/cleanup excluded; activation quantization/full routing/
all matrix math/head inside. No changed quality/rate inherited from metadata.

## Numerical, packing and archived-output gates

Both modes:32131 decoded/input cases,128524 four-row tile lanes,14641 mixed
adjacent pairs, extrema/zero/nearest-even/nonfinite/unsupported-S8 correction
negatives; independent scalar4448 bank/output samples,29384 all-bank code and
book edge checks. Integer relativeL2<=1e-6.19 grouped router/reference checks,
unbiased selected sigmoid normalized/scaled2.5, BF16 lookup2048 exact.
64 F64 independently decoded Q4 rows, relativeL2<=1e-5; Q4 bit mutation detected.

Pair-specific:131072 decoded packed lanes across all4096 A/B combinations,
32768 local pair-entry lanes,29384 table bank edges; ALL4096x8 entries for214
checked first/last bank instances =7012352 lanes. BOTH low and high code nibbles
mutated, pair table mutated, stale book/table inconsistency detected where
applicable; wrong row bias/palette/code/invalid bank controls retained. Complete
byte bridge supplements these samples; it does not replace independent errors.

One three-worker direct reference process, then balanced optimized processes
3,6,6,3,3,6. EACH3 repetitions of10 fixed inputs,0/1 warmup and2-9 measured.
ALL210 complete-output SHA must agree; archive first-rep ALL10 inputs per process,
70 full archives directly byte-equal across modes/processes/profiles. Includes
all matrix outputs/QK norms/mixtures/route IDs/probabilities/gates/head input/full
logits, finite values checked. Output magicS410OUT1, remaining408 schema unchanged.
Old408 SHA EXPECTED to differ and is not a changed-format math control.

Same qualified388 actual3/6 physical profiles[0,2,4]/[0,2,4,6,8,10], fresh374
Windows topology/setup/event/end worker IDs/groups/masks; negative injected
binding fault exits2 before allocation. Compiler/runtime hashes374, ACTIVE/
infinite/DYNAMIC FALSE/MAX_ACTIVE_LEVELS1/KMP_AFFINITY none/PROC_BIND absent;
sanitize inherited OMP/KMP/GOMP/worker fault settings. Exact403 known unrelated
tiktok_publisher daemon exception, record and leave untouched; no model overlap.
Core group0 is not cache topology or DRAM evidence.

## Prospective decision and resources

Optimized mode only: each profile has nine rep medians and24 fixed-input medians.
Require ALL<=14ms AND pooled max/min<=1.10 in at least ONE profile. Direct mode
is a numerical byte-control; its three rep medians cannot select the performance
profile or claim speedup causality. Retain all optimized observations, no deletion/
retry. A pass permits ONLY a new source-aware small-second-book/Q4 precision
feasibility protocol, then actual source-derived whole quality and SAMEartifact
FULL>=50 accepted batch1 IDs/s. Fail closes THIS joint format before fitting.
No easing gate, no automatic larger synthetic bank sweep, no useful-n claim.

MAIN<=600s, compile<=120s, each native<=120s, checked combinedRSS<=6GiB,
output<=2GiB. Expected~1min/~4.4GB. Local CPU only/no network/weights/GPU/T4.
FIRST apparatus/outcome retained immutable before a new numbered repair/variant.
Final project goal multiple-family transfer and RAM-scalable useful experts/
CPU LUT/routing/quality/physical DRAM remains incomplete regardless of this gate.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth410_ling_static_pair_lut.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth410_ling_static_pair_lut_result.json
```
