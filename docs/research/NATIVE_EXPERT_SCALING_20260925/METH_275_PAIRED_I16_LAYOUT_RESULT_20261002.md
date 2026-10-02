# METH-275: paired/aligned layout conserves function, misses cost gates

Apparatus frozen `4c3ba4b`,session20495 exits0; no repair/retry/model job.
All24 original gate/up code matrices unpack byte exact; all313 nonpaired
segments retain hashes. All338 new segments/header/EOF read back exactly.
Native weight buffer and all paired layer starts are64-byte aligned with
no duplicate input-code arrays. All59,768,832 paired/scalar-I64 dots and
5,505,024 CPU/GPU input codes/12,288 parameters have zero discrepancies.

**All6144 complete native output vectors including header are bitwise equal
274**, SHA256 `0336e15c46e80f8d4d45816525e16cce8f34de44e635c04fac584f12257ca5ba`.
Both384 timing checks equal274; all256-state checksums79.317655639 unchanged.
These finite controls conserve274's measured source/reserve/GPU-native
fidelity,not fresh complete-model quality or universal arithmetic parity.

| Arm | Three ms/24-layer token passes | Median |
| --- | --- | --- |
| Contemporaneous unchanged274 |11.036016 /11.609710 /10.247170|11.036016|
| Paired/aligned275 |10.645058 /11.476697 /10.191785|10.645058|

Ratio .9645743536435613: observed3.5426% lower than control,short of5%.
Absolute10ms gate also fails. Close this fixed layout/cost recipe; do not
retime,widen gates or claim statistically general improvement.274/271/272
cost stops and267 semantic stop remain unchanged. No complete candidate
or native promotion follows from these component checks.

Fixture337,596,480bytes,+28 alignment bytes versus271/274,SHA256
`ba39a219a1fe4101eed6ec1e5066dea4939f1cd621eb50801d038e5a611d36d5`.
CPU-only31.422s after imports,harness RSS60,366,848bytes;native peak
365,359,104bytes qualifier/364,748,800bytes changed timing. Integer
qualification2.1225283s. LocalRyzen5 3600X,clang21.1.8/original flags,
six threads,no active job. Raw [result](meth275_paired_i16_result.json),
SHA256 `4d2b9bcb69c8b1d39d1d6a7c1090b212b129373b11322791edd376c3eea9a0c9`.
Commands/format/unpack/qualification/gates in
[protocol](METH_275_PAIRED_I16_LAYOUT_PROTOCOL_20261002.md).

## Next research scope

Repeated isolated-kernel tweaks do not establish complete transfer or the
user's joint quality/50tok/s requirement. The10ms FFN allocation is a
prospective component design gate,not a measured whole-model infeasibility
proof. Preserve every cost failure. Separately freeze a **diagnostic-only**
complete model with the validated I16/private128 function to test actual
composition and make a whole-model path reviewable. This explicitly revises
the earlier assembly-order assumption; it does not promote a failed kernel,
relax its gates or reinterpret its results. Fresh full-quality and actual
same-artifact end-to-end>=50 remain mandatory before any method promotion.
