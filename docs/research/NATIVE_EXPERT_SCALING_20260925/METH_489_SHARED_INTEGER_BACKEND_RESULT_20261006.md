# METH489: exact shared integer backend; fixed GPU layout fails speed

6 October 2026. Goal ACTIVE/INCOMPLETE. The complete staged inquiry is admitted:
all eleven inherited integer controls and all 192 teacher/own-state cases pass.
The fixed synchronized per-operator GPU layout fails every prospective speed
and matched-benefit gate on both source artifacts. It is closed at that gate.

## Scope and reproducible objects

The candidate is the unchanged C486 implementation, compiled during METH488.
CPU0 and GPU1 use the same executable and the same payload for each comparison.
The 169 shared attention/dense-FFN/head matrices run in integer cuBLAS in GPU1;
original CPU routing, sparse experts, normalization, quantizers, cache and
state updates retain their original operations. No fallback or new precision.

| Source | Distinct pretrained parameters | Original expert IDs per sparse bank | CPU affinity |
| --- | ---: | ---: | --- |
| Switch n128 | 7,415,217,408 | 128 | 3 physical cores: 0,2,4 |
| Switch n256 | 14,664,154,368 | 256 | 6 physical cores: 0,2,4,6,8,10 |

Hardware: Ryzen 5 3600X, 80 GiB RAM, RTX 3060 12 GB; actual CUDA runtime 12.4.
Distinct parameter counts are retained in the [source storage
derivation](ALGEBRA_454_STORAGE_DERIVATION_20261005.json).
These are different trained weights and different CPU affinities. Their
comparison is not a causal experiment changing only n, nor a second family.

Each source has the original 24 books x 4 cases. Teacher length is 14;
own-state generation cap 64, stop ID 32095, batch 1, one warmup and three timed
repetitions, both original profiles and the frozen order. Donor-quality
references are the original qualified METH387/METH363 cohorts; generation
references are retained from METH458. No new donor inference or new books.

The [prospective protocol](METH_489_STAGED_COMPLETION_PROTOCOL_20261006.md)
and [binding](meth489_binding.json) specify every source, runtime, payload,
reference and executable digest. Apparatus freeze `767944d`, binding freeze
`8ff9c60`; the original C and engine are unchanged.

## Arithmetic and complete output evidence

The inherited eleven controls independently verify I32 digit products, exact
I64 reconstruction and original F64 postscale/F32 cast, including source-valid
extremes, padding, cancellation, zero and actual core/head shapes. The backend
decomposes signed q16 into three I8 digits plus a zero fourth column:

```
q = d0 + 128*d1 + 16384*d2
d0 = unsigned16(q) & 127
d1 = (unsigned16(q) >> 7) & 127
d2 = (unsigned16(q) >> 14) - 4*(unsigned16(q) >= 32768)
```

The fourth column is an internal digit layout, not four requests. Source A16
range remains [-32767,32767]; the original CPU AVX lane proof must not be
extended to q=-32768 combined with weight=-128 at width 4096.

All 3,456 whole output files, 1,728 per source, match the original qualified
source bytes: encoder/decoder states, heads, route IDs, acceptance, routing
probabilities and generated IDs. Counters and work/transfer equations also
pass the independent audit. Both original complete cohorts inherit ALL18
source-to-donor quality gates through that exact output identity. This is a
valid transitive result on the same data, not a new independent quality set.

The audit imports neither the main controller nor its check helper. It reads
every retained wire file and rederives NLL/argmax/health/counts/time/rates and
source-quality transitivity. All six retention gates and all five final
admission gates pass. Arithmetic correctness does not imply speed.

## Primary profile0 whole performance

Rate = accepted generated IDs / SUM of all 96 case mean whole-generation
times. Every rejected case keeps its time in the denominator. Ordinary and
the stricter prose filter are both retained. Three timed repetitions are
averaged per case. The original prescribed bootstrap uses 10,000 book
resamples, seed 485485, lower/upper 5th/95th percentiles. The historical JSON
field names `warm_lower95` and `warm_upper95` therefore denote these two
quantiles, not a central 95% interval. No seed or timing selection occurred.

| Source/backend | Whole time sum (s) | Ordinary accepted IDs | Ordinary IDs/s [5%,95%] | Prose accepted IDs | Prose IDs/s [5%,95%] |
| --- | ---: | ---: | --- | ---: | --- |
| n128 CPU0 | 12.06327577 | 1,142 | 94.6675 [92.6189,96.6428] | 662 | **54.8773 [53.4759,56.2270]** |
| n128 GPU1 | 32.40238673 | 1,142 | 35.2443 [34.7060,35.7525] | 662 | 20.4306 [20.0277,20.8153] |
| n256 CPU0 | 13.62003107 | 895 | 65.7120 [61.7363,69.7839] | 490 | 35.9764 [33.7592,38.2470] |
| n256 GPU1 | 37.08519383 | 895 | 24.1336 [22.8151,25.4471] | 490 | 13.2128 [12.4969,13.9320] |

All 96 n128 cases and 81 n256 cases are generation-healthy under the original
criteria. Exact outputs keep those counts equal between CPU0 and GPU1.

CPU n128 genuinely passes the warm >=50 criterion even under the stricter
filter and its bootstrap lower bound. Retain this positive evidence. CPU n256
passes ordinary >=50 but fails the joint ordinary/prose gate. Both GPU sources
fail ordinary/prose >=50, matched benefit and the 100 prose stretch. The
complete CPU source result is not the sought convenient transformed capacity
artifact; it does not by itself complete the architectural goal.

### Initialization and first request are charged separately

| Source/backend | SUM first request with load (s) | SUM load (s) | First ordinary IDs/s | First prose IDs/s |
| --- | ---: | ---: | ---: | ---: |
| n128 CPU0 | 24.0679109 | 0.2813564 | 47.4491 | 27.5055 |
| n128 GPU1 | 69.6094051 | 26.8412840 | 16.4058 | 9.5102 |
| n256 CPU0 | 35.2625717 | 0.5786961 | 25.3810 | 13.8958 |
| n256 GPU1 | 84.7982768 | 28.5682209 | 10.5545 | 5.7784 |

First-request rates use the actual retained first executions. They differ from
the modeled warm-plus-load amortization for 1/10/100 requests per process,
which is also retained in [RETENTION489](RETENTION_489_20261006.json). A warm
file cache or mapped payload is not a cold physical-DRAM measurement.

### Finer component profile is not admitted

All four source/backend profile1 decomposition gates fail because some book
ratios fall outside the prespecified [0.85,1.15] range; the original aggregate
range is [0.9,1.1]. Do not use profile1 kind fractions to explain this speed
failure, and do not mix METH458 fractions with these new rates. Whole profile0
rates and its encoder/cross-KV/decoding phase timestamps remain admitted.
[Phase algebra](meth489_primary_phase_algebra.json) derives conditional
ceilings using only those primary timestamps, with explicit assumptions.

## First data, interruptions and staging remain visible

METH488 first main hit its 2,700.125 s apparatus time bound after 174 complete
cases and 1,048 commands; all original outputs and partials are immutable.
Its first audit isolation fault and successful R1 admission are retained in
[METH488](METH_488_SHARED_INTEGER_PREFIX_RESULT_20261006.md).

METH489 adds exactly 107 previously unsuccessful/missing native blocks for the
last 18 source256 cases. It preserves the already completed CPU teacher of
case78 and all successful original timings, controls and outputs. No completed
block, compilation or earlier main/audit namespace was replayed.

The old helper's seed 486486 contradicted the frozen protocol's 485485. Those
old numerical summaries remain retained and unadmitted. METH489 changes only
the metadata bootstrap seed, then reconstructs all statistics from unchanged
first timings after completing the full cohort. This is not a new seed choice.

Foreign Python jobs caused bounded quiet waits and were untouched. Guards
check boundaries; foreign activity can start during a native interval. The
two-stage observations do not establish continuous machine exclusivity or a
causal breakdown of GPU overhead. The measured failure closes this fixed
layout in the declared conditions, not all GPUs or all exact arithmetic paths.

## Actual execution, resources and immutable receipts

| Operation | Actual tool/session/terminal | Exit | Wall time | Peak measured host bytes |
| --- | --- | ---: | ---: | ---: |
| Sole completion | bcaaef / 22140 / 529b42 | 0 | 675.875 s | parent 131,026,944; native 1,614,770,176 |
| Sole independent audit | 928197 / 44508 / c4992d | 0 | 55.281 s | 150,290,432 |
| Sole metadata finalizer | 93293b | 0 | metadata only | no inference |

Main PID29128/create1791257993.6883795; audit PID27716/create1791258895.2756495.
Both actual Windows event queries are available with zero matching events.
Main hashes 51,553,898,977 bytes; audit hashes 39,250,289,125 bytes. Combined
retained inventory: 5,769 files, 11,350,598,730 bytes, including old partials.
New outputs: 1,032,090,074 bytes; inherited: 10,318,508,656 bytes.

Explicit GPU operator allocation is 178,552,832 bytes: 166,232,064 shared
codes + 3,932,160 I/O + 8,388,608 workspace. Pinned host I/O is 3,932,160 bytes.
CUDA context/library allocation is not a measured WDDM process peak; retained
global memory snapshots do not supply that missing measurement.

- [Raw](meth489_shared_integer_backend_result.json), SHA256
  `db73d42f51867f6574ca024e1660186b7c97f7cdcb6a7c2bc973fcc3ffdcac1a`.
- [Independent retention](RETENTION_489_20261006.json), SHA256
  `72a85d3883e54b70ffe48fdfc82669310d30bd78301173358d365307232d361c`.
- [Complete admission](ADMISSION_489_20261006.json), SHA256
  `d329799e10503a7e2e4b8ea61d6b9ff62486132dcf86056e6483d26e4de30956`.
- [Main registration](meth489_main_sole_first_registration.json) and
  [audit registration](meth489_audit_sole_first_registration.json).
- Original physical executable, adjacent OpenMP DLL, payloads and runtime
  catalogs are identified by full path/SHA in [binding489](meth489_binding.json).

No engine edit, new resource, T4, installation or download. All attempted
namespaces remain immutable. See [whole algebra and next
decision](METH_489_WHOLE_ALGEBRA_AND_NEXT_20261006.md). Full goal stays active.
