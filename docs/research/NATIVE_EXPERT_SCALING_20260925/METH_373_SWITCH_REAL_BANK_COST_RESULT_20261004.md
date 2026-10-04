# METH-373: bounded fourfold actual learned-bank CPU cost

**PASS: all seven prospective gates.** Freeze `bbd0599`, authoritative
controller exit0, raw SHA256:
`0ac1ab8a7547e02ca94a28f204b7085a10ae4f525aed59c28297b9eca1cc88cf`.
Main time1,060.891s, maximum checked combined RSS1,420,296,192B;
end available RAM68,067,745,792B. All first observations retained.

## Fixed intervention and oracle

Same356 native executable / CPU1 / every child affinity[0], actual physically
compact370 n64/128 and338 full256, original14.664B donor/common core/head/
I8 weights/A16 inputs/F32 router/top1/capacity. Each of ALL96 consumed362 cases
has source29 and forced decoder14 at EACH n. Three cyclic orders balanced32
cases each; one warmup/three measured repetitions per child/profile0, followed
by all cases with one warmup/one measured/profile1. Every warmup, measured and
profile output SHA equals complete prior363 teacher (256) or372 teacher
(64/128). Fresh payload/spec/metadata/compiler/runtime/binary identities and
independent manifest readbacks exact. Fresh physical CPU topology exact371.
All288 primary and288 profile cases complete, all258 routes accepted per case,
all dynamic descriptors and retained function IDs exact. No competing model
or native timing worker. Source payloads/runtime/engine unchanged.

## Primary phase cost

Sum of per-case medians over all96 cases, not a single long contiguous decode.
Complete time includes encoder, crossKV and cached forced decoding; load,
startup, tokenization, serialization and teardown excluded. Warm/hash-primed,
pretokenized inputs. This is fixed-length CPU cost, NOT accepted generation.

| Actual n | Encoder s | CrossKV s | Decode s | Full s | Full ms/forced position | Full repeat ratio | Decode repeat ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 9.849998 | 1.450898 | 9.893470 | 21.238405 | 15.802385 | 1.009222 | 1.005312 |
| 128 | 10.104527 | 1.471065 | 9.993249 | 21.578709 | 16.055587 | 1.009338 | 1.007667 |
| 256 | 10.396428 | 1.440805 | 9.943513 | 21.855844 | 16.261788 | 1.003975 | 1.013628 |

Phase medians are computed separately and need not sum exactly to full medians.
Repeat ratios compare ALL96 summed timings by repetition0/1/2; these do not
claim every individual case repeats within10%.

| n256/n64 phase | Ratio of sums | One-sided lower95 | One-sided upper95 |
| --- | ---: | ---: | ---: |
| Encoder | 1.055475 | 1.049528 | 1.061447 |
| CrossKV | 0.993044 | 0.982304 | 1.003837 |
| Decode | 1.005058 | 0.997535 | 1.012656 |
| Full | 1.029072 | 1.022030 | 1.036202 |

Both primary full AND decode upper95 <=1.20 PASS. Paired book-bootstrap,
24 books/four cases each,10,000draws/seed373373. Primary full cost grows2.91%
(upper3.62%); decode0.51% (upper1.27%) for fourfold actual bank cardinality.
Additional descriptive128/64 phase ratios are retained in raw, with same rubric.

## Routing, lookup, consultation and memory

Profile1 matrix times are secondary and include per-matrix clock overhead.
The router matrix total (encoder+decoder) is0.206176s at64,0.404311s at128,
0.790247s at256, approximately3.83x from64 to256. Relative to each profile's
OWN full summed wall time, these are0.973%,1.865%,3.625%. Encoder router alone
0.112606/0.224226/0.446536s; decoder0.093570/0.180086/0.343711s. The increase
is consistent with the measured linear F32 classification workload; timing
does not establish a cache or DRAM causal mechanism. Core/expert/head profile
matrix times and each case's phase wall times remain in raw.

| Actual n | Whole mapped payload B | Decode logical matrix B/position | Mean unique consulted expert codes+scales B/case | Sampled maximum child RSS B |
| --- | ---: | ---: | ---: | ---: |
| 64 | 3,903,912,448 | 125,478,400 | 792,295,904 | 1,114,243,072 |
| 128 | 7,541,946,880 | 126,658,048 | 956,702,112 | 1,268,727,808 |
| 256 | 14,818,015,744 | 129,017,344 | 1,048,668,992 | 1,354,825,728 |

Decode matrix I8 codes123,764,736B and F32 row scales534,016B per position
remain constant. F32 router18,432*n B, namely1,179,648/2,359,296/4,718,592B.
Each encoder position and each decoder position selects six actual WI/WO
pairs. No expert cloning or synthetic candidate padding.

Accepted expert unions across ALL96 cases: encoder banks60-64/109-128/
173-252 at64/128/256; decoder61-63/116-127/209-235. Per-case unions and exact
source expert IDs are retained. Across the cohort, the unique consulted
expert code+scale regions sum3,550,464,000 /6,954,175,488 /12,412,422,144B.
These are unique addressed regions, NOT bytes fetched from DRAM. Sampled RSS
includes process memory and mapped pages; it does not measure exact expert
residency or show that the whole file fits in cache. No hardware DRAM counters
were collected. This runtime uses direct expert lookup and flat F32 routing;
no new hierarchical or ternary LUT algorithm is qualified by this experiment.

## Cost, decision and remaining work

2,880 output/log files occupy6,312,272,256B. The prospective approximately1GiB
output estimate was too low; it was an estimate, not an acceptance/resource
gate. Main17.68min exceeds expected<=15min but stays inside the frozen30min
stop; peak1.42GB stays below16GiB. No optional rerun or threshold change.

The full CPU cost does not grow proportionally with actual bank size for this
64-to256 contrast. Flat router cost does grow and must remain charged when
extending n. This bounded evidence does NOT extrapolate to thousands of experts,
~100B, other families, cold service, long contexts or physical DRAM throughput.
It also does not establish useful monotonic capacity scaling:372 mixed quality
is retained, including improved128 masked NLL and poorer original agreement.
Changed64/128 artifacts cannot inherit363 whole donor-quality acceptance.

Prior364 SAME useful256 accepted full rate45.1114/lower42.5362 FAIL remains;
forced373 timing does not replace it. Prior366 repeat failure remains closed.
Next decision should target a NEW execution variable on the useful256 artifact
with complete byte equivalence, then separately frozen accepted full rate.
Useful larger-n quality/routing/real DRAM and multiple-family/scale proof remain
required. Final goal active.

Reproduction: [frozen protocol](METH_373_SWITCH_REAL_BANK_COST_PROTOCOL_20261004.md),
[controller](../../../benchmarks/native_expert_scaling/meth373_switch_real_bank_cost.py),
[raw result](meth373_switch_real_bank_cost_result.json).
