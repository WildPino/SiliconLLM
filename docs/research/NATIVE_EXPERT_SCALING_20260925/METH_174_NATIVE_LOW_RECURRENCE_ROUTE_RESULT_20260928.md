# METH-174: 1,685-entry native CPU route passes parity and cost

**Decision:** the fixed threshold-8 route passes the conditional native CPU component gate after METH-173's prospective route pass. Python and C agree on the four checked context classes and the candidate table's 1,685 exact entries; the source child and four choices remain unchanged. This prices warm parent/child/grandchild routing on actual E1,280 hidden states, not E12,800 factor arithmetic or end-to-end `engine.c` generation.

The [fixture report](meth174_native_low_recurrence_fixtures.json), SHA-256 `5469d26342c8b14b9fef02696f1e853e4500908d8d2c73187f54d7f7688d0a32`, binds the 138-entry 1,672-byte baseline table, 1,685-entry 20,236-byte candidate table, 256×24 actual BF16 states and context files for baseline hits, candidate-only recurrent hits, both tokenizer delimiters and misses. The candidate-only golden `(11,198,56)` maps child 89/layer 3 from baseline content 892 to shared 890; the delimiter maps to 890 in both, and `(123,45,67)` remains content 899. The runner built the [C implementation](../../../benchmarks/native_expert_scaling/meth174_native_low_recurrence_cost.c) with Clang `-O3 -mavx2 -mfma -std=c11`.

| Context cell | 138-entry baseline median ms/token | 1,685-entry candidate median ms/token | Ratio |
|---|---:|---:|---:|
| Baseline table hit | 2.498 | 2.470 | 0.989× |
| Candidate-only recurrent hit | 2.435 | 2.427 | 0.997× |
| Chat delimiter | 2.472 | 2.457 | 0.994× |
| Content miss | 2.433 | 2.438 | 1.002× |

Five paired order-alternating repetitions per cell pass <=3.5 ms/token and <=1.25× baseline gates. All 20 timed old/candidate page-fault deltas are zero after warmup. The [machine result](meth174_native_low_recurrence_cost_result.json), SHA-256 `55db1199b54a84463efc15f62c38d6e3896d08312bb31d872ef53d94f6c33e3c`, includes every repetition, hashes, checksums and memory usage. Process RSS was about 534.7 MB and elapsed time 30.439 s. Ratios below one are timing noise; the evidence is that this larger lookup adds no detectable material cost in this component fixture.

METH-173 and METH-174 together allow a matched long-budget E1,280/E12,800 *learning test*. They do not establish useful additional experts, donor-relative quality, compact LUT quality/cost, full-model parity, >=50 accepted tokens/s, or transfer to 10B/100B models.
