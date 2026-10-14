# Actual 24-layer recurrent cost: the current backbone exhausts the budget

14 October 2026. **COMPLETE / stored numeric and accounting audit PASS /
RECURRENT_MEDIAN_EXHAUSTS_20MS. Goal ACTIVE / INCOMPLETE.**

This is the first sequential cost screen of all 24 distinct qualified native
branches on real own-history inputs. Fresh contexts reconstruct every prefix;
no state snapshot accelerates initialization. All 24 calls execute sequentially
per timed position, batch 1, single-threaded C, affinity [0].

| Parameter cache intervention | Samples | Minimum ms | p10 ms | Median ms | p90 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Resident, no deliberate eviction | 64 | 24.60770 | 25.56244 | 26.97095 | 30.01076 | 70.26580 |
| Parameter-evicted before whole24 | 64 | 24.47650 | 24.97985 | 26.14040 | 27.98012 | 28.84800 |

The 20 ms budget is exceeded by 6.97095 ms at the resident median, before
contextual generators, FFN, attention, head, tokenizer or transport. Even
the minimum observed whole24 exceeds 24 ms in both modes. These are recurrent
latencies, not useful accepted model tokens/s or a universal impossibility
bound. This implementation needs a real change before additional conditional
banks or larger donors can fit into the final same-artifact goal.

Arms run in fixed order on a shared desktop, without heavy concurrent research
workers. The smaller evicted median does not establish a causal speedup.
Parameter CLFLUSH/MFENCE is outside the timed span and separately recorded:
median 7.1253 ms, range 6.5832..8.8440 ms. State/code/rows are not deliberately
evicted. Actual DRAM remains UNRESOLVED. The 24-call same-ABI no-op wrapper
median is .0387 ms, reported without subtraction. No outliers were removed.

## Frozen inputs and numeric scope

Shortest/longest existing history per split, last 16 positions each:

| Split / length | Record | History | Positions, zero-based |
| --- | --- | ---: | --- |
| FIT short | broad_fit_explore_instruct_rewriting_008 | 58 | 42..57 |
| FIT long | broad_fit_smol_magpie_ultra_022 | 1507 | 1491..1506 |
| DEV short | broad_dev_explore_instruct_rewriting_035 | 61 | 45..60 |
| DEV long | broad_dev_smol_magpie_ultra_036 | 1488 | 1472..1487 |

These are consumed histories from the qualified full-width quality bridge,
not new held-out quality evidence. Layer zero uses real uint32 IDs/F0; later
layers use actual archived preconv rows, excluding their generation cost.
Producer checks all prefix/continuation BF outputs and final state/ring/seen/
last against that bridge. Stored auditor replays every prefix/continuation
and saved intermediate. Independent integer matmul, manual RNE quantization,
ordered F64 scales and BF rounding verify all 6,291,456 saved readout words.
All 192 site records / 3,072 calls / 299,827,200 prefix BF words pass, as do
sample coverage, summary arithmetic and byte metadata. Recurrence DLL is shared;
this is not a new independent recurrence equation proof or authentication of
wall-clock values. The source bridge's historical custody limitations remain.

## Physical footprint, costs and custody

All 24 parameter packets load simultaneously with distinct addresses:
310,855,680 bytes of array payload. Readout q is 12,582,912 bytes/site plus
16,384 scale bytes/site. State payload is 28,311,552 bytes, rings 516,096,
total native context structs 28,828,032 bytes. Largest current row buffers
478,826,496 bytes, expected outputs 146,276,352 and timed intermediates
29,039,616 bytes are replay footprint, not final model parameters.

The algorithmic operand ledger distinguishes payload from repeated reads:
the readout reuses a 6,144-byte I16 input across 2,048 rows, giving 12,582,912
logical input reads/site in that loop. State read plus write is 2,359,296
bytes/site. These counts do not measure cache-line reuse or actual DRAM.

Original preparation failed before binding/compilation on a nonexistent root
engine.c pin (150.703 s); receipt preserved. Corrected freeze ba53952 removes
that extra pin and retains canonical benchmarks/phase60/engine.c plus all
other protected files. No timing, threshold or data change. V2 preparation
13.797 s / OS 52,207,616 bytes; producer 227.344 s / OS 1,082,679,296;
stored audit 183.375 s / OS 229,744,640; closure 3.609 s / OS 28,860,416.
All corrected phases exit 0, with successful PID/create identities absent.
No GPU/source forward/optimizer/RESERVED/delegation. E family 194 files /
463,653,248 bytes; declared pins and outputs sealed.

- [Protocol](NATIVE_24_COST_PROTOCOL_20261014.md), [state and commands](NATIVE_24_COST_STATE_20261014.md).
- [Binding V2](native_24_cost_binding_v2_20261014.json), SHA256 f2ac53bb82f1adf760316d7a56b2681c1e3cff4fb5868ddce57efa97a331c73e.
- [Producer](native_24_cost_result_20261014.json), SHA256 5f5b97119f8b3152a9e79aab27b8b3a758888bdbca6fa673ab2da48a7ce3cb0d.
- [Audit](native_24_cost_adjudication_20261014.json), SHA256 46526b8f5634252930a3da9ac36caaf85ee5b6032ca57575549baf8aa132d983.
- [Closure](native_24_cost_session_closure_20261014.json), SHA256 7fdb543f8934ecdd8fff8d6e13cd7e776c086784adb9ad78703f8bb05b812e95.

## Decision

Do not expand failed FFN grids before addressing measured backbone cost.
The [next exact accumulation variable](NATIVE_READOUT_ACCUMULATION_NEXT_20261014.md)
reduces unnecessary horizontal reductions while preserving int64 dots and
scale/BF semantics. It is a proposal; exactness, paired physical benefit and
complete trace qualification are required. If insufficient, reduce active
donor-derived core/bytes with own-model quality. Useful conditional n, actual
DRAM, multiple-family/scale transfer and final same-artifact CPU quality and
>=50 useful tokens/s all remain incomplete.
