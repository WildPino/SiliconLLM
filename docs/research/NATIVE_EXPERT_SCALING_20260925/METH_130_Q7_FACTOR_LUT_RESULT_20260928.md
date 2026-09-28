# METH-130: Q7 child-B LUT passes the fixed-state E1280 component screen

**Decision.** Retain the Q7 child-B format for a full-model quality
experiment. It reduces the quality-gated E1280 bank from 496,779,304 to
**276,578,344 bytes (44.3% smaller)**. On 256 actual varied positions
across 24 layers, its 6,144 expert residual vectors have pooled relative
L2 error **0.046529**, 95th-percentile per-vector error **0.086113**,
and no changed parent/child route or gate. Its paired single-thread
factor median is **0.8825 versus 1.0388 ms/token-equivalent** for BF16.
All frozen component gates pass. This is a component screen on viewed
states, not a model-quality or >=50 accepted-token/s result.

The [protocol](METH_130_Q7_FACTOR_LUT_PROTOCOL_20260928.md) was
committed at `b04b9f3` before the export and CPU checks. The
[exporter](../../../benchmarks/native_expert_scaling/meth130_export_q7_factor_bank.py)
binds the exact METH-126 shared-A bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`.
It keeps every router and shared-A byte intact, quantizes only each
eight-coefficient child-B output row to seven signed levels, packs four
nibble pairs and one FP32 scale into eight bytes, then rereads every
record. The new `M130FB01` bank SHA-256 is
`20329a07f7dfcd3bfee08d4ee64b7e243d0004f4e18ec3543c78516c203b0265`.
The [export ledger](meth130_q7_factor_export.json) contains all 24
layer-wise raw-weight errors (relative L2 0.1243–0.1394) and readback
status. Export took 11.328 seconds and ended at 734.5 MB process RSS.

The [native checker](../../../benchmarks/native_expert_scaling/meth130_q7_factor_lut_cpu.c)
loads both banks plus the METH-125 E1280 vector file SHA-256
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
It verifies all dimensions and lengths, all router/A bytes, every Q7
scale and nibble, then reroutes both banks on the same BF16 inputs. It
uses four 256-entry activation-pair lookup tables per selected child,
preserving the reference's BF16 part/gate/output rounding. The
[raw log](meth130_q7_native_raw.log) gives every layer and five
alternating-order paired timing repetitions. A one-byte invalid nibble
made the reader exit 1 with `bad Q7 nibble`; the source bank was
restored to its recorded SHA afterward. The
[negative-control log](meth130_q7_bad_nibble.log) retains the failure.

| Frozen component gate | Result | Limit |
|---|---:|---:|
| Parent/child route and gate mismatches | 0 / 24,576 selections | 0 |
| Nonfinite / zero-norm reference residual vectors | 0 / 0 | 0 nonfinite |
| Pooled residual relative L2 | 0.046529 | <=0.10 |
| 95th-percentile per-vector relative L2 | 0.086113 | <=0.20 |
| Maximum per-vector relative L2 | 0.169378 | reported |
| Distinct selected children per layer | 252–402 | reported |
| Q7 bank size | 276,578,344 B | <=300,000,000 B |
| Exact BF16 factor median | 1.038789 ms/token | baseline |
| Q7 LUT factor median | 0.882483 ms/token | <=1.246546 ms/token |
| Q7 / BF16 factor time | 0.8495 | <=1.20 |

The checker reported 811,016,192 bytes process RSS and finished in
3.992 seconds. Its five paired measurements use the same 256 source
positions and alternate which bank runs first. This reduces order bias
but is still a finite hot-replay CPU component result; it does not
measure physical DRAM traffic at a larger trained child count.

The next binding step is to test the **same Q7 bank** as part of the
BF16 donor plus centered E1280 composition on new source-disjoint
documents, greedy continuations, tasks and grounded semantics. Only
after that quality gate can the Q7 path be integrated with a compact
native body and measured end-to-end. E12800 or 10B/100B capacity and
route quality cannot be inferred by copying these 1,280 learned slots.
