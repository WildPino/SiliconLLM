# METH-132: Q15 reduces factor error in the same eight-byte child-B row

**Decision.** Retain Q15 as the next child-B candidate for a **new-source
full-model quality audit**. All frozen component gates pass. It keeps
METH-130 Q7's 276,578,344-byte bank and pair-LUT layout, but lowers
the fixed-state pooled expert residual relative L2 error from 0.046529
to **0.018943**. Its 95th-percentile per-vector error is **0.036900**;
all routes and gates are unchanged. The paired single-thread factor
median is **0.8738 versus 1.0511 ms/token-equivalent** for exact BF16.
METH-131's semantic failure for Q7 is not overturned by this component
result; Q15's full-model quality remains unknown.

The [protocol](METH_132_Q15_FACTOR_LUT_PROTOCOL_20260928.md) was
committed at `67326a3` before the export and CPU checks. The
[exporter](../../../benchmarks/native_expert_scaling/meth132_export_q15_factor_bank.py)
uses the unchanged quality-gated METH-126 bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`.
Each child-B output row uses one FP32 scale and four packed low/high
nibble pairs, now representing signed levels −7..7 rather than −3..3.
It verifies every packed record after writing. The `M132FB01` bank is
276,578,344 bytes, SHA-256
`9e03941ace0ecc441bd9dbd71108a0ac05d42642dea5c8fa506621002536eb18`.
The [export ledger](meth132_q15_factor_export.json) records 24
layer-wise raw-weight relative L2 errors of 0.0542–0.0603, down from
Q7's 0.1243–0.1394. Export took 11.703 seconds and ended at 734.4 MB
process RSS.

The [native checker](../../../benchmarks/native_expert_scaling/meth132_q15_factor_lut_cpu.c)
compares both banks on the METH-125 256-position E1280 vector file
SHA-256 `f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
It checks all router/shared-A bytes, every Q15 scale and nibble, and
reroutes both banks before computing each residual with four 256-entry
pair tables per selected child. The [native log](meth132_q15_native_raw.log)
retains per-layer error/coverage and five alternating-order paired
timing repetitions. A nibble value 15, reserved by the Q15 format,
made the reader exit 1 with `bad Q15 nibble`. The bank was restored to
its recorded SHA; the [negative log](meth132_q15_bad_nibble.log) saves
the check.

| Frozen component measure | Q15 result | Gate |
|---|---:|---:|
| Parent/child route and gate mismatches | 0 / 24,576 selections | 0 |
| Nonfinite / zero-norm reference residuals | 0 / 0 | 0 nonfinite |
| Pooled residual relative L2 | 0.018943 | <=0.025 |
| 95th-percentile per-vector relative L2 | 0.036900 | <=0.05 |
| Maximum per-vector relative L2 | 0.077698 | reported |
| Distinct selected children per layer | 252–402 | reported |
| Bank size | 276,578,344 B | <=300,000,000 B |
| Exact BF16 factor median | 1.051055 ms/token | baseline |
| Q15 LUT factor median | 0.873755 ms/token | <=1.261266 ms/token |
| Q15 / BF16 factor time | 0.8313 | <=1.20 |

The checker took 4.014 seconds and reported 811,016,192 bytes RSS.
The finite source positions were already viewed in METH-125 and 130;
this is a component measurement, not independent quality or cold DRAM
traffic evidence. Next, bind this exact Q15 bank into the BF16 donor
plus centered E1280 model and evaluate **new** documents, greedy
responses, PIQA regression and blinded excerpt grounding. Only a
full-model pass would justify C integration with a compact core and
same-artifact >=50 accepted-token/s testing. Neither this result nor
Q7's failed quality screen demonstrates useful E12800 or 10B/100B
learned capacity.
