# Exact wide-lane readout: correct, below the acceleration admission gate

14 October 2026. **COMPLETE / full stored numeric audit PASS /
EXACT_CANDIDATE_ACCELERATION_REJECTED. Goal ACTIVE / INCOMPLETE.**

Changed only integer accumulation in a separate native translation unit:
1024-column SIMD lane sums widen to int64 before horizontal reduction,
replacing original128-column horizontal int32 reductions. Two inner
accumulators, three wide chunks, one final int64 reduction. Admitted-domain
lane bound1,526,680,064<2^31; whole dot36,640,321,536<2^53. Source parameters,
input quantization, ordered F64 scales, BF rounding and recurrent code remain
unchanged. Original qualified source and DLL are preserved.

| Whole24 paired mode | Original median ms | Wide median ms | Wide/original | Reduction | Frozen gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Resident, no deliberate eviction | 26.15890 | 25.78135 | .9855670536605132 | 1.4433% | >=5%, FAIL |
| Parameter-evicted per arm | 26.18470 | 25.69570 | .9813249722165998 | 1.8675% | >=5%, FAIL |

Each mode has64 paired positions, last16 of the same shortest/longest FIT/DEV
histories as the qualified cost screen. Arm order alternates by position.
All four stratum ratios remain<=1.05, but both overall ratios fail<=.95;
no threshold change, outlier removal or promotion. Resident wide minimum
24.4309ms, evicted24.4332ms; median remaining20ms budgets are NEGATIVE
5.78135ms/5.69570ms before generators/FFN/attention/head/transport.

Readout-only direct calls after both whole24 arms have median original/wide
12.50045/11.96355ms in resident iterations and12.64025/12.23750ms in evicted
iterations. These direct calls receive no additional eviction, and exclude
conv/state/gate/norm/quantization. Do not label their cache condition as a
separately evicted readout or subtract their medians as an exact component
decomposition. The observed small kernel gain does not create whole24 budget.
No-op24-call median.0394ms is reported without subtraction. Shared desktop,
affinity0/single-threaded C, stable shared parameter owners across arms;
310,855,680-byte parameter payload unchanged. Actual DRAM unresolved.

## Exactness and scope

64 bounded adversarial/random3072-term dots equal independent Python integer
sums, including full positive/negative saturation, zero and cancellation.
BF half-way ties/zero/sign controls pass manual bit rounding. Candidate and
original continuations match every sealed intermediate and final state/ring/
seen/last at all24 sites:192 reference packets /3072 candidate calls /6,291,456
independently verified int64/scaled/BF readout words. Timing coverage/order/
summary arithmetic and all declared before/after hashes pass.

Producer AND auditor replay ALL97 qualified bridge trajectories using fresh
contexts and actual rows:2328 sequence-sites /580056 positions /
1,187,954,688 BF output words /686,555,136 final state words, all exact.
Shared candidate recurrence DLL/ABI/buffer/replay helpers are explicit;
integer/scales/BF checks and timing summaries are independently implemented.
This qualifies stored native compatibility with the known14/16 quality bridge,
not a fresh GPU model run, new held-out quality, end-to-end CPU throughput,
conditional capacity n or general-family transfer. Historical GPU custody
and transitive producer pin limits remain declared in the parent result.

## Custody and costs

Numerical freeze9d88176. Original preparation8452/create1791967567.8373785,
exit0/884.031s/OS52,748,288 bytes, preserved. Its I/O cost motivated an explicit
pre-observation wall amendment1800=>2700s via separate wrapper7a185d9/bindingV2;
no recompilation or numeric/gate change. V2 preparation24628,0.313s/OS39,309,312.
Producer23252/create1791968732.6442983,exit0/1014.281s/OS1,248,886,784;
auditor29360/create1791969991.8596592,exit0/934.015s/OS1,142,366,208;
closure19284/create1791971333.2079585,exit0/17.672s/OS40,128,512.
Actual corrected phases finished within even the original1800s duration.
All declared input/output pins and successful worker identities sealed.
New E family5 files/942,704 bytes; existing463MB packets and full bridge
archives reused. No GPU/source forward/optimizer/RESERVED/delegation.

- [Numerical protocol](NATIVE_WIDE_READOUT_PROTOCOL_20261014.md), [resource amendment](NATIVE_WIDE_READOUT_RESOURCE_V2_20261014.md), [state](NATIVE_WIDE_READOUT_STATE_20261014.md).
- [Binding V2](native_wide_readout_binding_v2_20261014.json), SHA256 a1c433f85df05294fdde3577075e3575926e1bc03013916c4fc32cf8f4432d22.
- [Producer](native_wide_readout_result_v2_20261014.json), SHA256 113a8c504f5bcb4e2322963d1e798e62818ae649bc74a89096d362dc66716c39.
- [Full audit](native_wide_readout_adjudication_v2_20261014.json), SHA256 d72e21992e324bfa0caf6fb28c3cc2e292f57a4c8283eb0a3bea44a46c838087.
- [Closure](native_wide_readout_session_closure_v2_20261014.json), SHA256 0ff1689722f14024cb793916e7e326781e1caa9dd4ab0e09fe497e05f604532b.

## Decision

Do not replace the qualified backend or continue a minor accumulation layout
grid. Next [finite-BF activation lookup](NATIVE_BF_ACTIVATION_LUT_NEXT_20261014.md)
targets a different source operator family: thousands of double transcendental
calls after explicit BF input rounding. Prove all finite table entries, FP
environment controls and complete trace compatibility before physical timing.
This is a proposal, not another admitted acceleration or knowledge bank.
If insufficient, reduce active donor-derived core/bytes/layers with own-model
quality. The compact conditional architecture and SAME final CPU artifact
quality/useful>=50, actual DRAM, useful n, multiple families/scales and
conversion costs remain the original unfinished goal.
