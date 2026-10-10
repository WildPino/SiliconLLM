# Grouped saved-gradient steps: actual original C descent and full audit

10 October2026. **GROUPED_DISCRETE_DESCENT_QUALIFIED** after complete independent
stored audit. Absolute sample quality FAIL; full goal ACTIVE/INCOMPLETE.
[Frozen protocol](ORIGINAL_CATEGORICAL_TRUST_STEP_PROTOCOL_20261010.md),
[baseline/backward](ORIGINAL_CATEGORICAL_CAUSAL_PREFLIGHT_RESULT_20261010.md).

## What changed and what the result proves

Three genuinely changed92-master models were built from the SAME qualified
saved90-gradient set. Per-expert/projection, embedding-row, router-row/bias and
core-tensor relative displacements use alpha=.0001,.001,.01, fixed before new
observations. Each candidate has one mathematical parameter displacement from
actual27; the candidates are independent, not three sequential training steps.
All3 complete masters, original-format packed files and statistics are retained.
No Adam restoration/execution or new neural forward/backward/source query.

Each candidate was evaluated in the original qualified C executable on the SAME
58-ID/18-label FIT history. Fixed canonical norm/head retained; all110 fields
verified against new masters, full trit/scales drift and integer witnesses
audited. Reuse old C scores as baseline; no baseline history replay.

Selected alpha.001 reduces actual native first-balanced KL138.9608564 ->
58.8205998 (**57.6711%**) and first-label KL129.9724928 ->44.1511238
(**66.0304%**). This qualifies local discrete descent under frozen gates, not
useful chatbot conversion. All18 source argmax labels remain wrong.

## Frozen grid and observed geometry

| Alpha | Weighted native KL | Mean label KL | First-label KL | Wrong labels |
|---|---:|---:|---:|---:|
| Existing baseline | 138.9608564 | 146.9505129 | 129.9724928 | 18/18 |
| .0001 | 126.9881072 | 137.3586907 | 115.3212008 | 18/18 |
| **.001 selected** | **58.8205998** | **71.8601340** | **44.1511238** | **18/18** |
| .01 | 123.6732108 | 96.0091172 | 154.7953161 | 18/18 |

| Alpha | Dot(actual F32 displacement, saved STE gradient) | Max relative displacement | Moved coefficients |
|---|---:|---:|---:|
| .0001 | -21.7424946 | .000100059191 | 53,908,460 |
| .001 | -217.4249526 | .001000057934 | 54,342,201 |
| .01 | -2174.2495546 | .010000058105 | 54,462,458 |

All radii pass alpha+1e-6 and global directional predictions are negative.
The large step nevertheless worsens relative to the middle step and regresses
the first token relative to baseline. Local derivative sign alone is therefore
insufficient to choose a finite step. This is not proof of a loss minimum, a
universally safe alpha, or the cause of older clipped-Adam failures.
Moved coefficients count nonzero numerical F32 differences; raw state hashes
also preserve exact bit custody.

## Discrete effects and complete field checks

Each candidate covers100156 groups with6 saved F64 statistics/group, all90
trainable gradient sources and all92 model states/721008128 F32 entries.
Head/norm remain the canonical frozen coefficients. Original header/table bytes
and110-field ABI identical in layout; full new field values verified.

| Alpha | Changed trits /679477248 | Changed row scales | Certified unchanged trits | Changed among certified | Changed route sets /348 |
|---|---:|---:|---:|---:|---:|
| .0001 | 2,717 | 436,652 | 679,471,577 | 0 | 52 |
| .001 | 26,613 | 447,399 | 679,421,762 | 0 | 136 |
| .01 | 267,337 | 449,299 | 678,924,271 | 0 | 315 |

Ranked route-slot changes361/1006/2225; maximum mass deltas by ranked slot
.04989886/.07497016/.13590059. This comparison does not align mass by expert
identity when IDs change. Mass sum defects1.49012e-7/1.63913e-7/1.34110e-7,
all below1e-6. Valid distinct IDs and finite nonnegative masses checked on all
174x6 routing rows. All18432 integer witness outputs/candidate match new packed
weights, with complete symbol/scale verification in addition to these witnesses.

The stable-cell certificate uses actual old/new F32 exporter scales and an
unexceptional division-rounding allowance; F64 evaluated, no interval claim.
Uncertified cells may stay unchanged. Scale and routing changes are meaningful
even when very few symbols flip; small parameter movement is not a guarantee
of unchanged execution paths or generalized chatbot behavior.

## Independent summation gap and repair before audit

The original planned audit cast reductions to longdouble. Read-only platform
inspection found Windows NumPy2.4.6 longdouble8 bytes/mantissa52, equal in
precision and dtype comparison to F64 (dtype char differs). A cast alone would
not provide the intended independent summation. The producer/frozen source and
criteria remain immutable. No original stored audit was launched.

[New fsum supplement](ORIGINAL_CATEGORICAL_TRUST_FSUM_AUDIT_20261010.md) and
adapter execute the full frozen stored audit with `math.fsum` for2D F64 axis1
reductions, including parameter/gradient/actual-displacement norms and dots.
Other reductions remain unchanged; full-V probabilities use independent
shifted sequential logaddexp. A cancellation fixture distinguishes NumPy sum0
from fsum1 with exact expected1. No higher-precision datatype claim.

All criteria unchanged. Every updated coefficient matches the independent
proposal **0 ULP** (limit1); max group relative statistic delta4.36156e-16
(limit1e-9); max54-label independent native KL delta2.13163e-12 (limit1e-8).
All input/output hashes, source90 parameters/gradients, new3x92 masters,
3x110 export fields, all groups/certificates/witnesses, actual native commands,
query bytes, counters, resources and frozen selection PASS. Complete audit
despite18/18 quality failure. No repeated model/backward/native/optimizer call.

## Provenance and cost

Producer code/holder freeze `738896a6bace46c0707124aaac9cd52f5408b6aa`.
[Numeric binding](original_categorical_trust_step_binding_20261010.json)
SHA256 `c6500378ef3e37d6cca2a72fb05b1fe52dfb8c483a9bee228af1f4f53aa5cd42`:
2917 inputs/15,936,162,843 bytes, actual imported source/bytecode/DLLs behind
junctions, Python/Torch/NumPy/psutil and qualified parent receipts/artifacts.
Producer holder `a8e57aa014f391ce91be8b139528df2d762d597a2177b118b1e601436d7a7480`.
[Producer result](original_categorical_trust_step_result_20261010.json)
SHA256 `fbde76bcadb5f2da0b4d9e9c47ebef1a6e4ec1c1d9d0e2e72be8ec2c5301dae7`.

Audit adapter/holder freeze `cf22e7ed9a8493d336eda59f3dc5252a277827a0`;
audit holder `721cfb2db35b895c805eb37dfcd113b7e1228796a4d7203e8246721988ab55c5`.
[Independent adjudication](original_categorical_trust_fsum_adjudication_20261010.json)
SHA256 `e6278a018928d87d1e15de3a0c7ab9b97507da69fd7e8fa2ccd37131890bd724`.
Commands, PID/creation, logs and full recursive303 output extents are in the
same-stem live/terminal receipts. Output10,273,007,351 bytes plus211317-byte
producer metadata, below12GiB. Three complete masters each about2.884GB plus
three520029440-byte packs, grouped arrays, scores/routes/witnesses/receipts.

Producer session16983/holder25220 created18:22:14.219557+02:00;
worker3720 creation**1791649350.6792483**; native18552/10060/28292 created
1791649487.7876115/1791649621.03956/1791649757.081238. All CLOSED/exit0.
Worker404.125s/held449.766s; worker OS11781296128 + native527511552 +
holder31649792 =**12340457472 bytes**, below24GiB. A manually typed live INDEX
creation-time typo was corrected from the authoritative terminal; original
process/creation/command receipts were always accurate and used for control.

Audit session10761/holder31652 created18:30:09.624404+02:00;
worker29292 creation1791649833.7996647. All CLOSED/exit0/errornull/fullhashes exact.
Worker1148.984s/held1213.532s; worker11735347200 + holder31461376 =
**11766808576 bytes**, below20GiB. **Held total1663.298s (27.72 minutes)**.
Structural and runtime read-only preparation probes are additional preparation
costs, not counted as held inference/optimizer calls. No speed admission;
foreign publisher daemons preserved. CUDA hidden throughout, no GPU/T4.

Counters: three mathematical proposals/parameter displacements, three NEW C
histories/174 positions/54 paired labels; zero neural forwards/backwards,
optimizer API updates/source/SVD/DEV/RESERVED/GPU/T4. Stored audit re-evaluates
proposal arithmetic and complete raw data, not original histories or optimization.

## Decision and next point

Actual local discrete descent QUALIFIED, absolute quality FAIL. Selected
`alpha_0.001/masters.pt` and `candidate.packed` are actual retained modified
states suitable for the NEXT numerical prerequisite, not a useful chatbot.
[Changed-state causal GPU/backward and route-mass comparison](ORIGINAL_CATEGORICAL_CHANGED_HISTORY_NEXT_20261010.md)
is PROPOSED/UNIMPLEMENTED: one new changed-point GPU history/backward, compare
existing C scores/routes, then finite multi-case first-balanced transfer.
[Scale geometry and primary-paper checks](ORIGINAL_CATEGORICAL_SCALE_GEOMETRY_20261010.md)
give conditional n/gauge reasoning and an optional untested neutral-direction
correction. They do not establish new useful expert capacity or universal STE
convergence. Donor-relative own-history chat/held-out quality and useful>=50 on
the SAME artifact, useful n/structured CPU IDs+mass/DRAM and families remain open.
