# METH-301: four-coefficient source-shaped LUT cost and native preflight

## New variable and prospective decision

300's U8/two-coefficient format fails the complete560MB addressed-weight
allotment even with impossible free head/palettes. Change the encoded vector
length to FOUR original coefficients/U8 into a shared256-entry FP32 4D
non-Cartesian palette, preserving every source row/channel. Freeze this
protocol, controller, C backend and opt-in phase60 entry before observation.
This is an enabling cost experiment, NOT learned donor transfer or quality.

Reuse all414 real GigaChat base source/Q4 descriptors from hash-bound300.
No source weights/captured quality inputs are read or palettes optimized.
Old34/130/132 factor-LUT and197-200 tiny synthetic banks are different:
here all full-source projection shapes, hundreds of millions of codes/token,
MLA per-head inputs, per-expert down queries, full head and flat routing are
priced/executed. Their passes/inconclusive rates do not transfer to301.

## Complete descriptor calculation and explicit native omissions

Price all284 BF16 source MLA/routed/shared/dense/head projections:
one U8 per consecutive four coefficients, FP32 scale per original row,
one shared4096-byte palette per tensor. Unchanged actual Q4 embedding row,
F32 router, norms and biases retain300's count. n64 and analytical n640
must EACH fit560,000,000 addressed bytes/token BEFORE native eligibility.
Charge table construction entries/7 multiply-add operations, FP32 writes,
lookup/adds, FP32 table gathers and RAM payload separately.

An actual phase60 compile selector `SILICON_VECTOR4_LUT_PREFLIGHT` enters
the new C implementation only. It consumes a write-once binary catalogue
of309 matrices:284 encoded projections plus25 F32 routers, source names/
dimensions ordered from300 and SHA-bound by the controller. Other phase60
profiles and old sources are preserved. Keep six CPU threads, no fast-math,
FP32 fixed-order table generation/row sums and `-ffp-contract=off`.

C fixture weight banks are fully allocated and populated in physical RAM.
Each stored expert/head has its own deterministic pseudorandom code bytes
and row scales; palettes have four independent real coordinates per entry.
Uniform indices exercise actual gathers but are NOT encoded donor weights.
They demonstrate cost/correctness only, not real or useful new capacity.
Source-layer shapes are exact; n640 expands only routed banks/flat routers.
No disk bank/model download/GPU/T4. Initialization and random input
preparation are excluded from the operator timing but their full cost/RSS
is recorded. The resulting operation outputs are independent input-ready
probes: there is NO causal model composition or accepted-token timing.

Codes are stored bank/output-tile32/input-group/row-within-tile. A table is
FP32 `T[query,group,code]=sum_{z=0..3}palette[z,code]*x[query,4group+z]`.
AVX2 loads eight adjacent U8 indices, widens them, gathers table values and
accumulates across groups; row scales are then applied. All284 source
projections execute. Four selected routed banks reuse a single gate/up
input table per tensor, but down gets four distinct tables. K-B and V-B
each get32 distinct head inputs. No implicit cross-projection palette reuse.
Flat F32 routers scan all n×1536 coefficients and select stable top4; actual
selected code/scale banks use those results. Source normalization, SwiGLU,
router bias/softmax/mixture, attention/RoPE/KV/embedding and norms are NOT
executed. Their descriptor bytes remain priced; they need later whole-model
integration. Timing includes all router scans/top4, table construction,
encoded matvecs/scales and OpenMP dispatch, not output hashes/logging.

## Controls, timing and frozen stop rules

- Header catalogue source shape/payload reconcilations remain300. New
  d%4/output%32/bank/query dimensions,309 count,284+25 split,75 routed
  inventory, complete406,405,120 lookup and70,352,896 table-entry counts
  must match before timing. Code storage is physically touched in full.
- On EVERY projection/bank, eight rows spanning first to last must be
  byte-exact to a separately indexed scalar FP32 LUT sum. Every router
  row and those projection rows also compare to decoded original fixture
  coefficients in FP64; pooled relative L2<=1e-5. This is numeric decode
  fidelity of fixtures, not approximation error versus pretrained BF16.
- Add128 to the first addressed table value: the projection must change
  against the unchanged palette. Restore it before timing. All outputs
  must be finite. The complete output/route hash sequence must be identical
  across three timing repetitions at the same n.
- First run n64, three repetitions of the SAME10 input fixtures:2 warmup,
 8 measured operator timings each. Save all30 observations, hashes, union
  of actually selected IDs, allocation/RSS/initialization and total time.
- Repeatability requires max/min of the three measured medians<=1.10.
  If it fails, stop as inconclusive, with NO n640/optimization promotion.
- All three n64 medians must be<=14ms, leaving the existing6ms complete
  target balance for missing operators. If this fails with stable timing,
  reject this UNCHANGED native layout BEFORE palette training/n640.
- Only if n64 passes, run n640 identically: numerical controls, stable
  hashes, repeatability<=1.10, all medians<=14ms, and median large/small
  ratio<=1.25 required. This two-run ratio is a scoped fixture measurement;
  no generalized scaling claim from it. Synthetic added banks remain
  ineligible as quality/useful-capacity evidence even if timings pass.

Any cost-only pass merely licenses a separately frozen source-bound palette
experiment with matched Cartesian baseline using already verified298 inputs.
No training, full quality or>=50 accepted-rate promotion is automatic.
Stable cost failure closes only this specified native implementation/fixture
profile. It cannot refute every vector dictionary, layout or real distribution.

## Resource budgets and reproduction

Expected local CPU minutes, hard600s per native child,40GiB peak RSS,
120s compile, fewer than10MB source/spec/log/result/executable outputs.
Require at least8GiB free RAM before n64 and36GiB before n640. Observe child
RSS every5s, no model/performance process overlap; preserve partial logs and
failure stage. After each repetition C also checks its600s/40GiB bounds.
No T4/new external resources or agent delegation. Do not alter gates after
results; an apparatus repair must preserve the original failed execution.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth301_vector4_lut_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth301_vector4_lut_preflight_result.json
```

The full donor-quality/useful-large-n/same-artifact accepted50token/s goal
is unchanged. Component costs can reject a prospective route; they do not
establish that route's quality or completion.
