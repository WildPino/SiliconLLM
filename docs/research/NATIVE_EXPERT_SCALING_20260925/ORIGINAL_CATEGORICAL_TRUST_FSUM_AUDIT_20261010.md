# Stored grouped-step audit: independent summation on Windows

10 October2026. Supplement to the immutable
[producer/protocol](ORIGINAL_CATEGORICAL_TRUST_STEP_PROTOCOL_20261010.md).
Producer is LIVE, unchanged. No stored audit has run yet.

## Concrete gap

Read-only runtime inspection found NumPy2.4.6 Windows longdouble itemsize8,
mantissa52 bits: it aliases F64, so a dtype cast alone does not provide the
independent norm summation intended by the original audit. No extended-precision
or independently summed norm PASS is admitted from that planned branch.
Retain original frozen source/protocol; no changed-model/native/backward replay.

## New adapter and unchanged gates

In a fresh stored CPU audit process import the frozen trust-step tool, replace
NumPy's module sum ONLY for 2D F64 arrays reduced over axis1 with `math.fsum`
over each row. Keep all other sum calls unchanged. This covers group parameter,
gradient and actual-displacement squared norms and directional dot sums, and
per-label KL sums. All source raw F32 squares are exactly representable in F64;
fsum provides a distinct accurate accumulation before square root. It does
not claim arbitrary exact real products or directed intervals.

The frozen audit executes every input/output hash,92 master/90 source gradient
check, all three proposals/all groups, full110-field packing/quantization/stable
cell counts/witnesses,54 label metrics/routes, native commands/queries/receipts,
counters/resources/selection checks. Change neither data, normalizers, decisions
nor original tolerances:1 ULP proposal,1e-9 group/aggregate,1e-8 metrics.
Independent full-V probabilities remain shifted sequential logaddexp.

Record adapter/protocol hashes and actual dtype precision in the resulting
adjudication. Replace the misleading longdouble-verification label with an
explicit independent-fsum label; scope explains no extended-precision claim.
Adapter does not generate candidates or execute an optimizer/model/native call.
One full stored audit through this adapter replaces the unexecuted planned
audit; no redundant audit pass is required to rediscover the alias.

## Freeze, budgets and failure custody

New launch binding seals unchanged numeric binding, immutable producer/frozen
script/holder, adapter, this supplement, runtime probe and Python executable.
Original producer freeze is kept as numeric freeze; separate adapter/holder
freeze records the new commit. Held audit1800s/OS20GiB/output2MiB/log8MiB,
CUDA hidden/CPU1; unchanged budget, no T4. No audit until original producer has
closed and terminal/full outputs are sealed; never overlap owned benchmarks.
Preserve first faults and complete audit even candidate quality/descent FAIL.
