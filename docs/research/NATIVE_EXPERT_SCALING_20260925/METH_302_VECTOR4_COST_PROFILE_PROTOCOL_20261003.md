# METH-302: attribute unchanged vector4 native operator cost

301's full-feature vector4 descriptor fits, but actual native medians81.63-
83.61ms stably FAIL14ms. Remaining uncertainty: where is the cost, and would
table-only tuning address it? Freeze source/controller/this protocol before
profile observation. This is a bounded diagnostic, never a regrade of301.

Include the immutable301 C source with only its main symbol renamed. All
fixture setup/input generation, router/table/matvec/scaling/arithmetic/layout,
thread count and self-test code are reused exactly. New main times the same
operations in the same order, separating each tensor's table and projection
duration. Map every309 tensor by source-bound shape/role to MLA130, routed75,
shared75, dense3, head1 or router25. Reject other inventories. Actual new
phase60 opt-in selector is `SILICON_VECTOR4_LUT_PROFILE`; no generic donor
port/default or source quality execution. Only n64, no larger bank.

Pin301 raw result/controller/C hashes, reuse its exact catalogue SHA, and
compile the same flags. Require same allocations, all30 output+route hashes
EXACT301, inherited scalar/FP64/negative controls, finite outputs and sum of
instrumented component durations within1ms of each whole operator duration.
No timing-based parameter/quality-source selection. Three same10-input
repetitions, two warmup/eight measured each, inherited6threads/n64 fixture.
Repeatability max/min repmedian<=1.10; otherwise attribution inconclusive.

Report per-organ table, matvec and combined median, plus actual per-observation
aggregates before medians; do not sum individual medians as if exact. Record
all observations/hashes, source/compiler/binary/catalogue bindings, resources
and logs. If projection-only matvec median fraction>=70%, next mechanism
must reduce active matvec/lookup complexity; table-only tuning is not licensed
as a sufficient rescue. Otherwise inspect construction/router/matrix balance.
Also calculate a CONDITIONAL zero-table bound and required matrix speedup
to14ms at unchanged observed table/router costs. Those are accounting
deductions, not measured counterfactual implementations or physical bounds.

Reuse301 controller's native launcher/resource/numeric checks without editing
it; set new executable and isolated `results/native_expert_scaling/meth302_profile`
log directory. Existing log base names describe the reused301 backend. Hard
120s compile/600s child/40GiB RSS/8GiB RAM availability, expected seconds and
<10MB outputs. No model/performance overlap, GPU/T4/download/optimization,
new donor inference or additional capacity. Preserve any apparatus failure.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth302_vector4_profile.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth302_vector4_profile_result.json
```

The fixture operations are input-ready and not a causal model. 301 failure,
source low-bit quality stops, useful-large-n and same-artifact50-token/s
requirements remain unchanged whatever this attribution shows.
