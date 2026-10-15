# Large-family protected-projection rejection result

15 October2026. **Numerical/placement/custody PASS; physical variant REJECTED.**
Freeze6f98977. Source geometry only, all tensor values synthetic; no donor
weight acquisition, source forward, training or GPU call. This is an incomplete
favorable subset, not an exact source core or a universal CPU lower bound.

| Independent three-core process | Median operator, ms | Median with check/hash overhead, ms |p95 operator, ms |
| --- | ---: | ---: | ---: |
|0|34.39055|35.67330|39.97590|
|1|50.94880|52.48290|66.74720|
|2|53.87885|55.20445|114.38780|

The predeclared median of process medians is50.94880ms, above20ms. Even the
fastest process median34.39055ms exceeds the entire accepted-token allowance.
All72 fixed-fixture timings and dispersion are retained. No run or outlier was
dropped. Logical CPUs0/2/4, actual worker thread masks/readbacks, all-row scalar
header-reference controls, router IDs/masses, output hashes and before/after
input seals passed. Header reference shares its format code with the active
kernel; earlier independent decoded layout controls are complementary, not
actual source-weight parity. All three workers and controller exited0.

Measured operations:36 layers of physically packed QKV/O projections with F32
biases, Q8 activations, RMS norms, synthetic residual addition and F32 router/
top4 softmax. Q4 projection payload557,383,680 bytes includes2880->3072 padding;
router weights53,084,160 bytes. Operator timers exclude full-output hashes and
finite checks; wall timers include them. RoPE, attention scores/value work,
F16 KV traffic, sinks, experts and vocabulary head are excluded. This is not
measured physical DRAM traffic, full model execution, quality or useful rate.

Total controller time13.953s, combined OS peak649,322,496 bytes, within600s/
4GiB/128MiB. Input, compiler/runtime, process-identity and output-log hashes
are in the [complete result](protected_core_120b_result_20261015.json),
[terminal](protected_core_120b_terminal_20261015.json) and
[ownership](protected_core_120b_ownership_20261015.json) receipts;
[protocol](PROTECTED_CORE_120B_CPU_PROTOCOL_20261015.md).

## Decision

Reject this Q4_K protected-projection implementation as the basis for a20ms
family conversion. Head retrieval and fine-grained experts alone do not repair
this measured variant. Do not acquire the entire120B donor or fit selectors
on the strength of the earlier ideal byte forecast. Do not infer that every
possible kernel, core representation or native MoE family is impossible.

A further same-family path needs an explicit core transformation and an
actual-weight/function test, with complete cost and quality still separate.
Q/O rank reduction is one unvalidated hypothesis; source-derived error and
calibration economics must be checked before treating it as a solution.
No synthetic fast student or larger parameter count can close useful capacity.
