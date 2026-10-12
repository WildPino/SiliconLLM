# Auxiliary generator lift into the original S6 equations

12 October2026. Prospective arithmetic experiment; goal ACTIVE/INCOMPLETE.

## Uncertainty and new variable

Current source-informed initializer transfers no trained sequence core. Previous
fixed/copied/mixed384 channels and linear decoders already failed local output
budgets; do not repeat their projection objectives. This tests a different
operation: exact generator transport into S6 by adding auxiliary feature channels.
It makes a reproducible transfer component and exposes its active cost. It is
not permission to conclude the goal with a donor-sized generic runtime.

For the actual Falcon-H1 one-group mixer, source x/B/C are SiLU outputs of
separate depthwise4 convolutions; delta is softplus of a linear input projection.
The original Mamba1/S6 generator reads B/C/delta linearly from activated x.
Add512 B/C channels and96 channels for48 signed delta pairs:

```text
features = [x_source, B_source, C_source, SiLU(a), SiLU(-a)]
SiLU(a) - SiLU(-a) = a
B,C = feature selectors
delta_primary = softplus(a_head + source_bias_head)
A_primary[channel,:] = source A_head
Dskip_primary[channel] = source D_head
gate_primary = source gate projection
```

Auxiliary output columns and gates are exactly zero. Copy convolution coefficients
and biases for x/B/C; delta channels use identity convolution, opposite input
rows and zero convolution bias. Required dimensions: D2048,DN3680,N256,DT48.
No feature or state coordinate is discarded, no projection before nonlinearities
is treated as exact. Auxiliary state calculations may be eliminated because their
output columns are zero, but this experiment uses the original dense recurrence
and does not claim that optimization measured or implemented in C.

Source postgate one-group RMS needs a dynamic scalar unavailable in the original
S6 output. Fold source norm gamma into output columns; compute denominator over
the3072 PRIMARY gated channels only, then divide the output by that scalar.
This extension is represented in the numerical assay, not implemented in engine.c.
Including zero auxiliary gated channels in the RMS mean would change its scale.

## Predeclared experiment and decisions

Actual pinned source coefficients at sites0/12/23. First13 IDs from the shortest
existing FIT history (58 IDs), read actual source embeddings and each site's input
norm gamma. Inputs at sites12/23 are injected normalized embedding vectors, NOT
captured source intermediate states; this is an operator identity assay.
All real-arithmetic coefficients/inputs computed F64, no BF16 cache parity claim.
No model loader, generation, loss, optimizer, new teacher data, GPU, T4 or C call.

Independent source factorization versus compiled original-S6 field operations:
all x/B/C/delta/gate/scan output/denominator/output relative RMS<=1e-10,
Python scalar first-head/first-channel recurrence over all13 positions with
math.fsum <=1e-10. Actual generated operand arrays retained for all3 sites.
Two mechanistic ablations: omit negative delta-pair features; omit postgate norm.
Report errors without selecting a model based on them. All sites and ablations
are evaluated even if a numerical/engineering result fails, unless a resource
bound or execution fault intervenes; preserve first failure with stage/receipt.

A PASS qualifies a structural arithmetic transport component. Next compact
conversion must price/represent the nonlinear generators, required normalization
and remaining attention/FFN/depth composition jointly. A failure requires local
operand attribution before proposing transfer. Neither result admits useful
chatbot capacity, same-artifact speed, useful large n, or generality.

## Resources and custody

CPU NumPy one thread, CUDA hidden,300s including before/after hashes, OS4GiB,
outputs16MiB. Full source weight file and every bound code/runtime input hash
checked before and after; principal runtime binaries, not full DLL-tree binding.
Exact source/protocol frozen before prepare; actual binding and invocation frozen
before run. Report PID/create/command/OS peak, source/derived extents and limits.
Foreign publisher preserved, process isolation and clean timing not admitted.
Inputs immutable; outputs exclusive new namespace. No old campaign restart.
Original engine.c unchanged. The final goal remains useful compact capacity with
>=50 accepted tokens/s, useful n driven by RAM, and multiple source families/scales.
