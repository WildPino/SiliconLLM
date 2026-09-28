# METH-132: use fifteen levels in the same child-B nibble footprint

METH-130 reduced the centered E1280 bank from 496.8 to 276.6 MB with
seven signed levels per child-B coefficient and improved selected-factor
CPU time. METH-131 then failed the frozen blinded grounding gate, 34
versus 33 unsupported and 14 versus 12 severe claims, despite passing
the automatic full-model gates. Q7 used only seven of sixteen values
available in each four-bit nibble. This experiment asks whether using
signed levels -7..7 reduces factor distortion without losing the
storage and lookup-table cost gains. It does not reuse METH-131's
responses as a quality test or tune to their individual errors.

Bind the unchanged quality-gated METH-126 BF16 shared-A bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`
and METH-125's 256 actual E1280 pre-MLP positions SHA-256
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
Keep every router, shared-A and selected-child rule identical. For each
eight-value B output row, use initial scale max absolute coefficient/7,
round-to-nearest-even clipped to -7..7, then refit a nonnegative FP32
row scale by least squares for the chosen codes. Zero rows retain zero
scale/codes. Pack low/high nibbles as code+7, four bytes, plus FP32
scale: eight bytes per output row, unchanged from Q7. Export a new
versioned `M132FB01` bank and verify every router/A byte and code/scale
record on readback. The native paired-LUT checker must reject invalid
nibble 15 and nonfinite scales, and compare actual routes, gates and
residuals with the exact BF16 bank.

On all 256×24 fixed states require zero route/gate mismatches and
nonfinite residuals, pooled residual relative L2 <=0.025, and
95th-percentile per-vector relative L2 <=0.05. Report maximum,
per-layer error and route coverage. Require bank size <=300,000,000
bytes and five alternating-order paired CPU factor repetitions with
Q15 median <=1.20× exact BF16. Stop for a failed binding, >10 minutes
CPU, >4 GiB process RSS or >500 MB new disk. Local CPU only; no T4.

A component pass licenses a new-source BF16 full-model automatic and
arm-blind grounding audit with the same thresholds as METH-131, then
native same-artifact numeric parity and accepted-token throughput.
This component experiment does not establish quality conservation,
useful learned expert growth beyond E1280, 10B/100B transfer, or the
>=50 token/s end-to-end goal.
