# METH-130: Q7 child-B LUT on the quality-gated E1280 bank

## Question and decision

METH-123 established donor-relative quality for the centered BF16 E1280
bank. METH-126 made its sibling A matrices shared, reducing its stored
size to 496,779,304 bytes without changing inference. METH-125 measured
1.068 ms/token-equivalent for the selected BF16 factor computation in
one varied-position shared-A run; METH-127/129 remain too slow overall
because of their FP32 core. The remaining child B tensor is the dominant
bank storage term and grows linearly with child count. A CPU LUT
representation could reduce its RAM scaling, but numeric damage and real
CPU cost on this quality-valid bank are unknown.

This experiment decides whether a fixed Q7 child-B format is worth a
full-model quality audit. It does **not** itself promote a quality-valid
LUT model, establish 10B/100B transfer or prove that new child slots are
useful. Fixed-state component agreement is weaker than autoregressive
quality. If the screen fails, retain the exact BF16 bank and pursue a
different B representation.

## Frozen assembly and method

- Input: METH-126 `M126FB01` shared-A bank SHA-256
  `1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`,
  derived from the quality-gated METH-107 checkpoint. Actual BF16
  pre-MLP vectors: METH-125 E1280 varied file SHA-256
  `f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`,
  256 positions across 24 layers from the viewed METH-121 sources.
- Keep the FP32 parent/child router, BF16 shared A, top-four parents,
  selected child IDs and BF16 gates unchanged. Quantize only each
  eight-coefficient child B output row to signed levels -3..3. Initial
  scale is the row maximum absolute coefficient divided by three;
  round-to-nearest-even and clip codes to -3..3. Refit one nonnegative
  FP32 row scale by least squares against the chosen codes. If the row
  is zero, use scale zero and all-zero codes. Pack adjacent codes in
  low/high nibbles, four bytes per row, and store one FP32 scale per row.
  A row is eight bytes rather than sixteen BF16 bytes. The exporter
  must verify exact readback of all packed codes and scales from a
  versioned bank with unchanged router/A bytes.
- In native C, construct four 256-entry activation-pair tables per
  selected child from its eight BF16-rounded hidden activations. Each
  table maps a packed nibble pair to a float contribution. A B output
  row reads four code bytes plus its scale, sums four table lookups,
  multiplies by its scale, then retains METH-124's BF16 part/gate/output
  rounding. Invalid nibbles or nonfinite scales fail. Compare against
  the original BF16 bank on identical routed states; no route change is
  allowed.

## Component gates and stop

- On all 256×24 fixed states, require finite outputs, exact parent and
  child IDs and gates, pooled residual relative L2 error <=0.10 and
  95th-percentile per-state residual relative L2 <=0.20. Also report
  maximum error, by-layer errors and route coverage. Zero-norm vectors
  are reported separately rather than silently divided by epsilon.
- Require the Q7 bank <=300,000,000 bytes and selected-factor median
  CPU time <=1.20× the exact BF16 bank in a same-process paired
  five-repetition varied-position run. Keep route cost separate. The
  two arms must use identical source positions and token order; record
  peak RSS and physical bank size. Timings are a component screen, not
  end-to-end accepted-token rates or cold DRAM measurement.
- Stop for a binding/format failure, nonfinite output, >10 minutes CPU,
  >4 GiB process RSS or >500 MB new disk. Use local CPU only; no T4.

Passing the screen licenses integration with the actual BF16 donor and
centered E1280 composition and a **new-source** document/generation/task
and grounded-semantic quality protocol before any quality promotion.
The C runtime and `engine.c` integration, >=50 accepted token/s on the
same artifact, and a learned-child count ladder remain subsequent gates.
