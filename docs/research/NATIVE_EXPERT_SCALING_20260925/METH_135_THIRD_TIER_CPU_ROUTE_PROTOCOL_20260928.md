# METH-135: CPU cost of a ten-way third expert tier on actual states

## Decision and binding

METH-134 establishes a one-layer CPU-master sparse-gradient path for
an exact-clone E1280→E12800 expansion. It does not measure the added
CPU choice among ten grandchildren per selected E1280 child. This
experiment decides whether that *routing component* is affordable
before spending resources on learned E12800 training. It cannot
establish useful distinct capacity or a quality-valid LUT factor
format: the new third-tier keys are seeded but untrained, and Q7/Q15
factor formats failed blind full-model grounding.

Bind METH-126's exact centered BF16 shared-A bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`
and METH-125's 256 actual E1280 vectors SHA-256
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
Use the METH-134 seed 134134: for layer `l`, generate one 32×896
FP32 projection and 1,280×10×32 FP32 keys with CPU PyTorch generator
seed `134135+l`, in that order. Freeze a versioned sidecar and verify
its byte-for-byte readback. Keep all existing parent/child router,
four active selections, BF16 gates, shared A and exact BF16 B bytes.
The third tier only chooses an ID in `[child*10, child*10+9]`; its
factor is an exact copy of that child for this cost test.

## Frozen checks and cost stops

- On all 256×24 actual states, require the original E1280 child IDs
  and gates to be identical with or without third-tier routing, every
  grandchild ID divided by ten to recover its source child, all finite
  scores and a nontrivial distribution of grandchild IDs. The factor
  residual with cloned grandchildren must be bit-identical to the
  exact E1280 residual. Reject a corrupted sidecar magic header.
- Make five paired single-thread route-only timing repetitions,
  alternating order. Require third-tier median <=3.0 ms/token-
  equivalent over 24 layers and <=2.0× the matched E1280 route
  median. Report all repetitions, coverage and nominal addressed key
  bytes per token. This is warm repeated-position component timing,
  not cold random DRAM or end-to-end accepted-token throughput.
- Stop after two minutes, 3 GiB RSS or 100 MB newly written disk. Use
  local CPU only, no T4. A cost pass licenses full-model third-tier
  training/quality work; it does not by itself show quality at E12800,
  a CPU LUT format that conserves quality, or 10B/100B transfer.
