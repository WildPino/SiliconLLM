# METH-126: exact shared-A storage for the centered E1280 bank

## Uncertainty and prior evidence

METH-124/125 establish native routing and selected-factor cost for the
quality-gated centered E1280 BF16 bank, but the 893,141,032-byte export repeats
each parent's rank-8 A matrix ten times. METH-107 trained only child B; a
layer-0 byte check finds all ten sibling A matrices equal. Whether this holds
for every parent in all 24 layers, and whether a native index can read a
single A without changing the numeric output, remains to be checked. This
storage change cannot establish a LUT benefit, compact donor core, or full
engine throughput.

## Fixed operation and gates

- Input is the METH-124 bank with SHA-256
  `8e4ef1f8abea39d9562e8b617f460e89591fa2b2d7fa7478fac0901804064580`.
  Preserve the router, child keys, and all B bytes. For each layer and parent,
  require the other nine A byte arrays to equal child zero. Abort without a
  promoted output on any mismatch.
- Write a versioned `M126FB01` bank containing one A array per parent and all
  child B arrays. The native reader addresses A by `child_id / 10` and B by
  `child_id`. Check complete readback and the expected 496,779,304-byte file
  length; record SHA-256.
- The 96 METH-124 fixture rows must all pass route, gate and residual checks
  against the compact bank, including exact route identity and the prior
  residual tolerance. The planted negative fixture must fail. Compare the
  original and compact C checkers' numeric summaries.
- Run the METH-125 256 varied-activation positions with the same five repeats
  and compiler/thread settings for old and compact E1280 banks. Their routes
  and deterministic checksum must match. Report factor and total component
  medians, bank/RSS bytes and unique selected A/B bytes. Do not infer cold
  DRAM or end-to-end token speed from this microbenchmark.

## Cost, stop, and decision

The export and checks are CPU-only, bounded to 15 minutes, 3 GB additional
disk and 4 GB process RSS. Stop on an input hash mismatch, a nonshared A,
malformed lengths, any positive parity failure, or a negative fixture that
unexpectedly passes. If all gates pass, adopt shared-A as an exact storage
representation for this checkpoint; quality evidence transfers because the
selected factors and arithmetic are byte-identical. If any gate fails, keep
the METH-124 bank and investigate before claiming an improvement.
