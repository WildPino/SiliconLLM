# Preserved dense organs: physical-worker parity passes, 20ms budget fails

14 October 2026. Pipeline20304 exited0 after97.078s, OS peak2,252,173,312B,
no GPU or RESERVED. Compilation, all384 measured groups, full stored audit,
input/output seal and10-file/813,637,712B family closure completed.

## Same arithmetic, actual placement

The same binary performs23 BF source contextual generators and24 original
I16 readouts, in layer order, on actual supplied own-history operands. Arms
1/3/6 use distinct physical cores in group0, dynamic teams disabled, static
row scheduling, unchanged AVX lane/dot/rounding arithmetic. The serial arm
calls the unchanged functions. OpenMP passive wait and zero blocktime prevent
idle workers spinning during that arm; parallel wake/barrier costs remain.
Six actual thread identities and single-core masks are verified before/after.

The separate stored auditor executes2,944 original serial generator calls and
3,072 original readout positions. All8,832 candidate generator calls and9,216
candidate readout positions match:177,629,184 raw/GG/field cells and56,623,104
integer/F64/BF cells exact. All384 measured groups, orders and accounting pass.
Original independent leaf-arithmetic qualification is inherited for the same
operands. This is full scheduling parity, not independent new source-model
arithmetic or a new chatbot-quality panel. Audit21.546s.

## Measured partial groups

Median milliseconds;64 groups per cell. Flush is outside timed groups.

| Physical workers | Resident whole / generators / readout | Evicted whole / generators / readout |
|---:|---:|---:|
|1|78.90790 /61.20140 /17.79655|75.19905 /58.58110 /16.18695|
|3|45.02860 /28.94605 /14.74960|45.47035 /29.21445 /16.44020|
|6|48.78380 /28.86820 /17.70060|50.79675 /29.81295 /19.18410|

Separate component medians are not an additive decomposition of whole medians.
Three-worker minimum/p90/maximum are34.89090/61.02105/175.12930ms resident,
33.93570/66.07366/199.09460ms evicted. All six arms/modes have minima above
33ms. Three workers improve these observed medians relative to one, but six
do not improve them. The latter is not a universal optimal-worker claim.
Shared desktop, scheduling, passive wakeup, interleaving and memory traffic
can affect scaling; no causal bandwidth-saturation diagnosis is established.

**Decision:** every tested complete partial group exceeds20ms before omitted
operators. Scheduling-only parallelism in this implementation does not provide
a viable50-token pipeline. Preserve the exact parallel primitives for later
compressed/conditional artifacts; do not expand a worker-count grid or publish
the reciprocal group time as chatbot token rate.

## Limits and next decision

Inputs h and normalized readout r/a are supplied from the qualified histories.
This program does not compute convolution/state/norm/input quantization, FFN,
attention, routing, head, tokenizer or frontend. It is not a joined decoder.
Eviction is not physical DRAM measurement. The apparent slowdown from historical
single-thread baselines is not a controlled causal regression: this inquiry
interleaves the two organs and has its own shared-desktop measurements.

Preserving24 layers requires reducing active coefficient traffic/arithmetic,
not only assigning the current rows to more cores. Quantization, exact encoded
readout, conditional FFN/generator representations and adapted depth changes
remain hypotheses with distinct numeric, cost and composed-quality obligations.
Unexecuted depth6/12 arms remain unselected; frozen gates are unchanged.
Historical experiments will be cataloged before choosing the next variable.

Authoritative [binding](physical_worker_organs_20261014_binding.json),
[report](physical_worker_organs_20261014_result.json),
[audit](physical_worker_organs_20261014_audit.json),
[closure](physical_worker_organs_20261014_closure.json),
[pipeline](physical_worker_organs_20261014_pipeline.json) and
[terminal](physical_worker_organs_20261014_pipeline.terminal.json).
Goal ACTIVE/INCOMPLETE; no numerical worker from this screen remains live.
