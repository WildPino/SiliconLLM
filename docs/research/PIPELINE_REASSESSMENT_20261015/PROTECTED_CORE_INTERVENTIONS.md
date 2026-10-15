# Protected core interventions: retrieved head and block verification

15 October2026. Proposed mechanism screen; no new model/rate result.

## Retrieved vocabulary head

The primary [vector-index output-head paper](https://arxiv.org/html/2608.27460v1)
uses HNSW over output rows and evaluates CPU batch-one decoding. It reports
Recall@2 around99% at ef200 on its tested small models, with a slight relative
quality loss; its headline speedup uses F32 dense baselines. It explicitly warns
that gains can shrink against quantized baselines. The
[official implementation](https://github.com/martinloretzzz/vector-index-embedding)
uses a modified HNSW implementation; neither the paper nor that code validates
our model, packed format, hardware, exact sampling or native C engine.

This suggests a different experiment from local M129/M509/M510 full-I8 scanning:
retrieve a small token shortlist without reading every vocabulary row, then
refine it against the source rows. Approximate retrieval can miss the winner;
shortlist-only normalization changes probabilities. No exact full-distribution
claim follows. EOS/structural token inclusion must be bound in advance.

A future assay must use actual final hidden states, FIT-only index/precision
choices, untouched DEV recall/top1 and own-history generation, and time search,
row reads, refinement and special tokens together against the SAME packed dense
head. Include index build time/pool RAM and both recall and output consequences.
No new source forward or index build is launched by this note. Removing the
head does not remove attention, state, parent router or expert cost.

## Block verification: existing counterevidence

[Phase63](../../ENGINE_PLAN.md) and
[the scale-up architecture](../../SCALEUP_ARCHITECTURE.md#38-execution-model--measured-e5-design-gate--phase-63-engine-2026-07-05-ar-lossless--threads-block-verify-built-conditional)
already implement exact greedy verification and Activation Replay state commit.
The compliant small-model cold-weight test achieved only0.679x AR speed; expert
unions near iid saturation made routed-weight amortization0.96x/0.86x at K4/K8.
These results do not test today's large donor, but reject a free amortization
assumption. Original custody limitations remain in ENGINE_PLAN's audit.

For K proposed positions, charge draft + shared stream + routed expert UNION +
K-position projection/recurrence/attention/head arithmetic + state commit,
including rejected work. Require total seconds <=0.020 * expected accepted
positions. A FIT-trained/DEV-evaluated offline draft acceptance test plus actual
route-union census is the cheapest next falsifier if this path becomes credible.
Exact sampled-distribution verification additionally needs an explicitly correct
rejection/correction rule and target probabilities. Stored BF16 trajectories
alone do not establish that distribution. No repeated block runtime grid is
selected now.
