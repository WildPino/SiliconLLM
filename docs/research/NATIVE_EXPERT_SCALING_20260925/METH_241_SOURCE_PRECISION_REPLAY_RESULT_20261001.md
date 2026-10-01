# METH-241: exact BF16 target replay qualifies an isolated source-value test

Frozen at `63bdd3b`; session80461 completes,exit0. Source/snapshot/capture/
native bindings and fit-only canonical geometry pass. Original BF16 FFN
reproduces **all58,720,256 captured output elements exactly**:zero mismatches,
zero SSE. Original BF16-weight/FP32 smooth FFN has normalized SSE
**1.3428216602676273e-5** against those same targets. Frozen replay<=1e-8
and tenfold improvement gates pass.

Child156 has18 fit states,one unique input and one unique target; its
BF16-rounded center equals that input. Relative L2 to target mean:

| Value/function | Relative L2 |
| --- | ---: |
| Original FP32 source |.0035591543|
| Canonical BF16 source |0|
| Fitted E16 parent15 |.0005493147|
| FP32 source child prior156 |.0035591543|
| Fitted E160 child156 |.0034976623|

The FP32 source reset is6.48times less accurate than the already fitted
parent here. This explains a local precision mismatch, not the entire
22.39% count loss. Rounded-backward BF16 autograd sensitivity differs
from the FP32 smooth analytic response by.302%–.492% across15 fixed
direction probes. It is not a mathematical Jacobian of discrete rounding,
and no source-derivative equality or new prior is established.

**Decision:** separately test canonical BF16 source values only, preserving
actual fitted weights, routing and regularization. See
[METH-242 frozen protocol](METH_242_BF16_VALUE_PRIOR_PROTOCOL_20261001.md).
No new validation,fitting,native timing or full quality in this diagnosis.

[Raw result](meth241_source_precision_replay_result.json),SHA256
`33568b9dc99a1ba93d43346db58a97865815e35cb8c669ea8656d44dd70b1980`.
Runtime5.110s after imports,RSS2.511GB,GPU peak965.571MB,local3060,six
host threads,no T4. All128 replay blocks and sensitivity probes retained.
