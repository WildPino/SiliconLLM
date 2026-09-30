# Candidate expert-count geometry: reuse the existing child projection

METH-204–209's third tier uses a separate random rank-32 projection
and nine globally shared keys; mode-specific biases do not pass old
raw reserve even with tripled support. Change the key geometry and
its coordinates, rather than recalibrating the same representation.

**Proposal, untrained:** use the already computed E1280 child-router
projection output, normalize a separate copy, and fit nine local
rank-32 content keys per selected E1280 parent. Share those keys
between text/ChatML, retaining two mode-conditioned parent-bias banks
and the structural slot. The old child choice uses its original
unnormalized projection output and keys, so the proposal need not
change the validated E1280 route. Only the new leaf choice uses the
normalized coordinates. Its keys, calibration and useful child B
functions still need to be fitted and independently verified.

A read-only [input/layout preflight](reused_child_projection_key_proposal_ledger.json)
binds METH-107 child checkpoint SHA
`15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`
and METH-135 third-router SHA
`692db3dd1a647debb051ef66612263b5e16ef27c73978e9abd3beb6b3229abc9`.
It reads each `expert_state[layer].child_projection` and compares
its FP32 bits with `meth136_matched_sparse_train.load_third()`.
All 24 pairs have shape 32x896 and **none is byte-identical**.
Therefore this is a changed router representation; old third-tier
route metrics cannot be carried forward as if they were unchanged.
Per-layer hashes are retained for exact input binding.

| Layout quantity, 24 layers / four selected parents / nine leaf choices | Current parent count 1,280 | Hypothetical parent count 12,800 |
| --- | ---: | ---: |
| Shared-across-mode parent keys, FP32 | 35,389,440 stored bytes | 353,894,400 stored bytes |
| Two mode-bias banks, FP32 | 2,211,840 stored bytes | 22,118,400 stored bytes |
| New selected-key/bias weight addresses per token | 114,048 bytes | 114,048 bytes |
| Additional projection matrix weight addresses | 0 | 0 |

These selected-address counts hold four selected parents and nine
leaf choices fixed. They do not price the route that selects those
parents at tenfold count, prove constant full-token cost, or establish
useful larger-model capacity. The larger column is a layout hypothesis,
not a trained 100B result. Native access must be measured without
assuming key-bank or expert-pool cache residency.

Combining the first column with METH-211's proposed E1280 weight
ledger gives 559,908,224 ideal weight bytes/token, 91,776 below the
560 MB design ceiling. Avoiding a second 2,752,512-byte projection
read makes this candidate layout feasible before fitting. This is
not a proven composition: normalization/lookup/structural execution,
new-state route health, distinct specialists, untouched quality and
actual native precision/traffic/rate remain required gates.
