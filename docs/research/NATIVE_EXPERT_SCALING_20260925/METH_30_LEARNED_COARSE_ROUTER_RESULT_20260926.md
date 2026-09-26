# METH-30: fixed learned coarse gate misses route-fidelity gate

The [prospective protocol](METH_30_LEARNED_COARSE_ROUTER_PROTOCOL_20260926.md)
froze the METH-29 balanced 16×8 expert groups, E128 rank-8/top-4
adapter and packed R8 core. It fitted a per-layer D896→SiLU64→16
coarse classifier to predict groups containing the exhaustive fine
router's top-4. The 32 training and 8 internal validation source
rows were fixed before scoring in the [manifest](meth30_learned_coarse_router_manifest.json),
SHA-256 `07d14a2296048a12d821e26b6d98635d410b51224c48a914fe19001b20407108`.
The external 24-prompt route audit reused the METH-27/29 manifest,
so it is diagnostic, not a new independent promotion set. The
[machine result](meth30_learned_coarse_router_result.json) has SHA-256
`af56c4ab4785953d5f7cea068ee2ad52294c993202663fdc10499c62b96abd94`.

| Case | 32 candidate rows: exact IDs / full sets | 64 candidate rows: exact IDs / full sets |
|---|---:|---:|
| Internal validation, 49,152 input-layer cases | 78.698% / 37.821% | 93.562% / 77.409% |
| Reused external prompts, 147,456 input-layer cases | **75.553% / 32.319%** | **91.609% / 71.523%** |

With 64 candidates the external exact-ID inclusion is 90.343%
code, 94.673% prose and 89.810% technical/general. The minimum
layer is 87.923%; mean missed selected gate mass is 0.06441.
The frozen gate requires ≥99.9% pooled exact IDs, ≥99% full sets
and ≥99% IDs in each category. Both candidate budgets fail. The
external 64-candidate recall improves on METH-29's 85.848% centroid
recall, but the increase is insufficient. Internal validation also
fails, so this is not solely a shift to the reused prompt set.
The training BCE fell from 0.516685 to 0.070909 across the fixed
20 epochs; training fit does not establish held-out route fidelity.

Command: `.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth30_learned_coarse_router.py --manifest-sha256 07d14a2296048a12d821e26b6d98635d410b51224c48a914fe19001b20407108 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth30_learned_coarse_router_result.json --gate-out results/native_expert_scaling/meth30_coarse_gate.safetensors`.
The native Windows `.venv` was used; the system Python lacked the
experiment dependencies. The run exited 0 in 62.532 s on the local
RTX 3060, peaked at 3.495 GB allocated GPU memory and ended at
4.173 GB RSS. The reload-verified gate artifact is 5,697,984 bytes,
SHA-256 `7c294fccf8145d86fe69d9cf13bc5510eca6bf088ceca4690febb340cf0d1c86`.
It remains a local ignored artifact; the manifest, runner and complete
route counts are retained in the repository.

**Decision:** reject this fixed post-hoc gate recipe before a C
implementation. At E128, its 64-candidate arithmetic is already
~129.14 D896 row-dot equivalents versus 128 exhaustive, before
nonlinearity and dispatch. This result gives no CPU LUT throughput
or large-E quality claim. A useful next route experiment must change
the training/index structure, then prospectively check route recall
and end-to-end quality on fresh inputs before measuring C cost across
a RAM-scaled expert ladder. Distinct learned expert expansion and
the existing R8 composition's generation repair remain separate
requirements for the goal.
