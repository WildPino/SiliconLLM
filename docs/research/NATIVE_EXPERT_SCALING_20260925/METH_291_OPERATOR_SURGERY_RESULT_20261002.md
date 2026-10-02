# METH-291: exact native closure, no single-group causal rescue

Frozenf4d45b0 before observations; session53504 terminal exit0,61.516s.
Same276 archive/all725 fields/all76 helpers. Consumed source0/full147-token
prefix/tail8, compact-core-only ablation. All-CPU hybrid reproduces original
285 native hidden AND full-head logits BYTE-exact across all8 rows. Original
GPU repeat is also byte exact. Thus the batch-wrapper/GPU residual/RoPE
apparatus closes against actual native execution before interventions.

Each restored group receives ACTUAL intervened inputs; original teacher
hidden states are never replayed. Restoring groups is a nondeployable oracle.

| Restored GPU group | Hidden median/max relativeL2 | Logit median/max relativeL2 | Fixed 50% reduction of all four metrics |
| --- | --- | --- | --- |
| None (all CPU) | .0277745/.0427415 | .0293126/.0542238 | Baseline |
| Norms | .0270834/.0431247 | .0293534/.0449204 | FALSE |
| Projections including head | .0229683/.0469126 | .0271181/.0548035 | FALSE |
| Attention | .0269358/.0466897 | .0299196/.0470474 | FALSE |
| Source FFN | .0296693/.0673788 | .0328798/.0767163 | FALSE |

All40 top1 choices equal original GPU. No group alone halves median/max
hidden/logit error; source FFN restoration worsens maximum materially.
Norm/attention improve logit maximum on these8 states, but no prospective
causal-rescue indicator passes. Do not select a speculative replacement or
claim universal group dominance. The error depends on interacting numerical
trajectories, despite small same-input errors and exact native closure.

EndRSS3,708,952,576bytes/peakCUDA1,427,121,664bytes; allresource stops pass.
CPU/GPU cooperate within numerical assay; no rate or CPU benchmark claim.
[Raw result](meth291_operator_surgery_result.json) SHA
`fc740ace69a054eecb9c43a48a2742770ed5767db45a0c7f2cd22e9c115e6483`
retains arrays/callback counts/loader/code/DLL/reference hashes.
[Protocol](METH_291_OPERATOR_SURGERY_PROTOCOL_20261002.md) supplies command.
Original285/288/289 numeric failures remain, no native promotion.

Next method decision must distinguish CUDA numerical-orbit fidelity from
actual donor-relative model usefulness. Exact native arithmetic/cache/loader
are reproducible; usefulness is UNMEASURED. A declared prospective evaluation
revision can measure native donor-relative quality on new sources without
regrading any prior numeric failure or relaxing the user's final quality/
same-artifact accepted>=50/useful large-n/multi-family requirements.
