# METH-46: learned routing is useful, but strongly imbalanced

The [predeclared diagnostic](METH_46_ROUTE_UTILITY_DIAGNOSTIC_PROTOCOL_20260927.md)
used only the METH-15 raw heldout slice and METH-44 same-corpus development
prompts. It did not open METH-45's frozen external documents, generation
set or PIQA outcomes. The [measured result](meth46_route_utility_diagnostic_result.json),
SHA-256 `4fd3471bab4421968ec34a701b032a5313230508ae20da077bbeea4b33d1ffc7`,
binds both METH-45 checkpoint hashes. The local RTX 3060 run took 8.844 s,
with 1.747 GB peak allocated GPU and 3.306 GB final process RSS.

| Update | Intact raw BPB | Permuted router BPB | Permuted minus intact | Minimum used experts per raw layer | Worst raw load/mean | Minimum raw normalized entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.946198 | 0.966528 | **+0.020330** | 102/128 | 24.56× | 0.736 |
| 512 | 0.942302 | 0.965080 | **+0.022778** | 103/128 | 25.66× | 0.721 |

On the viewed chat prompts, minimum used experts per layer were 120/128
at update 256 and 119/128 at 512. All 128 output factors in every layer
were nonzero and more than one tenth of their layer's median factor norm.
The worst raw layer's most selected expert received 19.19% of its top-4
selections at update 256 and 20.04% at 512, versus 0.781% for uniform
selection. This is load concentration, even though most slots are touched.
The script restored each router after its permutation and verified the
original BPB returned exactly.

**Decision: retain this conditional geometry for a retention repair.**
The update-256 sensitivity exceeds the frozen +0.002 BPB threshold by
an order of magnitude while that checkpoint still passes the 95% chat
development gate. This demonstrates that the learned association between
router rows and residual factors matters on the raw slice. It does not
prove every selected expert adds useful capacity, nor that E1280 or an
order-100B bank would remain balanced or improve quality. Any distinct-E
ladder must measure load, expert exposure, quality and CPU shortlist cost
together. METH-45's chat-retention failure still blocks this checkpoint
from external evaluation and native promotion.
