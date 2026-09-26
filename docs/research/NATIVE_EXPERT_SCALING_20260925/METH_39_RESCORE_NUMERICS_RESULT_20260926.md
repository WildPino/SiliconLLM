# METH-39: full-score oracle restores parity; batched rescore does not

The [prospective diagnostic](METH_39_RESCORE_NUMERICS_PROTOCOL_20260926.md)
reused METH-37's already inspected prompts and the same stored
rank-64 int8 sketch. The exact, C64, C96 and C128 elementwise
score arms reproduced METH-37/38 controls before the new
arithmetic was interpreted.

| Fine-score method | Candidate count | Top-4 set agreement on own trajectory | Next-token top-1 agreement vs exhaustive |
|---|---:|---:|---:|
| Elementwise multiply+sum | 64 | 99.747% | 6,043/6,144 = 98.356% |
| Elementwise multiply+sum | 96 | 99.994% | 6,104/6,144 = 99.349% |
| Elementwise multiply+sum | 128 | 100% | 6,123/6,144 = 99.658% |
| Batched matrix-vector | 96 | 99.994% | 6,093/6,144 = 99.170% |
| Batched matrix-vector | 128 | 100% | 6,109/6,144 = 99.430% |
| Full `F.linear` then gather | 128 | **100%** | **6,144/6,144 = 100%** |

The full-score oracle passes its exact parity control and
pinpoints fine-score arithmetic as the source of the residual
21 top-1 changes at C128 in METH-38. It scans all E128 fine
router rows and is not a bounded route. The candidate-only
batched calculation does not pass the prospective ≥99.99%
C128 numeric gate; it changes 35 top-1 outputs. Its C96
diagnostic cannot be promoted under the protocol. The
candidate-only elementwise C96 arm remains a possible *new*
quality candidate, with its numeric differences explicitly
accepted and tested on separate documents rather than hidden
by an oracle full scan.

The [machine result](meth39_rescore_numerics_result.json),
SHA-256
`8702866eb43d8169d7c1292f9a563a71711a5bf65936683e496873fc2da05875`,
contains per-layer route counts. Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth39_rescore_numerics.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth39_rescore_numerics_result.json`.
The local RTX 3060 audit took 95.05 s, peaked at 2.281 GB
allocated GPU memory and ended at 3.184 GB RSS. No T4 was used.

**Decision:** no arithmetic method here restores parity while
keeping bounded candidate rescoring. A further bounded-route
claim requires a new independent quality gate on the exact
candidate-only arithmetic and a subsequent C implementation.
