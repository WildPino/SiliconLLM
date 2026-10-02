# METH-299: half-width nonlinear block omission fails on GigaChat

**Decision: close the fixed residual-aware greedy32-channel block rule at
<=half width.** All24 fidelity/control gates fail across5/10/20 selected
blocks. Even20 of40 blocks leave about52% relative output L2 error on the
distinct domain. Do not train an input-only selector for this unchanged
omission rule or reinterpret the oracle's runtime as sparse decode speed.

## Frozen source and complete controls

[Protocol](METH_299_GIGACHAT_NONLINEAR_BLOCK_PROTOCOL_20261002.md) and
script frozen `c8195f7` before scores. Reuse298's verified BF16-source
post-SwiGLU inputs and original nine BF16 down projections:layers1/13/25,
experts0/32/63,all22,549 initial-domain and15,977 Cyrillic states.
All18 expert/domain rows and all per-state errors/coordinates are saved.
No new donor inference,GPU/T4/downloads or native rate benchmark.

Each full1280-channel output is calculated independently in FP64. Sum of
all40 original32-channel tile outputs matches it with maximum relative L2
**9.394e-17**, below1e-12 on every state. Original down tensor SHAs and both
capture hashes are reverified. Selector sees the full target output and
chooses maximum direct residual-norm decrease,lowest-ID ties,no duplicate
tile,no coefficient refit/rescaling. Static selector uses down-column
Frobenius norm alone. The selector is not a mathematical optimum over all
subsets; failure does not reject every subset or trained compensation.

## Actual original-channel output preservation

| Selected tiles / channels | Initial greedy median | Initial p95 | Cyrillic greedy median | Cyrillic p95 | Cyrillic static median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5/40 /160 | 82.193% | 88.267% | 82.680% | 87.547% | 93.846% |
| 10/40 /320 | 70.479% | 78.866% | 71.217% | 77.283% | 86.908% |
| **20/40 /640** | **50.872%** | **61.402%** | **51.787%** | **58.681%** | **70.759%** |

Every domain pooled median<=1%/p95<=5% gate fails at all three counts;
every-expert gates fail too. Greedy improves on static but its pooled
median is not <=half static on either domain at any count. All24 gates
are false; no count qualifies. Independent stdlib reaggregation from saved
per-state errors confirms pooled values and all gates.

This direct nonlinear omission fails on BOTH domains, so changing
calibration language alone cannot repair it. GigaChat's measured block
outcome is distinct from185's individual-channel dense-Qwen rule; neither
proves that trained cores or another channel partition cannot work.

## Cost, reproduction and limits

Session33375 terminal exit0,48.718s/endRSS723,890,176B. Scientific FAIL is
separate from apparatus completion.15min/12GiB/50MB limits pass;raw result
is11,381,053bytes. Command is in the protocol. Both calibration domains
are consumed; this is not full-model heldout/task/generation evidence.

[Raw result](meth299_gigachat_block_selection_result.json) SHA
`ec4d1ea0c20a60b32b3563d0d31a160f14e9add5133030fcf29c3a9e3cf42327`.
No runtime source/archived helper was changed. Original297 semantic,
298 rank and185 channel stops remain recorded.

Forty tiles partition each of the64 actual pretrained experts into2560
labels/layer using the SAME original parameters. This is no new learned
capacity/useful tenfold n result. The oracle computes ALL original
activations/down tiles;it saves no gate/up bytes. Hypothetically halving
routed Q4 payload with all other organs unchanged still leaves838.141MB/
token,above560MB allotment. Actual packing/DRAM/router/LUT cost is unmeasured.

## Next representation boundary

Global linear rank192 and direct half-width block omission both lose
substantial real source output. A next compact source operator should
preserve all original nonlinear feature channels, changing how source
coefficients and repeated products are represented. The
[full-feature shared-vector-LUT proposal](FULL_FEATURE_VECTOR_LUT_PROPOSAL_20261002.md)
is a NEW unmeasured hypothesis:small non-Cartesian two-weight palettes
shared across source expert matrices,explicit codes/row scales,and input
dot-product tables independent of expert count. Price every table build,
lookup,bank read and other model organ before training/export work.
It cannot inherit old Q15 factor-LUT passes or synthesize useful capacity.
