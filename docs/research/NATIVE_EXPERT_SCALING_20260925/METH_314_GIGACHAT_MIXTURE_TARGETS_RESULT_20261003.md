# METH-314: complete mathematical source MoE target archives qualified

Freeze `76edd7a`, exit0, 69.500s CPU-only; maximum checked RSS 3297579008B, not OS sampled peak. All five gates PASS; BF16 rounding261120 cases/truncation97920 differences also pass. No source-context collection/GPU/model training. Fresh complete21.356GB BF16 source hash and2,301,494,016B of selected tensor payload hashes verified.

This makes available the COMPLETE source FFN function on actual source inputs: four selected original parent outputs, original normalized probabilities, shared output and their sum. Previous310/311 gates approximated each parent independently; that does not measure the loss of this composed function. Targets are source-faithful mathematical reconstructions with activation guards, not a compact model or captured full native MoE output bit-parity certificate.

## Actual datasets and coverage

| Layer/domain | Unique actual inputs | Source parents observed /64 | Smallest/largest parent count | Target archive bytes |
| --- | --- | --- | --- | --- |
| 1/fit | 8948 | 64 | 25/3504 | 385159524 |
| 1/test | 8318 | 64 | 41/5423 | 358041804 |
| 13/fit | 5042 | 64 | 33/2536 | 217029660 |
| 13/test | 5411 | 64 | 18/2873 | 232912896 |
| 25/fit | 7811 | 64 | 10/3118 | 336218496 |
| 25/test | 1851 | 63 | 0/845 | 79676256 |

21,801 fit +15,580 test =37,381 unique layer/input states;149,524 selected source-parent outputs. Archives total 1609038636B (1.609GB). All input states are the union of captured anchors0/32/63, deduplicated by layer/domain/chunk/batch/token with bit-exact shared-input checks. Every original captured row is retained. Input coverage is conditioned on those anchors; source tests/corpora were previously consumed. Counts expose rare parents (minimum fit10, one test parent absent in layer25); no640 learned functions, useful added capacity or adequate all-child exposure follows.

Six uncompressed NPZ archives under `results/native_expert_scaling/meth314_mixture_targets`, each containing F32 input, integer coordinates, four selected parent IDs, F32 gates, four unweighted parent outputs, shared output and full_mixture_output. Every array reloads bit exactly. Per-file hashes/schema/counts and64-parent frequencies are in [raw result](meth314_gigachat_mixture_targets_result.json); archive binaries remain local reproducible assets. Both domains are diagnostic, not final untouched quality data.

## Original-source controls and retained failures

- [312 failure](METH_312_GIGACHAT_MIXTURE_TARGETS_FAILURE_20261003.md): directF32 activations with decodedBF16 weights changed the source operator; original post-SwiGLU error0.300325%>0.01%.
- [313 partial/failure](METH_313_GIGACHAT_MIXTURE_TARGETS_FAILURE_20261003.md): originalBF16 activation packing lowers error to<=3.153e-7 in three completed cases; NumPy router then fails source slot replay. Preserve all three partial archives and exact failure.
-314 repeats sourceBF16 conversion/rounding controls and uses ORIGINAL GGML F32 router/sigmoid/correction/argsort/gather/clamp/normalize from pinned libraries. Every captured anchor matches its original SLOT across ALL six cases. Unknown token columns are zero-padded to original128-token batch width; known-column fidelity is verified by those slots.
- All38,526 original captured post-SwiGLU states match reconstructed nonlinear states, max relativeL2<=3.153e-7 versus SAME1e-4 limit. This confirms the source packing repair, without threshold/slot/source relaxation.
- Full weighted-plus-shared arithmetic closes at three fixed positions/case within1e-6; missing selected-parent contribution changes target as required. All target arrays finite, positive output norms and exact archive round trips.

Original CPP source SHA `42df3410eda5ae2b8b087d73fbc77be1fac193bb53fcf0a1373feac1355a31e4`, executable `ed11342ede5edbbf2ef07561ef2be8f1f5abc6d6facf0cdbcd07b82be9aaa0b2`; original library/header/compiler manifest pinned to298. Router replays total about1.002s, reference apparatus timing only. Sparse replay inputs/outputs add233,674,604B, recorded separately. Source-context re-collection was avoided; source hash timings are storage/cache dependent, no conversion speedup claim.

## Reproduce and next decision

Command `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth314_gigachat_mixture_targets.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth314_gigachat_mixture_targets_result.json`. Refuses existing destination/result; preserve312/313/314 assets. [Protocol](METH_314_GIGACHAT_MIXTURE_TARGETS_PROTOCOL_20261003.md), raw SHA `a46b9395df437442e113651c06d9ee8fc253c6731f9026deaf431a5fb01b30b5`, controller `186e08f6bb9660442e80b26cc3e33c40ae816076e2765d17bc3e486afbb1f0e7`, protocol `cdb3b98c64672d9d40794a746f90023b49ee1153942886f76556f040ba25bc12`.

Only separately frozen complete-mixture representation checks are licensed. Need quantify error of shared-plus-four-selected residual functions on this COMPLETE target and audit rare-parent/region coverage before a student investment. These inputs do not include full attention teacher contexts or all FFN layers. Existing CPU cost profiles still fail/are inconclusive. Untouched full-model prediction/generation/task quality, same-artifact>=50 accepted tokens/s, useful RAM-scale n and actual cross-family/100B transfer remain required.
