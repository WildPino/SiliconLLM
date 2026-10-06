# PQT-007: complete matrix conversion and whole-model behavior

Prospective, 6 October 2026. No PQT-007 full-model fitting or evaluation has
occurred. Previous screens covered one projection and two expert functions;
PQT-006 now qualifies same-artifact CPU arithmetic, with timing awaiting
owner coordination. Neither establishes complete-model language preservation.
This screen addresses that missing scope directly, without pretraining.

## Exact scope, reference and counted mixed representation

Use the same immutable Qwen2.5-0.5B checkpoint/revision/file identity and
qualified T4x2 runtime as PQT-001-R1. Original archived coefficients are
loaded into the F32 SDPA inference reference, TF32 disabled, no AMP/dropout.
Do not compare against an undocumented dtype or changed original model.
Frozen revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, archive 988,097,824
bytes, SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The initial prospective draft mistyped this digest; corrected against the
already qualified source binding before any PQT-007 model access or dispatch.

Convert **every unique 2D weight matrix**: token embedding/tied readout and
q/k/v/o plus gate/up/down matrices in every transformer block. Expected
architecture: 24 blocks, hidden 896, FFN 4864, vocabulary 151936, tied head;
169 unique matrices / 493,961,216 matrix coefficients. Runtime must verify
these expectations, alias identity, widths divisible by 64, and complete
coverage of named parameter tensors. Failure stops instead of silently
leaving unsupported matrices float.

Original norm/bias vectors remain explicit F32: expected 71,552 coefficients.
Total unique parameters 494,032,768; every parameter must belong to converted
matrix or counted float-vector category. Thus this is **complete matrix
conversion in a mixed model**, not an assertion that all scalar parameters
are pure ternary. Converted matrices account for ~99.9855% of parameters.
Count tied embedding/head once by actual shared identity and store both aliases.
Count norm/bias, scales, headers/manifests and all actual serialized bytes.

Four fixed arms: D, PR, I4, I8. Same source/matrix scope for every arm.
D/PR group-64 L2-optimal original-weight scales; D direct ternary ties to zero.
PR: source-input curvature compensation (damping .01 mean curvature diagonal,
block 128, original column order), then two discrete coordinate passes targeting
the original matrix's response on source calibration inputs, scales fixed.
This is source-guided matrix fitting, **not** full-network optimization or
compensation from intermediate student network validation.
I4: original direct group-64 symmetric [-7,7] control, existing five-factor
scale choice and nearest-even rounding. I8: original direct row maximum/127,
nearest-even [-127,127], zero-row handling and reserved -128 rejection.
No optimized low-bit baseline or pretraining claim.

Implementation detail fixed before scientific access: independent output-row
panels of at most 8192 rows limit temporary memory, with no row subsampling.
Each PR matrix uses one original-input curvature factor, shared across its
panels; each row follows the declared column/block and two-pass schedule.
All per-panel losses and row coverage are retained. Targets exclude unchanged
bias vectors. Loss-energy reductions use 64-position chunks with F64 sums.

Each arm is retained as one deterministic, uncompressed ZIP archive containing
all exact NPY matrix-code, scale and vector bytes plus its representation
descriptor. Count the actual complete archive size, including ZIP/NPY headers
and manifests, against the same FP16 reference. Audit rejects extra, duplicate,
missing or compressed members. Verified temporary member files are removed
only from the newly created owned run directories; raw bytes remain in the
archive. This keeps final output file count small for remote retention without
changing quantization, selecting weights or compressing codes further.

The tied embedding/readout has two roles. Use original final-hidden calibration
states for PR fitting of its readout role; both lookup and readout then share
the same exported weight. Do not silently keep the original embedding or
give the readout a separate uncounted matrix. This fitting proxy does not
guarantee lookup fidelity; whole-model evaluation tests the resulting coupling.

All matrix representations for all four arms must be frozen before news
selection/source evaluation/candidate evaluation or generation. No checkpoint
selection, matrix exemptions, calibration reweighting or threshold changes
based on final behavior. No bias/scale adaptation beyond declared fitting.

## Calibration and independent-of-fitting evaluation

WikiText train revision/file hashes remain PQT-001's immutable sources. Use
first sixteen contiguous 129-token windows, first 128 positions each, same
2,048 calibration positions. These are consumed fitting contexts. Capture
every original linear module input, plus original final hidden states for
the tied readout. Do not use news data, student evaluation states or source
validation outcomes for fitting. Original weights receive no gradients.

Capture inputs once from the unchanged original model. PR targets W_original*X;
unchanged linear biases cancel in the reconstruction residual. Retain token
bytes, matrix/input identities, shapes, raw calibration-input hashes, fitting
loss histories and costs. Large inputs may be regenerated rather than saved
only if separate audit replays the original model and verifies **every** input
hash exactly. Failure prohibits adjudication. Record shared input identity
without assuming q/k/v aliases are numerically identical before checking.

News source admitted separately before scientific source freeze:
`fancyzhx/ag_news`, revision `eb185aade064a813bc0b7f42de02595523103ca4`,
`data/test-00000-of-00001.parquet`, 1,234,829 bytes, SHA-256
`71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb`.
See [source binding](PQT_007_NEWS_SOURCE.json),
[retention](PQT_007_NEWS_RETENTION.json), and
[provider card](https://huggingface.co/datasets/fancyzhx/ag_news/blob/main/README.md).
Assert 7,600 rows and text field; classification labels are ignored.
No previous source reference found at `d55f188` in docs/benchmarks/scripts.
Declare independent of WikiText fitting and new to this research chain; no
claim of unseen donor pretraining or exhaustive parent-history novelty.

After all representations freeze, tokenize each news row with the pinned
original tokenizer, `add_special_tokens=False`, without normalization beyond
the provider text. CPU NumPy default_rng seed 20261017 permutes all 7600 IDs.
Take first sixteen rows in that permutation with >=129 tokens and unique
first-129 token bytes. First eight: prediction windows, first 129 tokens;
next eight: generation prompts, first 32 tokens. No overlap of row IDs or
selected token windows, no loss/category-based selection. If insufficient,
or if selected 32-token prompts duplicate each other or prediction prefixes,
stop and preserve the failure; do not lower the length filter after observing
behavior. Retain full permutation, selected IDs, token arrays/hashes and lengths.
Assert no selected first-128 prediction sequence or first-32 prompt occurs in
calibration token windows. The separate audit must reproduce this selection
exactly from source bytes. Rows become consumed after this experiment.

## Prediction, generation and frozen decisions

Prediction: all eight 128-position windows, 1024 labels from the next tokens.
Teacher-forced original contexts for source and complete converted models.
Compute each model's F32 logits with its own head, then FP64 log_softmax.
Do not reuse original readout for converted states. Save final hidden states,
per-token source/candidate top1, KL, NLL, labels, source reference energies and
scalar aggregates, plus per-window results. Main gates per arm:
top1 agreement >=99%, mean KL <=.01 nats, mean NLL delta <=.01 nats/token.
Retain ordinary label-based perplexity/NLL as diagnostic, not a classification
accuracy or proof of semantic/factual competence.

Generation: greedy, eight fixed news prompts, <=32 new tokens, no sampling,
same EOS/pad policy as prior screens. Each complete model uses its own changed
prefix/state and weights; no source-state substitution after the first step.
Retain prompt/full generated token IDs, every pre-choice hidden state, chosen
token, EOS/stop condition and execution identity. Position agreement uses
max(original,candidate) generated lengths; missing tail tokens disagree.
Gates: >=95% aggregate positional agreement and >=7/8 exact continuations.
These are behavior-preservation gates, not human-rated semantic quality.

Capacity: actual complete representation payload <=35% of full 988,065,536
FP16 reference bytes, not a ratio limited to just converted matrices. Count
all float vectors, scales and unique tied storage. Expected D/PR raw payload
154,649,088 bytes before serialized headers/manifest; I4/I8 compute and audit
their actual respective counts, never hide residual vectors or duplicates.
Fitting <=2700s total, experiment <=3300s; all arms reported, only final files
evaluated. Report GPU/process memory/elapsed costs, source acquisition/packing
separately. Any nonfinite values or budget/identity failure prohibit decisions.

A quality/storage pass would still require native deployment/usefulness and
broader validation before finishing the goal. PyTorch reconstructs F32 weights
from packed files for this behavior screen: **GPU resident memory and speed
are not the packed representation's execution cost**. No full native-transformer
kernel or integration claim from this test. PQT-006 has separate real-Switch
FFN artifacts/CPU scope; do not mix their numerical or cost conclusions.

## Independent audit and resources

Separate cuda:1 process, no fitting-module imports. Rehash original/data/source
and every final artifact, independently decode all matrices/vectors, reject
reserved codes/nonfinite/incorrect lengths/scales, reconstruct all aliases and
unique coefficient/storage counts. Verify source vectors exactly against the
original checkpoint and matrix coverage against independently loaded model.

Replay original calibration and reproduce all captured input hashes. Independently
check final local reconstruction metrics with FP64 accumulation and matrix-wise
finite tolerance <=1e-5 for F32/F64 numerical replay; do not mistake local
fidelity for final language quality. Reproduce news token bytes/selection.
Reconstruct original and each complete converted model; independently replay
all teacher-forced states and all greedy traces in F32 SDPA with TF32 disabled.
Require exact state/token replay against captured bytes and complete own-prefix
trace coverage. Recompute all F32 readout-derived point metrics/aggregates and
frozen gates; additionally check FP64 final readout numerical reconstruction
under relative RMS <=1e-5, reporting any argmax near-tie sensitivity separately.
Do not execute the original remote fitting script during audit.

acct3, fresh private `sirwildpino/pqt-007-full-model-20261006-001`, qualified
pinned T4x2 image; original tensor conversion on cuda:0, independent audit on
cuda:1. Source/bundle committed before live owned-kernel/quota admission and
dispatch. Install <=900s, experiment <=3300s, audit <=900s, server <=5400s.
No local GPU/full-model fitting, main environment changes or foreign process
interruption. Source acquisition/hash identity verified before use; all code
and inputs bound, first failures retained, never overwrite a run or retry an
unknown dispatch. Use a fresh owned PQT-007 namespace. Delegate long T4
monitoring to Luna and keep the coordinator waiting as requested.

Retain packed full-model files, selected tokens/states/traces, metadata/raw
points, calibration identities, logs and costs; small exact raw evidence in
the dossier, large immutable arrays outside Git. Update TERNARY_INDEX.md and
the full-scope matrix. Overall goal remains active; a failed method is scoped,
a passed numerical audit is not preserved capability, and a complete matrix
screen does not establish every requirement of a useful native procedure.
