# PQT-004: calibration coverage at a fixed fitting budget

Prospective, 5 October 2026. No PQT-004 numerical observation. PQT-003 failed
preservation despite limited head-fidelity gains; its online calibration losses
were much smaller than held-out losses. A coverage hypothesis needs a controlled
test before longer fitting or larger representations are selected.

## Question, arms and invariant budget

Does spreading calibration across the corpus improve source preservation with
the same 2048 token positions, 256 actual head updates and representation?
Five arms, all reported: PR-B-C, HK-C, PR-B-S, HK-S, I4. C uses the original
first sixteen 129-token train windows; S uses sixteen stratified train windows.
PR-B is unchanged progressive compensation, two coordinate passes and mean
calibration offset. HK initializes from its corresponding PR-B and uses PQT-003
pure head KL fitting, same Adam rates, clipping, constraints and hard ternary
forward. No margin arm or evaluation-selected hyperparameter. I4 is unchanged
direct original-weight quantization, independent of calibration.

Immutable Qwen/WikiText identities, original FP32 source values, last projection
[896,4864], frozen 494,032,768-parameter core/readout, RMSNorm surrogate and
runtime pins are inherited from PQT-003 and verified again. This remains a
single-projection diagnostic. It does not replace original-expert/native tests.

## Content-independent data selection

Tokenize complete newline-joined train/test rows with the pinned tokenizer,
no special tokens. Complete window count N=floor(total_tokens/129); ignore tail.
All IDs are derived before model fitting, without observing text content or
source/candidate errors. Retain full token counts, window IDs, selected token
arrays and exact parquet paths for independent reconstruction.

C: train window IDs 0..15, exactly the existing calibration. S: partition train
IDs [16,N) into sixteen strata, bounds low=16+floor((N-16)*i/16),
high=16+floor((N-16)*(i+1)/16). Select low + digest mod(high-low), where digest
is the unsigned big-endian integer from SHA256 of ASCII
`PQT-004:20261005:calibration:i:0`, with i replaced by stratum index 0..15.
Sixteen windows, first 128 positions per window: 2048 positions in either arm.
Ascending stratum order; no token count or sequence-length difference.

Evaluation: partition test IDs [40,N) into eight strata, analogous bounds
using minimum 40 and count 8. First ID in each stratum uses role `evaluation`
and slot 0 in the same hash text. It supplies all 128 next-token positions,
1024 total. Generation ID uses slot 1: draw low + digest mod(high-low-1), then
increment if >= the first ID. This selects from the ordered stratum with the
evaluation ID removed. Prompt is its first 32 tokens; <=32 greedy continuation
tokens, own prefixes/cache, EOS stop. Distinct evaluation/generation windows;
all previously consumed test/prompt IDs 0..39 excluded. The actual new IDs must
be added to the consumed-context ledger after the run; they are not contiguous.
Public WikiText is not fresh project-wide or pretraining-independent evidence.

Head schedule: same original CPU generator seed 20261005, eight permutations
of 2048 positions, 32 batches of 64 per epoch. Identical positional schedule
for C/S, but different selected source activations. Check each parameter's
Adam state step=256, finiteness, actual hard code changes and frozen-core
gradient absence. Infer only packed codes, positive group-64 FP32 scales and
896-value FP32 bias. Ternary payload 1,365,504 bytes; I4 2,451,456 bytes.

## Metrics, prospective decisions and audit

All-position layer SSE, full-vocabulary source-to-candidate KL, next-token NLL
increase, source argmax agreement; per-window and raw position metrics.
Changed-state generation positional and exact-trace agreement as PQT-003.
Same absolute promotion gates: >=99% source argmax agreement, KL <=.01,
NLL increase <=.01; generation >=95% agreement and >=7/8 exact traces;
payload <=35% of projection FP16 bytes; total fitting operations <=2700s.
Report total process cost including capture, final calibration checks and audit.
Measured fit excludes source capture, serialization and metric calculation;
these costs are included in the bounded experiment elapsed time.

Separate coverage signal, **not promotion**: for a matched method S versus C,
held-out KL decreases >=10% relative, argmax agreement does not decrease and
NLL increase is no more than .01 worse. Report PR-B and HK comparisons even if
one passes and the other fails. Generation is reported without relaxing any
absolute gate. Compare all arms with I4 on the same new contexts. Do not infer
cross-run gains from differently sampled evaluation sets.

Report final frozen calibration KL and source argmax agreement on all 2048
positions for every ternary arm, plus states and raw points, before evaluation.
These are fitted-set diagnostics, not independent validation. They resolve the
previous online-versus-final ambiguity and cannot select a checkpoint or budget.
Freeze all final representations before any evaluation outcome is calculated.

Controls before donor access: existing scale/head/math controls, plus selection
bounds, exclusion, disjoint paired windows and undersized-input failure.
Independent cuda:1 audit imports no fitting modules. Rehash and retokenize the
pinned corpus, independently derive exact hash-based window IDs, reconstruct
selected arrays. Recheck complete calibration permutations/update counts.
Retain decoded projection/readout checks and full nonlinear replay of all
evaluation contexts and greedy traces. Replay actual full-model calibration
states and independent FP64 RMSNorm for C/S; reconstruct source S projection.
Recompute all frozen calibration readouts in FP64: KL tolerance 2e-5; argmax
disagreement allowed only at a <=1e-4 top-two gap. Exact metric reaggregation.
Other tolerances remain PQT-003's 1e-5 reconstruction / 2e-5 readout limits.
Failed audit prevents adjudication; retain first failure rather than relax gates.

## Resources, freeze and retention

acct1, fresh private `wildpino/pqt-004-calibration-coverage-20261005-001`, qualified
pinned T4x2 image/runtime, cuda:0 fitting/evaluation and separate cuda:1 audit.
Seed 20261005, deterministic algorithms, TF32 disabled, no AMP. Each head arm
<=1200s, total fit <=2700s, install <=900s, experiment <=3300s, audit <=900s,
server <=5400s. Public pinned acquisition only. Live owner/quota/full owned-job
inventory admission before push. No local GPU/native jobs or main environment
mutation. Source/protocol commit, byte-verified bundle generation and commit
before dispatch. Retain complete raw logs/arrays/states/codes and actual server
source, hash-check terminal retrieval once, commit small raw evidence, update
TERNARY_INDEX.md. Number repairs; never overwrite prior source or run evidence.
