# PQT-009: calibration breadth at matched adaptation budget

Prospective, 6 October 2026. PQT-008 is complete and retained; its final
calibration agreement near 69% does not transfer to new news contexts (<4%).
This experiment tests a distinct controlled breadth hypothesis. No new
PQT-009 source-model function, corpus selection, fitting or dispatch precedes
this protocol. Prior numerical outcomes remain consumed evidence.

## Question and immutable reference

Does broader calibration improve independent complete-model fidelity with
the same applied-update and position-exposure budgets? Compare one-shot and
staged conversion within each breadth and retain separate same-domain and
news evaluations. Data coverage and the window sequence intentionally differ;
this is not a matched numerical optimizer trajectory or a universal remedy.

Original Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, safetensors 988,097,824 bytes,
SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
WikiText-2-raw revision `b08601e04326c79dfdd32d625aee71d232d685c3`:
train parquet 6,357,543 bytes SHA-256
`e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`;
test parquet 732,610 bytes SHA-256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
Pinned tokenizer files are verified against the qualified original bindings.
AG News revision `eb185aade064a813bc0b7f42de02595523103ca4`,
`data/test-00000-of-00001.parquet`, 1,234,829 bytes SHA-256
`71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb`.

Use the qualified pinned T4x2 image/runtime, F32 SDPA, deterministic operations,
seed 20261005 for runtime initialization, no TF32/AMP/dropout. Frozen source
teacher on cuda:1, student on cuda:0. Separate audit subsequently uses cuda:1.
All 169 unique matrices / 493,961,216 coefficients finally ternary, shared
embedding/readout counted once. Original 71,552 scalar vector coefficients
remain frozen F32 and are counted. No matrix exception, learned scale, extra
readout, deployed latent float matrix or optimizer buffer. No pretraining.

## Five final arms and exact fitting contract

D is direct original-weight group-64 ternary. N_ONE/N_STAGED are narrow
calibration arms; B_ONE/B_STAGED are broad. Every trained arm starts independently
at original F32 matrix weights and a fresh Adam optimizer. Exactly 256 applied
updates on every unique matrix; actual counters must equal the applied step
after every update, final counters all 256. No scalar-vector gradients/updates.

ONE converts all matrices before step 1. STAGED converts shared embedding/readout
first, then blocks 0..23, all seven matrices together, eight updates per stage.
Steps 193..256 have every matrix hard, giving 64 fully-hard updates versus
ONE's 256. All unconverted staged matrices remain trainable F32 until their
stage. Reuse PQT-008 fixed original L2-optimal group-64 scales, hard ties to zero
at +/-scale/2, exact hard forward code*scale and identity weight surrogate
gradient with no scale gradient. Adam lr .0002, betas (.9,.999), epsilon 1e-8,
weight decay 0, foreach/fused false; clip global gradient norm to 1.0 with
nonfinite error. Do not select checkpoints, search learning rates or add steps.

Tokenize each WikiText split as the newline-joined complete text column using
the pinned tokenizer, no special tokens. Partition train tokens into consecutive
nonoverlapping 129-token complete windows, discard incomplete trailing tokens.
Narrow calibration is windows 0..15, exactly PQT-008's 2,064 token bytes.
Broad contains those sixteen first, followed by 240 spread windows. For complete
train-window count M, require M>=256. Divide indices [16,M) into 240 integer
strata with boundaries 16+floor((M-16)*i/240), i=0..240. NumPy default_rng seed
20261023 chooses one integer uniformly in each half-open stratum. Preserve
all window indices, tokens, raw hashes, boundaries and complete token count.
Require 256 distinct full-window and first-32-prefix hashes; a duplicate or
insufficient stratum fails rather than changing selection. Broad first sixteen
tokens exactly equal narrow. These are calibration, not validation identities.

Narrow update order exactly PQT-008: NumPy default_rng 20261019, concatenate
sixteen independent permutations of 0..15. Broad order: independent NumPy
default_rng 20261023 permutation of 0..255. Same order within each ONE/STAGED
pair. Each update consumes first 128 predictive states in a whole 129-token
window. Narrow: 2,048 unique positions repeated sixteen times. Broad: 32,768
unique position instances once, not necessarily 32,768 distinct token types.
All four arms have 32,768 position exposures and 256 actual updates.

Teacher and student each use their own corresponding head. Mean KL(P_original ||
P_student), temperature 1, F32 logits cast to F64 log_softmax, averaged over 128
positions. Teacher is frozen and receives no gradient. Generate teacher targets
per window rather than retaining a 20 GB vocabulary-logit cache. Retain source
calibration states and an exact NPY-byte hash ledger for all 256 original logits;
rehash each actual update target against its ledger. Independent audit rebuilds
every target. For the first sixteen, require exact original PQT-008 retained
teacher-logit identities. Bound transient targets to one window per device.

Before any new evaluation, D and repeated narrow arms must reproduce PQT-008's
packed-code raw hashes for all 169 matrices, fixed scale/vector identities and
complete raw parameter byte count. Compare N_ONE to ONE and N_STAGED to STAGED.
These are frozen predecessor bindings in PQT_009_CONSUMED_BINDINGS.json;
archive descriptors/names may differ, numerical bytes must match. Reproduction
failure invalidates apparatus, preserves first failure and prevents evaluation;
do not attribute it to breadth. Broad hard changes relative to D must be real
and separately recorded. All five payloads freeze before new evaluation.

## Fresh selection, two domains and immutable decisions

Bind exact fetched predecessor records before selection. News excludes all
32 PQT-007/008 row IDs and their reconstructed first-32-token hashes; verify
each old 129-token hash against its original provider row. Fresh news order
NumPy default_rng 20261023 permutation of 7,600 IDs; take first sixteen eligible
rows with >=129 tokens, distinct first-129 and first-32 hashes, not consumed.
First eight predict 128 positions using 129 tokens, next eight yield 32-token
prompts. No loss/label filtering. Same provider split, not untouched holdout.

WikiText test order: independent NumPy default_rng 20261023 permutation of its
complete 129-token windows. Bind all previously consumed PQT-001-R1/002/003/004
prediction and prompt token files, retaining their token arrays and full/first-32
hashes. Exclude every consumed prefix/hash and every broad-calibration prefix.
Take first sixteen distinct eligible test windows, first eight predictions,
next eight 32-token prompts. No source/candidate loss filtering. This is fresh
within the recorded research chain, not project-wide untouched validation.

For both domains require disjoint prediction/prompt full windows and first-32
prefixes, no reused consumed sequence, and no selected 128-token prediction
sequence or 32-token prompt anywhere in flattened broad calibration. Also
require distinct new prefixes across the two domains. Prefix/corpus collisions
exclude a candidate according to these fixed rules; insufficient eligible
windows fails without relaxing rules. Preserve full permutation, accepted IDs,
full token lengths/hashes and reasons for exclusions. No evaluation identity
or output enters adaptation, checkpoint selection or hyperparameter choice.

Absolute gates applied **separately in both domains**, each with 1,024 prediction
positions/eight own-prefix greedy continuations <=32 new tokens: >=99% source
argmax agreement, mean KL<=.01, NLL delta<=.01 nats/token, >=95% generation
position agreement (unmatched tails mismatches), >=7/8 exact continuations.
Complete stored archive <=35% of full 988,065,536 FP16 parameter bytes, combined
four-arm adaptation/export <=7,200 s. Every domain and gate required for promotion.
No pooled score can hide a failed domain. Perplexity is diagnostic, not task
semantics or factual capability. Existing PQT-008 gates/results stay unchanged.

Primary relative breadth signal separately for ONE and STAGED: broad held-out
mean KL <=.9*narrow in **each** domain, NLL delta no worse, argmax no worse,
generation-position agreement no worse, identical raw parameter bytes.
Report each domain/criterion even if the joint signal fails. Secondary relative
staging signal within broad: B_STAGED versus B_ONE with those same >=10% KL/
no-worse conditions in both domains. Relative signals never replace absolute
promotion. No retrospective gate adjustment based on results.

Retain final hard wrapper/exported-state equality on each arm's own calibration
set, final frozen fitted-set metrics/points separately from online pre-update
losses, every applied loss/norm/stage/counter and final archive identity.
Calibrated sets differ by design; compare new-domain behavior on common
evaluation contexts, not final fitted-set metrics with different sample counts.

## Qualification, audit, costs and stops

Synthetic-only controls before original/new numerical access: inherited hard
forward/identity surrogate/true KL derivative/shared lookup-readout gradient,
actual optimizer counter and code change, causal SDPA backward and all 169
toy bindings; new strata/subset/schedule selection, exclusions/collisions,
per-window target NPY hashes, new named archive/read contexts, finite and
malformed rejection. No local full-model loading/fitting, GPU use or timing.

Independent process imports no fitting or selection module. Verify actual
embedded committed source, runtime/input/artifact hashes, original parameter
coverage and tied aliases, fixed scales/vectors, every deployed matrix member
and final raw/full archive bytes. Independently reconstruct both calibration
and evaluation selections, every teacher target/state, schedules and actual
counter/stage histories. Explicitly do not claim a full optimizer trajectory
replay. Recheck narrow predecessor numerical reproduction and actual broad
hard changes. Replay source and each complete candidate's fitted-set/evaluation
states and every own-prefix generation state/choice exactly, derive all metrics/
absolute/relative gates. Separate F32/F64 readout diagnostics <=1e-5, record
argmax near-tie differences. No audit pass implies quality preservation.
F32-derived point metrics must reproduce exactly for every fitted-set and
evaluation position. F64 matmul diagnostics cover every new evaluation position;
for each final fitted set they cover exactly 128 evenly spread positions,
indices floor(i*(n-1)/127), i=0..127. Report sampled calibration maxima and
coverage explicitly; do not claim an all-position F64 bound from that sample.
This diagnostic subset bounds audit cost without sampling the quality metrics.

Report both GPU peaks, RSS, target generation, original grid/D export,
adaptation/arm costs and every process phase. Fixed stops: four-arm adaptation
and export <=7,200 s; experiment <=7,800 s; audit <=1,200 s; install <=900 s;
server <=10,800 s. Exactly 256 steps per trained arm, never silent truncation.
OOM, missing gradient/counter, nonfinite value, hash/reproduction or stop failure
invalidates this run; preserve first evidence and number repairs. Deployed ZIP
archives uncompressed, headers/metadata counted; no full float weight cache
in the payload. GPU reconstructed F32 execution is not CPU native latency.

Fresh private acct3 `sirwildpino/pqt-009-calibration-breadth-20261006-001`, pinned
qualified image, fresh live quota/owned-session admission after committed source
and bundle. One dispatch only. Delegate exact-handle long monitoring and wait;
single terminal fetch/retention, require passed audit before adjudication.
Preserve large immutable artifacts outside Git and exact small raw evidence in
the dossier. Update TERNARY_INDEX.md at every decisive checkpoint. Overall goal
remains active: broader capability, original expert preservation and native
useful capacity require their own evidence. CPU timing still needs the explicit
owner reservation independently of this remote method work.
