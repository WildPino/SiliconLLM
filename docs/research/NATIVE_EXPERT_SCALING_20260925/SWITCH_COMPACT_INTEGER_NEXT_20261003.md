# Next compact Switch representation: all-bank W8A8 cost preflight

CURRENT STATUS426 (supersedes historical next steps below): original Switch
source128/256 quality and SAMEartifact source-specific rates remain qualified.
Specified Granite/Ling/Giga cost formats and lexical/state-map bridges closed.
418 capture/420 whole native forward384/4895 states/full head exact;424 local
approximate derivative qualified.425 first numeric failure retained;426 complete
6336-update replacement pilot FAIL7/9 capacity gates, function removal helps.
Close THIS fixed recipe before sweep. Next [function-aware final-bank pilot](SWITCH_FUNCTION_RETARGETING_PILOT_NEXT_20261004.md):
NEW427 original-preserving additive readout; original+one extra function when ON.
No427 code/protocol/outcome or combined384 C model. Charge all additional active
cost; usefulRAM-n/LUT/DRAM/another-family/~100B remain open. History retained.


**IMPLEMENTED; NUMERIC AND SIX-THREAD COST-MARGIN QUALIFIED.**335/337 export/
recovery interruptions retained;338 ALL canonical source and target bytes PASS.
336 complete actual serialized integer C vs independent reference PASS EXACT.
339 physical engine serialization stop retained/default restored.340 full
cost FAIL prefill/repeatability;341 exact batched prefill PASS at unchanged
criteria, actual6-thread full7.643/17.510ms per position for9/64 source tokens
and32 forced decoder positions. Original donor quality/accepted rate open.

## Decision and reuse

Uncertainty: can a direct compact complete source path have enough actual CPU
margin to justify untouched quality/adaptation work? Reuse327 ALL actual tensor/
alias/12x256 parameter-tuple hashes and334 qualified architecture. Cost must
include actual full banks, selected reads, router scan/dispatch/decode, core,
full vocabulary head, prefill and growing attention/cache. A pass only licenses
quality work, never establishes accepted generation rate. Final same artifact
must retain original-donor quality and >=50 accepted batch1 tokens/s.

## Implemented target recipe, qualification pending

- Keep ALL original256 experts in each of12 banks; all original core layers,
  encoder/decoder/capacity/top1 gating and vocabulary retained. No fitting.
- Symmetric per-row signed I8 for every large matrix except source router;
  row scale F32(absmax/127), nearest-even clipped[-127,127]. Zero row: scale1,
  codes0. Record representability/finite checks, exact rounding rule and hashes.
  Embedding lookup/norm/relative bias/router metadata stay original F32.
  Shared source embedding/head can require separate lookup F32 and head I8
  views; count the extra bytes and shared original provenance explicitly.
- Activation vector quantized once per matrix call to symmetric I8 with one
  F32 scale, same declared rule. This is W8A8, not just weight-only compression.
  Its loss is unknown. I32 exact dot safe for sourcecols<=4096 and[-127,127]
  operands: worst bound66,064,384. Scaled output prescribed F64 scale product/
  integer scaling then rounded F32; residual/norm/attention contract334 retained.
- Native AVX2 sign-extend I8 to I16, VPMADDWD then I32 sum; avoid saturating
  intermediate I16 multiply-add. Threads1/6 declared, no ideal-bandwidth claims.
  Independent reference uses serialized target I8/scales, integer matrix sums
  and explicit scaling, not a second independently requantized checkpoint.
- Full target export streams bounded source shards/rows into actual disk
  payloads, all source/tensor/target hashes and alias metadata. Count distinct
  target WI/WO code-and-scale tuples in every bank; quantization cannot silently
  substitute identical tuples for genuinely distinct source functions. Actual complete payload14,818,015,744B
  including F32 embedding/controls; preserve partials. Final
  converter/native format/gates/download-free resource budget must be concrete
  and frozen before target observations. Existing source never overwritten.

## Qualification order and limits

First adversarial extreme/cancellation/zero-row exact integer and scale oracles,
Tiny full semantic faults/caps/cache, then full engineering consumed controls
versus matched integer reference. Preserve every failure before repair. No
comparison to original donor can be relabeled as exact under quantization.
Then real all-bank CPU measurements on stated source/output lengths, warmup/
repetitions/order/threads and actual routing/byte/cost counters frozen before
observations. Forced decoder assays are costs only, not accepted quality rate.
Source-routing controls should reflect target trajectories, with choice changes
versus original donor logged as diagnostics, not silently forced back.

If complete measured CPU cost lacks margin for goal50, close unchanged W8A8
before extensive quality/training; choose a new variable based on measured
bottleneck. If cost and implementation pass, define NEW source exclusions,
reconstruction/prediction/generation/tasks and limits BEFORE observation against
ORIGINAL unmodified official donor. Quantization may fail quality; no guarantee.

This is a compact source baseline enabling investigation of cheaper LUT book/
conditional representations. It is not a LUT scaling result by itself. Dynamic n loader is implemented336; only real256/Tiny2 qualified.
RAM-scale useful n, real base128 comparison and useful larger capacity require
separate actual source functions, routing and whole-quality measurements; source
base256 availability alone does not prove10x expert scaling or~100B transfer.

## Actual quality result and next precision variable

343 original-primary NEW prediction FAIL top1 94.791667% versus95%; all342
books consumed, unchanged W8A8 closed before generation/accepted rate.344
fixed-state head-only A16 counterfactual reduces55 to35 original-choice
differences using same weights/scales. Its independent oracles pass; this is
diagnostic evidence only. Implement HEAD A16/core A8 as a new execution
profile: exact AVX2 I8/I16 dot, I32 lanes bounded2,130,641,408 at4096 columns,
I64 global sum bounded17,045,131,264. Target14.818GB unchanged. Freeze numeric,
Tiny/nine faults, all96 consumed-state exact composition and unchanged actual
CPU cost gates BEFORE observations; then NEW source quality required. No job
live; generation draft unexecuted/unfrozen pending quality.343 failure retained.

## Current actual qualification / live resumption

349 NEW original prediction ALL gates PASS, original top196.7803%, known
accuracy loss-.78125pp within-2pp. Strict8-token exact0/96 BOTH remains weak;
old343 FAIL unchanged.352 consumed traces consult original238-256 encoder/
208-231 decoder experts/bank, no n-scaling/DRAM/causal usefulness proof.
350 NEW24books/96four-two-token-mask labels source29/target14,87 exclusions;
all now consumed351.353 actual greedy/cache/stop wrapper full numeric PASS,
unchanged345 forward/new353 binary, not engineering quality/rate promotion.
351 original-primary full task FAIL16/18 gates, retained30ca546; exec94685
exit0 fully consumed, ALL96 actual teacher/natural outputs. Masked-only top1
94.791667% versus95%, prose-edit upper.122823041 versus.10 FAIL. Original task
signal informative19.53125% exact-field/88.541667% healthy, native89.583333%;
known-answer noninferiority PASS. All350 sources consumed.354 draft ineligible,
unexecuted;349 scoped prediction/353 numerical controls remain valid.355
consumed-state/common-prefix head/upstream attribution PASS retained3dc671b.
356 ALL quantized inputs A16, SAME338 weights/scales/F32 controls/routers/
lookup, independent whole numeric/cache/natural/long ALL7 PASS retainedb9c2dab.
357 apparatus path stop retained23141ef;358/360CPU6 repeat FAIL retained,
359 actual core topology PASS[0,1]..[10,11].361 PRIMARYsingle-thread/one actual
core logical CPU0/processaffinity[0], SAME356 binary/338 payload, ALL6 cost
gates PASS retained7719f03; full8.1876/17.9727ms per32 forced position source
9/64, repeats1.0123/1.0285. Not accepted natural rate.362 NEW source-only
24books/96 four2-token-mask source29/target14,111 exclusions, ALL gates PASS
retainedc0302f3.
[363 whole quality](METH_363_SWITCH_ALL_A16_MULTI_SPAN_QUALITY_RESULT_20261004.md)
ALL18 PASS, original-primary unmodifiedF32CPU1/nativeALL A16CPU1/affinity[0],
ALL96 NEW source362 cases, retained3396ee2. Overall top197.2470%/masked96.4844%,
prose-edit upper.091532738<=.10; original/native health82/81 of96. Bounded
short English four-span task, not identical outputs/instruction chat. All362
sources consumed. [364 SAME full accepted rate](METH_364_SWITCH_SINGLE_CORE_ACCEPTED_RATE_RESULT_20261004.md)
FAIL retained93e7cac:45.1114 ordinary generated IDs/s INCLUDING markers,
book-bootstrap lower42.5362<50; prose-only24.6979/lower23.3052. ALL96 exact363,
81 accepted/895 ordinary IDs/490 prose, ALL96 full time19.8398s including
rejected cases; aggregate repeat1.005005 PASS. Encoder52.67%+crossKV7.44% of
whole time. SAME unchanged356 execution not promoted; no optional rerun.
[365 exact execution](METH_365_SWITCH_ENCODER_BATCHES_PROTOCOL_20261004.md)
NEW prospective encoder O/denseFF/F32router batches and four-token integer
weight-conversion reuse. Same338 payload/math/route order/capacity, new engine
opt-in binary. Freeze independent91 batch primitive/Tiny/nine-fault/full/
engineering/long controls AND ALL96 forced/natural full bytes EXACT363.
Execution-only exactness can inherit only scoped363 quality. Numeric PASS
licenses separately frozen366 SAMEbinaryCPU1 cost, then367 unchanged accepted
FULL >=50 lower/1.10 repeats/ALL96 acceptance, ordinary/prose explicit.
No job live before365 freeze. Useful n/LUT/physical DRAM/causal learned bank
usefulness/real128 comparison/cross-family/~100B goal remains open.
Source128 payload absent; old GigaChat evidence reused, generic port paused.
Default engine body BYTE EXACT. All prior quality/cost/apparatus failures remain.

## Current execution stop and next bank question

366 SAME365 actual CPU1 cost FAIL repeat1.101982>1.10 onsource64, retained
c0611e7/exec84095 exit0 fullyconsumed; source9/64 full8.2531/19.2906ms per32
forcedposition, all othergates PASS. Closeunchanged365 cost/ratepromotion;
367 dependency-ineligible/unexecuted/unfrozen.363 scopedwholequality remains
valid via365 complete byteequivalence,364 actualaccepted rateFAIL unchanged.
Next368 independent bank-identity control contract: SAME356 binary/math/
CPU1profile/payload, ONLY smallserialized manifest redirects WI/WO pairs via
fixedoffset1/127 bijections atactualn256, unchangedrouter/core/head. No new
engine/codegenneeded; fresh mappings/actualconsulted identities and complete
independent Tiny/full/natural controls before369 pairedconsumed quality.
This diagnostic tests matched learnedbank usefulness, notnewuntouchedquality
or everyexpert/n-gain. Fixedgenuine64/128/256 nested-bank intervention planned
nextforadditional real alternatives; actual n artifactsnotyetimplemented.

## Current resumption, 4 October: supersedes earlier proposed steps

Qualified original Switch128391/256376 whole-quality/rate intact.401 union
fits RAM but403/404/406 specified interfaces FAIL;405 full unregularized fit
ineligible. No384 selector/model. 410 froze75649d7/retained84e7610: new256+16/static pair-LUT/Q4 format ALL8
numeric/210SHA+70 direct full-archive gates PASS; three-worker32.238-33.425ms/
repeat1.0368 PASS, six-worker20.892-31.803ms/repeat1.5222 FAIL; both14ms cost FAIL.
Close THIS joint format before Ling source values/fitting.411 ca13080/363ec2b:
ALL6 actual Granite3.1 MoE header/config/index/tokenizer/module gates PASS.
1B-named source actual1.334625280B/24 banks32/top8; hypothetical full row-I8
descriptor433231872B/storage1444581376B eligible only for CPU-cost/original
operator contract.3B actual3.298788864B/32 banks40/top8 descriptor892412928B FAIL,
unchanged row-I8 closed before values. No tensor values/finite/function
uniqueness/original inference/native quality/rate/DRAM. A tractable another-family
case does not replace real~10B/~100B/useful>256/10x final scope. All jobs terminal.
Next GRANITE_MOE_FULL_I8_NEXT_20261004.md; no412 code/protocol/run.
Use INDEX.md/METHOD.md; preserve unrelated work.
