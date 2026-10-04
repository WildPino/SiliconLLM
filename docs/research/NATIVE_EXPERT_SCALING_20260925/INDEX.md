# Native expert scaling: research control index

**4 October 2026. Branch `research/native-expert-scaling`. Goal ACTIVE/INCOMPLETE.**
Latest:435 shared nonlinear input fit fails3/4 geometry gates;436 correct
original function inputs with frozen output readouts still miss benefit/causality.
Next NEW437: one shared output readout versus equal-parameter own-input control;
oracle feasibility only, prepare/freeze exact qualification and fit protocol.

## Goal and constraints

Transfer pretrained knowledge/capacity to compact reusable core plus useful
selectively consulted functions in `benchmarks/phase60/engine.c`, preserving
explicit original-relative prediction/generation/task quality and>=50 accepted
batch1 tokens/s on SAME artifact (100 stretch). Multiple families/scales,
actual~10B/~100B as resources allow. User priority: useful n grows with RAM;
CPU LUT/routing/real DRAM and quality remain viable as choices grow. Distinguish
candidate IDs/stored distinct functions/active functions/actual read bytes.

Ryzen53600X/80GiB/RTX3060. Freeze scientific source/protocol BEFORE observations,
retain first failures before numbered repair, no modeljob overlaps native timing.
T4 requires prior reason/budget/stop communication. Routine Graphify disabled.
Preserve existing unrelated work and exact publisher daemons.

## Two research questions

| Question | Established | Open |
| --- | --- | --- |
| Useful conditional target |369 real-function bank matching;370/371 real64/128 subsets;373 fourfold bank cost bounded|372 mixed quality, no monotonic useful-n/10x gain; hierarchical LUT and physicalDRAM|
| Pretrained-to-target transfer |Original Switch7.415B/14.664B to7.542/14.818GB I8/A16 C artifacts, bounded whole-quality/source-specific FULL rate|Another family/~100B, broader contexts/tasks, useful new-function transfer|

## Qualified original artifacts

| Original source | Physical workers | Accepted ordinary IDs/s (lower95) | Prose IDs/s (lower95) |
| --- | --- | ---: | ---: |
| Switch128 /7.415B |3 [0,2,4]|96.56385 (94.58827)|55.97659 (54.59729)|
| Switch256 /14.664B |6 [0,2,4,6,8,10]|63.52543 (59.51178)|34.77928 (32.64971)|

[128 reproduction](SWITCH_BASE128_REPRODUCTION_20261004.md),
[256 reproduction](SWITCH_BASE256_REPRODUCTION_20261004.md):387/363 quality18/18,
391/376 SAMEartifact accepted FULL rate. Warm fixed IDs/context S29/T14 infilling,
encoder/crossKV/cached decode/head/argmax/stop included; startup/load/tokenization
excluded. No broad/cold/context-independent rate claim or new combined artifact.
Original374/389 binaries unchanged. Default engine/observation math unchanged.

## Latest decisive evidence and decisions

- 418 [capture](METH_418_SWITCH_FUNCTION_CAPTURE_RESULT_20261004.md):1936 archive
inventory,384 old complete outputs/routes/greedy exact,4895 final-bank states.
420 [whole forward](METH_420_SWITCH_FUNCTION_GRADIENT_RESULT_20261004.md):all
384/4895 native states and157266560 full-head rows exact.424 local approximate
saved-primal derivative qualified, not native rounding/hard-selection derivative.
- 426 [replacement pilot](METH_426_SWITCH_FUNCTION_PILOT_RESULT_20261004.md):6336
updates,7/9 capacity FAIL.431 [additive pilot](METH_431_SWITCH_ADDITIVE_PILOT_RESULT_20261004.md):
6240updates, ALL1344 zero-added predictions exact;7/9FAIL, no qualified useful
IDs, removal improves. Both exact recipes CLOSED before tuning or narrowed export.
- 432 [first failure](METH_432_SWITCH_OUTPUT_BOUND_RESULT_20261004.md):analytic
fixture passes, F32 CE comparator mismatch retainedc08299c.433
[output bound](METH_433_SWITCH_OUTPUT_BOUND_RESULT_20261004.md),74a649d:sole F64
comparator repair, ALL6PASS. Feasible mean256KL.02/worstbook.024565, output-only
mixture gain17.63%/128gain26.64%, numerical duality gap5.60e-14.55IDs have>=2val
positions before frozen gate, only2after. This oracle is not a model/transfer.
- 434 [frozen selection diagnosis](METH_434_SWITCH_FROZEN_SELECTION_RESULT_20261004.md),
47f3f43:zero updates, actual431 hard matrices/CEs exact. Prespecified oracle
with actual classifier route:53positions/12usefulIDs, mixturegain.4957%; teacher
IDs:61/10useful,.5162%. Both<1%; permutation harm.00493/.00282<.01. Neither
satisfies ALL9 potential gates. Close this checkpoint before new selector fit.
125.313s/max2.492GB/259.081MB full retained outputs, source hashes fresh.

- 435 [shared input](METH_435_SWITCH_SHARED_INPUT_RESULT_20261004.md),b970b34:
512updates/ALL5 apparatusPASS, validation error.60913mean-only/medianL2.87673/
p95.99102 =>3/4FAIL. Close map;28.843s/557MB/11.847MB retained.
- 436 [correct input oracle](METH_436_SWITCH_ORACLE_INPUT_RESULT_20261004.md),98a582f:
zero updates/ALL6 apparatusPASS, native source features exact. Classifier/teacher
mixture gain.17894%/.08674%; permutation improves. Neither8 diagnostic gates
passes, no matched-adapter capacity claim.99.921s/2.546GB/172.721MB retained.
Perfect input ALONE insufficient for frozen C/D; do not tune input map as sole change.

## Closed routes and retained context

Identity403, affine activation404/406, lexical414/416 bridges closed;405
conditioning/415 spectral diagnosis retained. Granite412/413/Ling408/410 and
specified Giga full-width/LUT CPU formats closed; no generic-port resumption.
QwenNext397 real79.674B main headers/top512 geometry active1.279GB>560MB,
no full values/quality acquired. Switch C2048 is a different~1.6T geometry,
not a verified matching base1024/2048 source. Details and all earlier decisions
in [historical index through431](RESEARCH_RECORDS_THROUGH_431_20261004.md),
[PRIOR_EVIDENCE](PRIOR_EVIDENCE.md), [METHOD](METHOD.md) and individual records.
Frozen donor-adaptation Giga work remains reusable evidence, not active old plan.

## Exact resumption

[Current plan](SWITCH_FUNCTION_RETARGETING_PILOT_NEXT_20261004.md):prepare ONE
NEW437 shared rank32 output-readout feasibility, real correct128 native feature
versus equal-parameter readout of its own128 input. All1008dev supervision shared,
source functions frozen; consumed336val and teacher input/IDs/masks oracle only.
Freeze independent composed numeric contract/exact fit/resources before outcomes.
ALL9 potential requirements before available-input/routing controls; no oracle
export. Whole quality/SAMEartifact50/usefulRAM-n/LUT/realDRAM/families/~100B open.
Original374/389 binaries/default engine exact, all sessions terminal.
