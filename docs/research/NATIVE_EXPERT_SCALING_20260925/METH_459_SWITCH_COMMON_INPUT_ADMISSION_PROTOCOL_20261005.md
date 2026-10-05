# METH459: exact common-input A16 fanout admission, no model execution

5 October2026. ONE prospective model-free admission. Freeze controller/protocol/
attributes before first import/observation. No native source/compile/timing/new
weights/fit/GPU/download/engine edit. Goal ACTIVE/INCOMPLETE.

## Uncertainty, existing evidence and decision

458 measured sparse expert matrix fractions24.8585/27.0944%, dense/control
44.0788/40.0130%. A new WI-local tweak alone has bounded whole effect. Which exact
common-input shared-core operation is missing in the QUALIFIED original paths?
Verify QKV/KV fanout legality, geometry, precision/output order and removable work,
including all rejected cases. Distinguish already implemented generic QKV/token
batching from this specific row-I8/A16 sharing contract before another kernel.

An admission PASS permits proposing ONE separately frozen native whole inquiry;
it proves no time benefit or new whole model. Any missing invariant/duplicate
implementation retains first failure/inconclusive result; no automatic retry,
further shapes or kernel tuning. Counts are not milliseconds. No global semantic
absence/novelty claim from a name search.

## Inputs and binding scope

Fresh SHA/physical HEAD for controller/protocol, full458 raw (SHA3762fd4e...)
and retention (SHA65ec6408...), BOTH374/388 original sources and engine matching
458's hashes. Source388 reverses its names/worker-admission exactly to374.
No old scientific main import. All8apparatus/BOTH profiler admissions/ALL7retention
gates must have passed. Their generation quality/timing evidence is INHERITED,
not rerun or newly hashed. Raw attributes must equal raw committed HEAD.

Fresh BOTH binary tensor manifest SHAs matching original artifacts. Parse
SWI8A001 independently using public struct bytes, bounded names/records/EOF,
one file, every tensor shape/encoding/offset/scale bounds and unique sorted names.
Payload path/size/mtime must equal458. No source values read or full payload SHA
refreshed in this MODEL-FREE experiment; this limitation is recorded explicitly.
No fresh compiled numerical equivalence claimed from metadata.

Catalog physically committed current C/H files in native_expert_scaling plus
phase60/engine.c, record SHA/size and bounded name-search hits for fanout/qkv/
mv_K/integer_dot4. Qualified exact QKV/KV separate callsite patterns and mv/mv_batch
quantizer bodies checked independently; current generic fp32 SWA jointQKV and
layer-major/token4 integer batching must be acknowledged. They do not establish
row-I8/A16 common-input fanout implemented in374/388. Catalog includes tracked
current code only, not unrelated untracked or every historic branch.

## Exact sharing algebra and invariants

Given identical x, length, absmax/RNE A16 rules and FE_TONEAREST:

    codes=Q_A16(x); alpha=scale_A16(x)
    y_m[r]=F32((F64(dot_I64(W_m[r],codes))*F64(row_scale_m[r]))*F64(alpha))

Separate integer sums/row scales and original floating product order remain
unchanged. One code vector may feed m=2/3 independent projections. Query codes
are constant during the parallel fanout. Original outputs are disjoint and must
not alias x/codes; respect token-major/cache original strides. No nonlinear/norm/
softmax/route/scale order changes. The equality is an algebraic rule requiring
native primal verification of the actual implementation in a later experiment.

Legal original sites, checked actual manifest I8 matrices[D768,D768]:

* each encoder layer self Q/K/V, SAME selfnorm x and source29 batch;
* each decoder layer cross K/V cache preparation, SAME final encoder vector;
* each decoder layer self Q/K/V per generated position, SAME selfnorm x.

Cross-attention Q uses a DIFFERENT normalized vector after self residual; never
share it with selfQ. Output O input differs from Q/K/V. Dense/expert WI and WO
have nonlinear/A16 boundaries; their quantizers cannot be merged by this rule.
No head/router/expert precision change. Dimensions/layers/sparse-step validated
from each actual manifest; distinct selected matrix offsets/scales preserved.

Original integer dot AVX2 uses8 I32 lanes and final I64 reduction, cols<=4096,
I8 including-128, A16 excluding-32768. Freeze universal bounds:

    pair <=2*128*32767
    per lane <=(4096/8)*128*32767 <2^31
    final dot <=4096*128*32767 <2^63

No saturating pair primitive/new integer blocking is introduced here. Source
qualified768 shape bound also recorded. Fanout does not increase an individual
sum bound. Sharing across different x/quantizers/row scales is not licensed.

## Frozen operation accounting

Use ALL96 cases/source in existing458, with original S and actual generated T
(including nonaccepted cases). ALL8 warm/measured rows/case must have SAME S/T,
and kind0 dense logical calls/code/scales/F32 bytes independently rederived from
actual enc/dec/dense-FFN layers. Kind0 combines denseFFN and attention/control;
no quantizer time is available. Do not equate code bytes with physicalDRAM.

Let Le,Ld be encoder/decoder layer counts and Ae,Ad their dense-FFN layer counts:

    encoder+crossKV dense calls=(4Le+2Ae+2Ld)*S
    decoder dense calls=(6Ld+2Ad)*T
    encoder+crossKV code bytes=((4Le+2Ld)*D^2+2Ae*D*F)*S
    decoder code bytes=(6Ld*D^2+2Ad*D*F)*T
    scales: sum each original output row count*4*query count

Each mv_batch call creates ONE OpenMP region but quantizes S different query
vectors; native logical calls count S. Do not confuse these counts.

| Fanout | Groups | Queries/group | Maps/group |
| --- | ---: | ---: | ---: |
| Encoder selfQKV |Le|S|3|
| CrossKV |Ld|S|2|
| Decoder selfQKV |Ld*T|1|3|

Quantizer evaluations removed=2Le*S+Ld*S+2Ld*T.
Parallel regions removed=2Le+Ld+2Ld*T. Sum all96 cases once, not warm/repeat copies;
also report candidate before/after totals and fraction of ALL dense evaluations/
regions. Integer dot products/weight code bytes/scales/storage are unchanged.
No whole speed estimate from operation ratios or dense cost fraction. Preserve
inherited total-time scope/all rejected generation work in operation counts.

## Gates, resources and record

Eight admission gates: fresh frozen sources/engine/manifest/parent metadata;
qualified callsite/quantizer exact; all36 legal fanouts/source with shapes/I8/
unique offsets; ALL1536 inherited kind0 counter rows independently exact;
all cases including rejected charged; integer extreme bounds; existing paths
distinguished; zero native/model work. All must PASS for the one proposed inquiry.
No performance/storage/quality gate is passed by admission alone.

Hard main<=60s excluding imports, process peak working set<=256MiB; guard during
hash/catalog/case loops. Small JSON metadata only, no results/native output dir.
No competing model job. Actual own live ancestors excluded; preserve only exact
publisher allowance pythonw.exe/2argv/argv1 resolving to
D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py. Fresh before/after audit.
UTF8noBOM source/protocolLF; raw/failure JSONCRLF. Exclusive-create raw and first
failure, never overwrite or silently repair after observation. Source immutable.

After freeze run ONCE:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth459_switch_common_input_admission.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth459_switch_common_input_admission_result.json
```

After completion derive report/metadata retention only. Next native implementation
requires its own prospective quality/whole-cost/resource/provenance gates.
One bounded shared-core decision cycle; no indefinite optimization sequence.
Useful capacity, fresh complete transfer quality/SAME50, router mass/physicalDRAM,
other families/~100B remain the final requirements.
