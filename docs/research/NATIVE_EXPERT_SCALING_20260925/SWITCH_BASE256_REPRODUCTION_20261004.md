# Qualified base256 route: reproduction and applicability boundaries

## What can currently be reproduced

Original immutable Google `switch-base-256` revision
`cdac1724c078ea4974b4087c59634799561e7835`,14.664B unique pretrained parameters,
to14.818GB row-I8 weights/F32 controls plus A16 projection inputs, complete
native phase60 encoder/decoder/greedy/cache.363 original-primary quality ALL18
PASS; exact374 six physical-worker execution;375 cost PASS;376 accepted full
model rate63.5254 ordinary IDs/s/lower95 59.5118 on SAME artifact. Includes
structural markers; prose-only34.7793/lower32.6497. No universal recipe claim.

## Immutable local artifacts

| Artifact | Path / SHA256 |
| --- | --- |
| Original archives and original tokenizer/config | `results/native_expert_scaling/meth326_switch_base256_source`; source file identities in326/327 raw |
| Compact payload | `results/native_expert_scaling/meth335_switch_w8a8_export/weights.bin`; `e0e5a940b0150b78d0080815a1fddd2a6b50f011351012ed5b4a48a88ba49056` |
| Compact spec | `results/native_expert_scaling/meth338_switch_tensor_recovery/manifest.bin`; `9c5be95504291cf9a71b087f714fc0b474ac6948389647be93919f611117fd81` |
| Qualified C binary | `results/native_expert_scaling/meth374_switch_physical_workers_contract/meth374_switch_physical_workers.exe`; `97965a35900cc0483e60f3e2e51f95da90af600e1fa326af8ec34f7c2a3dacd0` |
| Native runtime | Same binary directory `libomp.dll`; `6fc163dd513538a92a187d987438bebc5509afd1f824b20a88dd51463b3d5698` |
| Original reference environment | `results/native_expert_scaling/meth324_switch_reference/venv/Scripts/python.exe`; Transformers4.57.6/Hub0.36.0, originalTorch2.6.0+cu124 |
| Complete original-primary quality record |363 raw `ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb` |
| Exact execution quality inheritance |374 raw `4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1` |
| Same binary cost |375 raw `28b535b6a01ba67fcf45aaa008d8cf033d4f4cbc1aefcdc17c261a8749244375` |
| Same artifact accepted full rate |376 raw `5986695ed0966f110c72129b058109d028b9f4bb90c23b763d57d7a62507bea5` |

Paths above are relative to `D:\_THINGS\Progetti\SiliconLLM`. Versioned specs
contain absolute payload paths; moving them requires an explicitly rebuilt
spec and fresh parser/full numerical verification, not a presumed identical
spec hash. Native compiler is pinned clang21.1.8, SHA
`8ba7ddd7fce5574275dec0302caa1eb9f3d9ae812c6e1f44276a27b6d14fb9b7`.
Model weights and outputs are local external artifacts; Git tracks tools,
protocols and evidence, not the59GB original or14.8GB compact payload.

## Procedure from source

1. **Screen source.**321/325 check source config, indexes, actual remote tensor
   headers, namespace/ties/dtypes/core+active versus stored counts. Eligibility
   based on semantic support and measured cost, not RAM size alone.
2. **Acquire and bind.**326 retrieves all six original archives/seven side files
   and whole immutable file SHA identities.327 checks all6392 original finite
   F32 tensors, tied aliases and12x256 distinct expert pairs. Exact source
   identity precedes conversion; parameter distinctness alone is not usefulness.
3. **Convert and independently read back.**335 absmax-per-row /127 nearest-even
   I8 with F32 row scales; router/lookup/norm/relative-bias F32. Tied embedding
   and quantized head derive from SAME source; no new capacity counted from
   alternate views. Original335 final readback interrupted at20min;337 likewise;
  338 completes all canonical coefficient/code/scale/F32/padding/parser checks.
   Preserve interruptions. They are not a successfully completed335 run.
4. **Define and verify target arithmetic.**334 specifies exact rounded norm/
   softmax/scaling;356 A16 absmax/32767 nearest-even inputs for every I8 matrix,
   I32-lane bounds/I64 total/scaling, primitive/Tiny/nine-fault/full cache and
   natural controls. Original329/330/332 probability tolerance failures retained.
5. **Evaluate complete model against ORIGINAL.**362 creates24 initially unseen
   PG19 books/96 short four-span cases with prior-source exclusions.363 frozen
   all18 original-primary teacher/generation/known-answer/health/fidelity
   criteria PASS jointly. Selected source rows are now consumed; this evidence
   is not automatically new held-out evidence for another model or transformed
  bank. Read363 report for precise confidence bounds and original signals.
6. **Qualify exact worker execution.**374 reverses source edits to require
   exact356 math, validates placement detection, numerical controls and ALL96
   complete teacher/natural bytes exact363 at new six-worker recipe.375 same
   binary cost gates; only after PASS376 accepted full generation timing.

Each linked experiment protocol gives its full command, limits, parent hashes
and expected outputs. Controllers deliberately refuse existing output/working
directories. Reproduction requires a clean corresponding run environment or a
NEW named experiment retaining any existing data; do not delete or overwrite
frozen records to run commands. Original observations at configured host/root
are authoritative; cross-host/path portability is unqualified.

Compile the current qualified branch with the exact374 controller flags:

```text
clang -O3 -std=c11 -march=x86-64-v3 -fno-fast-math -ffp-contract=off -fopenmp
      -DSILICON_SWITCH_PHYSICAL_WORKERS benchmarks/phase60/engine.c -o <binary>
```

Use the pinned compiler path and libomp from374 raw. A newly compiled binary
needs its own identity/contract rather than inheriting an old executable hash.
Direct native natural-generation command schema:

```text
<binary> --generate <spec> <source_IDs_CSV> <new_output_prefix>
         6 64 32095 0 1 3 0
```

Here6=workers,64=cap,32095=closing sentinel for the four-span task,0=profile,
1=warmup,3=measured repetitions, final0=reserved. Source IDs/tokenizer are the
original source values. This is a span-denoising pretrained model, not an
instruction chat endpoint. For source29 inputs reproducing qualified363 cases,
use source_IDs in362 raw; complete output SHA and IDs are bound in363/374/376.
SWR32O01 outputs retain encoder/decoder/logits/routes, not only decoded prose.

Sanitize inherited OMP_/KMP_/GOMP_/SILICON_WORKER_BINDING_ variables; exact
runtime OMP_NUM_THREADS=6, OMP_WAIT_POLICY=ACTIVE, OMP_DYNAMIC=FALSE,
OMP_MAX_ACTIVE_LEVELS=1, KMP_AFFINITY=none,KMP_BLOCKTIME=infinite;
OMP_PROC_BIND absent.374 C discovers and pins actual workers; all normal rows
audit physical cores, thread IDs/group/masks. Current qualification is the
six-core single-group Windows host0,2,4,6,8,10, not any arbitrary six CPUs.

## Costs and scientific boundaries

Source acquisition58.860GB/1,687.563s.335 observed conversion shard times sum
1,062.437s; entire335 guard1,200.015s retained.337 separate interrupted recovery
and338 successful1,032.625s add verification cost, not zero-cost conversion.
Reference3631,721.672s/max7.336GB;374307.844s/max1.404GB;
37527.078s/max947MB;376200.453s/max1.337GB. Whole conversion/qualification is
more expensive than loading the already-qualified inference artifact.

Rate includes full encoder/crossKV/decoder/head/greedy/stop, excludes startup/
initial worker setup/model load/tokenization/serialization/cleanup. Warm/
hash-primed fixed IDs; all rejected-case times charged. This is complete model
computation, not cold text-to-text service or prose-only50. No further
experiment is implied by this guide: numbers374/375/376 are frozen evidence.

Loader d<=1024/ff<=4096/L<=24/vocab<=65536/heads*dk=d/buckets32/distance128,
column<=4096 integer bounds and Switch/ReLU/top1/cache semantics are necessary
checks, not quality guarantees. Actual64/128 exported subsets of base256 have
complete numerical contracts but mixed372 quality; they cannot inherit363.
Flat-router logical bytes scale with n;373 actual fourfold-bank full cost
upper3.62% is bounded to64-256. No universal greater-n/hierarchical LUT/physical
DRAM traffic/cold cache/100B or other-family quality/rate proof follows.

Independent original base128 now qualifies scale transfer using its OWN learned
core/router/banks, not370's removal of functions from base256. This is a twofold
candidate bank contrast and one family; neither10x scaling nor another family.
Other donor screens/failed transformations remain in METHOD/INDEX. Final goal
incomplete until the stated broader applicability and useful-capacity requirements
are evidenced.


## Current source revision and second qualified scale

The immutable374/375/376 observations use their frozen source commits. The
qualified three-worker source revision37af5e7 has388's opt-in before the six-worker
branch; current engine adds393's optional diagnostic capture before both;
frozen374 source-identity controller requires its original prefix, so execute
that controller from its frozen59e39d8 checkout for original reproduction.
Use389's source-identity controller from its frozene604af1 checkout;393 proves
current capture source reverses exactly to388 and strips to37af5e7 engine.
Do not edit frozen controllers to accept changed source. Existing qualified374 executable and
its original six-worker rate remain authoritative. Source128 now independently
qualifies at three workers through389/390/391; see
[base128 reproduction](SWITCH_BASE128_REPRODUCTION_20261004.md). Different worker
profiles/cohorts/sources prevent direct causal n or speed comparisons.

## Later opt-in prefixes and reproducibility boundary, through401

399/400 add GigaChat diagnostic LUT opt-in prefixes before393. The engine tail
from original393 SWITCH_ROUTER_AUDIT through all qualified Switch branches is
byte-identical after removing these new prefixes; default tail remains0ff9705.
Existing374/389 binaries are unchanged. For old controllers whose HEAD source
identity expects393 as first prefix, use their stated frozen checkouts; later
source additions do not silently satisfy that old HEAD gate. M401 reads/hashes
both complete qualified targets only, does not export/run a combined model.

## Later opt-in boundaries, through409

408/409 add Ling synthetic cost/profiling prefixes before400.409 controller
proves old408 engine reversal d49c951, old400 reversal02afce0 and default tail
0ff9705. No qualified Switch binary changed; old HEAD-bound controllers still
require their frozen checkouts. Ling costs are entirely synthetic and both fail
the14ms whole operator allowance; do not inherit qualified Switch rate/quality.

## Current opt-in boundary, through411

410 adds the separately frozen Ling small-pair-LUT/Q4 prefix; its source
controller proves409 reversal45c2386/default0ff9705.411 metadata touches no
engine or qualified binary. Original374/389 reproduction uses stated frozen
checkouts. Both Ling formats failed CPU cost; no original Switch quality/rate
inherited by the new Granite actual-header inventory or its proposed I8 variant.

## Current opt-in boundary, through416

412/413 add Granite synthetic I8 cost/four-row prefixes before410.413 reverses
engine to b5ceacf/default0ff9705 and full C source/math outputs to412. Both CPU
profiles fail the prospective14ms joint screen; no actual Granite values.
414/415/416 statistical source-weight/paired-input work changes no engine or
qualified binary.414 full lexical prerequisiteFAIL before scores;415 spectrum
controlsPASS;416 partial lexical ALL12 inputsFAIL, no384 model. Reproduce old
HEAD-bound controllers at frozen commits. Existing374/389 binaries/qualified
rates stay authoritative. The later417/418 observation contract is below.

## Current opt-in boundary, through418

417 adds SILICON_SWITCH_FUNCTION_CAPTURE before413. Immutable source/entries
strip all marked read-only observers and recover393 exactly; removal of new
engine prefix recovers3513e8b, default0ff9705 unchanged.417 controller schema
failure retained f2a57bd before compiler/native work.418 repairs only metadata
status fields, frozen5d596a3; same417 C compiled to new418 binary. ALL9 apparatus
gatesPASS, all192 shared405 teacher+192 original393 natural complete outputs/
router traces and greedy exact, independent ALL selected WI/WO rows byte exact.
4895 states retained; no fit/384 model/new original-quality or accepted rate.
Original374/389 binaries and source-specific profiles remain authoritative.
See [418 result](METH_418_SWITCH_FUNCTION_CAPTURE_RESULT_20261004.md) and
[current surrogate resumption](SWITCH_FUNCTION_RETARGETING_PILOT_NEXT_20261004.md).

## Numerical prototype boundary, through423

419-423 add only CPU analysis/autograd/diagnostic scripts, no new native engine
prefix or changed417 C. Original374/389 binaries/rates remain authoritative.
420 whole final-bank forward/ALL32128 logits at4895 positions byte-exact; first
real FD failed, loss conditioning diagnosed421.422 stable-loss actual256
controlsPASS/128 mixed-global gradientFAIL;423 local native-primal A/C reference
conforms but global loss primals differ. No fit/model or new rate/quality.
[Current matched-primal reference resumption](SWITCH_FUNCTION_RETARGETING_PILOT_NEXT_20261004.md)
is NEW424, not an in-place repair or promotion of422.

## Numerical prototype boundary, through424

424 adds only two CPU local-continuation/controller sources, no native C/engine
change. Frozen188a9ef, ALL8 numeric gatesPASS: both fixed real ALL6 gradient
fields max9.86120e-8, independent NumPy finite differences, fixed detached node
offsets at native values. Two actual zero-rank8 forwards exact418; full384 proof
inherited420 through exact source/fresh archives. Original422 mixed/global
contract stillFAIL. No fit/selector/export/combined quality or accepted rate.
Original374/389 binaries remain authoritative and unchanged. Next NEW425 is a
bounded actual-function/selector fit protocol, not a native rounding derivative.

## Numerical/learned prototype boundary, through426

425/426 add only CPU pilot/qualification scripts, no native C or engine changes.
425 first adapter-FD failure retained before updates;426 sole FD-conditioning
repair reproduces old error and qualifies same bounds. Complete6336 fixed
updates/matched adapter/control/causal outputs retained; capacity7/9FAIL.
Real hard mixture CE2.325199 vs original2.293060; teacher preservation fails,
function removal helps. Close specified replacement route before sweep, no
combined384 native model/quality/rate. Original374/389 artifacts still exact.
Next NEW427 is an additive original-preserving interface, TWO final-bank
functions when gate on. Its added active cost and whole rate cannot inherit
original accepted results or426's proposed single-function inference budget.
