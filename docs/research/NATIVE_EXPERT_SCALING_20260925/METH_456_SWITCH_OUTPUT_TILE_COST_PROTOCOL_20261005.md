# METH-456: ONE exact output-tile16 WI permutation and native whole-FFN cost

5 October 2026. PROSPECTIVE: new C/controller uncompiled/unimported at writing.
Freeze both sources, this protocol and attributes BEFORE first import/compile/
permutation/numerical observation. ONE controller execution; first failure retained
before a NEW numbered repair. Goal ACTIVE/INCOMPLETE. Previous turn PROGRESS:
455 admitted component measurements selected this ONE physical inquiry.

## Decision, evidence and frozen alternatives

455 direct WI is188.9428us/80.1103% of its profiled selected FFN, versus original
WI120.1746us. Sparse WO's roughly69us savings are offset by direct WI's roughly69us
overhead. ALL3360 full-state BYTE and21,168 timing-output checks passed; all15
perturbation/admissibility gates passed. The predeclared WI>=.50 condition selects
an exact coefficient permutation across output coordinates. It does not isolate
one particular scalar instruction or prove cache residency/DRAM/cycle behavior.

This experiment asks whether tile16 grouped coefficients remove repeated per-row
horizontal reductions sufficiently to improve FULL selected-FFN cost. SAME453
coefficients/scales/signs, original A16, original-coordinate ReLU and exact sparse
native I8 WO. No new coefficient roundings, fit, gate relaxation, precision/data
changes, omitted nonzeros or source masking. No tile-size/layout/LUT/WO sweep.

454 direct/full-pair-LUT cost recipes remain CLOSED. Old direct454 is a diagnostic
comparator only; it cannot be promoted by new456 observations. Only NEW tile16
is eligible for the predeclared candidate gates below.

## Fresh complete bindings

455 raw75,712B SHA
`049e8ea6ee63f0b2efa6204e37aaaf70842bf0185c1c814720e77fba8a2d07a5`;
retention455 SHA
`314b79a61a94b9de308b12d979a5fc520014ad856a6c6198a85f4a1f7f92e056`;
numbered455-R1 provenance correction SHA
`dcbf35cdb2d326872401e2663920c26bcc2fbaea56bdddc0b999b2e94fee1f52`.
Freshly verify ALL10 complete455 and ALL14 complete454 output hashes/sizes,
all frozen helper/protocol/parent-record hashes, original128 payload7,541,946,880B/
manifest, engine54194c36..., original374/389 binaries, prior runtime DLLs, compiler
8ba7ddd7... and libomp6fc163dd.... Own source/protocol/controller physical bytes
match committed HEAD and attributes raw HEAD. No old controller/main invocation.

Reuse454 fixed normalized trace336/real original IDs/books18..23, bank11,
D768/M3072/E128, and43,915,224B qualified primal. This is consumed local input,
not new held-out data or freely changed routes. Native original `mv`/model loader/
worker primitives included UNCHANGED through454 ->374. Source baseline is not
replaced by the455 split/timed surrogate. Original3worker391 rate is separate.

## Exact stored geometry and inverse proof

ONE streamed new bank; no all-bank model conversion yet. Header4096B:
MC456B01, U32D768/M3072/E128/header4096/tile16. SAME768 signed WI basis bytes.
For each of128 actual functions:

    packed WI: [192 tiles,12 blocks,32 adjacent-input pairs,16 outputs]
        1,179,648B
    WI F32 scales: [192 tiles,12 blocks,16 outputs]
        147,456B
    original WO I8 columns: [3072,768],2,359,296B
    original WO F32 row scales: [768],3,072B

Stride3,689,472B, expert0 offset4864B; all fields64-aligned. Every WI tile/block
is512B, each pair plane16B; scale plane64B. No padding/expanded-I8 deployment
weights. Total472,257,280B, nominal excluding header472,253,184B. Original native
I8 bank605,945,856B: ratio77.9365%, SAME453 budget. Original bank is retained as
diagnostic reference only, not additional candidate deployment memory.

Stream one expert at a time to avoid resident whole-bank copies. Independently
read both complete files and inverse transpose ALL128 WI code bytes/scales;
signs/WO/scales byte-identical. C validates shape/alignment/every sign/code/scale,
including reserved--8 I4 and--128 I8 rejection before operator use.

## Integer, floating and SIMD invariant

For EACH original output row r, retain:

    value = +0.0(F64)
    for b=0..11 IN ASCENDING ORDER:
        part = exact I32 sum of64 SAME I4*A16 products
        value = F64(value + F64(part)*F64(original_scale[r,b]))
    up[r] = F32(value * F64(original_activation_scale))

64*7*32767=14,679,616 bounds every partial integer sum; signed I32 safe.
Each pair plane loads16 packed bytes, sign-decodes to I16, broadcasts the SAME
two A16 codes and AVX2 pair-multiply-adds into16 output lanes. Sum32 pair products
in exact I32 lane vectors. No horizontal output reduction. Convert groups of4
I32 values to exact F64, multiply original scales, add in the SAME ascending block
order for every row. Multiply alpha last and castF32, FE_TONEAREST. No FMA or
floating reassociation. Signed A16 pair bit packing uses uint16/uint32 and memcpy
to avoid signed shifts/overflow or ambiguous value conversion.

Basis/A16/ReLU/WO use UNCHANGED454 functions. WO partials preserve512*127*32767
=2,130,641,408<2^31 then I64 combination. Zero-only skipping, exact original
coefficient bytes/row scales; no learned pruning or approximate sparse threshold.

## Qualification BEFORE cost

Retain complete454 tiny suite (signed direct, all LUT table/sentinel controls,
WO FULL scalar I64/extrema/zero/single, explicit Walsh). NEW tile-dot controls:
225 signed coefficient-pair states crossed with9 A16 extrema/zero pairs, tested
for EACH16 distinct output lanes with deterministic lane permutations. Also one
mixed pair/input/output pattern, FULL independent scalar I64 comparison. Native
reserved--8 fault child EXPECTED exit2 and named stderr; preserve both logs.

Then1680 MC456Q01 records (same26140B field wire as454, fresh magic): original336,
old direct672 correct/ID+1, NEW tiled672 correct/ID+1. Shape header336/768/3072/1680,
file43,915,224B. Python BYTE compares every record to matching454 record, including
IDs/counts, basis/raw/down, both A16 codes/scales. Prior LUT arm and direct arm
were fully byte-exact454; tile arm compares against that qualified matching arm.
First mismatch stops BEFORE any timing; no tolerance-only acceptance.

## Actual cost schedule and fixed candidate gates

Three arms: UNCHANGED original I8, CLOSED454 direct diagnostic, NEW tile16 WI +
SAME exact sparse WO. No internal profiler timers. Two warm whole336 traces and
five measured whole336 traces per arm. Repetitions -2..4; rotating arm start
`(rep+2)%3`; finish entire336-input trace before next arm. No same-input alternation.
ALL7056 output vectors BYTE compare to prior qualified454 AFTER timer; checksums/
logging after timer, SAME scope as454. Windows QPC around FULL selected FFN:
basis, quantizers, packed WI decode/products/F64 scales, ReLU, WO scan/indices/
integer accumulations/scales/output all charged. No preload-selected coefficients.

Report mean/median/p95/max, all5 complete sweep sums and EACH book mean, preserve
all raw7056 records including warmups. No outlier removal, tuned best run, timer
overhead subtraction or cross-run inferred significance.

For NEW tile16 candidate require ALL:

1. Mean selected-FFN time /CURRENT unchanged original mean<=.80.
2. EACH book mean /CURRENT unchanged original book mean<=1.00.
3. p95 time /CURRENT unchanged original p95<=1.00.
4. Nominal stored bank bytes /original native-I8 nominal bytes<=.80.

ALL apparatus gates required. If all pass, prepare NEW all-bank source geometry/
sparsity/composition/export inquiry, retaining local quality limits. If any fails,
close this fixed tile16 recipe and reassess transfer economics; no adaptive tile
or alternate-format sweep. Numerical success alone never rescues cost failures.

## Resources, stops and reproduction

Expected tens of seconds/<600MiB files, no optimizer/fit/GPU/download/engine edits.
Same compiler -O3 -std=c11 -march=x86-64-v3 -fno-fast-math -ffp-contract=off
-fopenmp -lpsapi, SAME ACTIVE/none/infinite/FALSE/one-level/OMP_NUM_THREADS1.
CPU0/group0/mask1 one physical worker with topology/readback. Native test maps
original source plus old/new complete banks; report each mapped byte count and
peak RSS, do not label combined test mappings as candidate deployment size.

Hard bounds: admission300s/new numeric600s/parent total900s; compile120s/negative
fault30s/primal240s/timing300s child wrappers, native total600s. Parent512MiB,
each native2GiB, conservative sum<=2.5GiB. Outputs600MiB, initial free disk2GiB.
First mismatches/resources/layout/affinity failures retained before numbered
repairs; timeout kills AND waits child and preserves available stdout/stderr.

Reject other model/Python jobs except actual current ancestors and exact approved
pythonw publisher two-argv allowlist. Future command records copy `list(argv)`;
separate immutable child argv avoids the retained455 provenance fault.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth456_switch_output_tile_cost.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth456_switch_output_tile_cost_result.json
```

After ONE execution retain raw/output hashes/logs/first failures; update INDEX,
METHOD and precise resumption. Scope is one real bank/consumed local routes:
no new full quality, matched whole-model f/accepted rate, hardwareDRAM/useful-n/
router mass or cross-family/~100B claim. Those remain full-goal requirements.
