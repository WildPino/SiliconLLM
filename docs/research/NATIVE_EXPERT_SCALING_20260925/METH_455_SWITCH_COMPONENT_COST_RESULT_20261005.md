# METH-455 result: valid component diagnosis selects exact output-tiled WI

5 October 2026. Goal ACTIVE/INCOMPLETE. Previous turn PROGRESS:454 actual C cost
closed two fixed recipes. This turn PROGRESS: a new admitted stage measurement
identifies where a different physical geometry must reduce cost.

ONE execution, session53109 terminal exit0; no prior/future controller rerun.
Scientific freeze `db7c388`, exact first-run resumption `211a23a`. Raw retained
unchanged at `80bff54`:75,712B SHA256
`049e8ea6ee63f0b2efa6204e37aaaf70842bf0185c1c814720e77fba8a2d07a5`.
[Prospective protocol](METH_455_SWITCH_COMPONENT_COST_PROTOCOL_20261005.md),
[raw](meth455_switch_component_cost_result.json).

## Qualification and admissibility

ALL7 apparatus gates PASS.3360 original/correct/wrong full-state records
(1680 split without internal timers,1680 with timers) BYTE exact454: IDs/counts,
bases, raw WI, WO outputs, both A16 arrays/scales. Retained tiny454 suite plus
split WO FULL scalar I64/extrema/zero/single checks. ALL21,168 timed outputs byte
exact454; exact nine-group whole336-trace order, two warm/five measured sweeps,
fixed real IDs/books. One physical worker CPU0/group0/mask1 readback in both
children. All14 prior archives/source/runtime/compiler/payload freshly bound.

ALL15 predeclared diagnostic admissibility gates PASS across three arms.
Mean split/unchanged ratios: original.992419, direct1.001393, LUT1.004638;
mean profiled/split ratios: .989901,1.002543,1.003514. EVERY book is within its
frozen15% interval, and empty QPC boundary sequences occupy .0685%/.1041%/.0372%
of profiled means. Observed signed differences include run variation and codegen
effects; these do not imply negative timer overhead or a statistically isolated
causal effect. No time correction/subtraction/outlier trimming.

QPC10,000,000Hz (100ns tick). Empty boundary means .16137/.24548/.27186us;
raw profiled residual boundary/return means .22458/.14518/.18149us retained.

## Whole selected-FFN costs in this NEW diagnostic

| Arm | Unchanged454 code mean us | Split/no internal timers mean us | Profiled mean us |
| --- | ---: | ---: | ---: |
| Original I8 |239.7245|237.9073|235.5047|
| Direct I4/sparse I8 WO |234.9280|235.2552|235.8533|
| Pair LUT/sparse I8 WO |724.0517|727.4095|729.9658|

These newly qualified reference costs support component diagnosis. They do not
reopen failed454 cost gates or represent whole-model accepted rate. All5 sweep
sums, p95/max and per-book means retained in raw; no selectively chosen sweep.

## Measured components (mode2; raw elapsed, not overhead adjusted)

| Ordered component | Original us | Direct us | LUT us |
| --- | ---: | ---: | ---: |
| Signed F64 basis/cast |0|1.2579|1.2715|
| WI A16 |1.6457|1.5763|1.5768|
| Complete LUT build |0|0|160.3541|
| WI dot/decode/reduction/scales |120.1746|188.9428|522.4023|
| ReLU |1.3355|1.2727|1.2786|
| WO A16 |6.0930|5.9947|6.0638|
| WO zero scan/indices |0|2.5124|2.5093|
| WO column products/partial-I64 accumulation |0|33.2576|33.4437|
| WO final scales/output |0|.8939|.8841|
| Original WO dot/scales/output |106.0313|0|0|

**Direct WI is80.1103%** of the profiled selected FFN; WO accumulation14.1010%.
Original WI is51.0285% and original WO45.0230%. LUT build21.9673% plus LUT
application71.5653%: even a zero-cost build would leave a large measured WI stage.
This does not isolate indexed reads from reduction/scales, or cycles/DRAM stalls.

Sparse WO scan+accumulation+output is about36.664us versus original WO106.031us,
while direct WI188.943us exceeds original WI120.175us. Their roughly69us opposing
differences explain the observed near-equal whole cost within this decomposition.
They are an algebraic accounting of measured stage means, not an unperturbed
counterfactual or evidence of cache residency/physical DRAM bytes.

WI decode/dot/block reduction/scales remain COUPLED in the actual row/block loop;
original WO dot/scales remain coupled. No claim that one particular suboperation
alone caused the cost. Per-block timers/new intermediate tensors were avoided to
preserve physical behavior, as disclosed prospectively.

## Postrun command-record fault and numbered metadata correction

Raw `native_commands.primal.argv` is INCORRECT: the frozen controller stored the
mutable `common` list by reference, then changed mode/output for the profile call.
The serialized primal argv consequently duplicates the profile argv. This affects
command provenance, not the separate exit codes/durations/stdout/binary/state data.

First fault/raw retained BEFORE repair at `80bff54`:
[first fault](METH_455_COMMAND_RECORD_FIRST_FAULT_20261005.json),1627B SHA
`47c08a6defb31fb66c6a363a2fabd5295b991bb4a40818eac56b2a9cbe6940ee`.
NEW455-R1 is metadata-only: derive the qualifying argv from the committed initial
list/call and hash-bound native stdout `mode=--qualify`, tiny=true, qualified wire
header/full byte comparison. No frozen raw/source edit, scientific import, new
forward or timing rerun. Correction is a source/log-derived reconstruction;
no OS process-argv snapshot was captured. Future controllers must store `list(argv)`
or immutable tuples and create separate argument lists for every child.
[Numbered correction](METH_455_COMMAND_PROVENANCE_REPAIR_1_20261005.json).

## Costs and retention

25.328s total=8.953 admission+16.375 numeric. Compile1.531s, exit0/empty warnings;
native primal3.0204262s (wrapper4.531), native profile9.0899279s (wrapper9.610).
Parent peak167,489,536B; native peak983,973,888B; conservative sum1,151,463,424B.
ALL resource bounds pass.10 retained outputs95,126,063B;8,437,598,852B hashed before
raw write/final raw hash. Zero new coefficient roundings/updates/GPU/downloads/
engine edits. Binary SHA `e6e64dae33c549bf508509be18375674422ea35a3793e8aa535f91b61cb9f001`.

Primal87,843,868B SHA6dcf8e2e...,timings5,686,905B SHAb3513fea...; complete
individual sizes/hashes/native logs/commands/compiler/environment in unchanged raw.
[Final retention audit](RETENTION_455_20261005.json) checks fresh files and metadata,
never re-executes science. No model processes live at terminal inventory.

## Decision and full-goal limit

The PROSPECTIVE direct-WI>=.50 condition is met. Choose ONE NEW exact output-tiled
WI physical format/operator, with inverse coefficient/scale qualification, full
BYTE primal and actual whole selected-FFN cost gates. No other format sweep.
[Next specification](SWITCH_OUTPUT_TILED_WI_NEXT_20261005.md).

If no viable format, stop this kernel refinement and reassess transfer economics.
If viable, all-bank composition/new routes and fresh donor-relative prediction,
generation/tasks must precede SAME>=50 acceptedIDs/s claim. Full matched component
fraction f is still missing; one-worker bank11 cannot supply391 full-context f.
Actual useful-n increments/router mass/DRAM and other families/~100B remain open.
