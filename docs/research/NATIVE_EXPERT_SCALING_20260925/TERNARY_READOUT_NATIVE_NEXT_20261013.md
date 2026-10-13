# Selected next: execute the qualified six-plane readout through a native LUT

13 October2026. **PROPOSED / UNIMPLEMENTED / UNBOUND.** Six source-weight-only
balanced ternary planes now fully qualified:all144 actual-input histories,
FIT-selected p6 passes DEV/additional raw and BF16 each-case1% and original
donor local gates. All packed codes/q/scales/spectra/BF16 outputs independently
verified. This is an accurate input-ready function artifact,not a native LUT
rate or an OWN-history chatbot. Preserve SAME selected codes/scales/inputs.

## Concrete native boundary

Implement one C readout function over BF16-normalized input3072, output2048,
p6 five-trit uint8 codes[plane,group,output],615 groups,F64 row scales. New
floating LUT backend;existing engine2-trit int8 pshufb/nibble qualification
does not transfer. Preserve input BF16 values, no activation quantization.
Freeze precise arithmetic/accumulation/BF16 output rules before binding.

Initial layout:8-group block-local F64 tables,243 values/group,15,552B table
workspace. Six2048-vector F64 plane accumulators98,304B;combine powers3^d and
row scale once/output. Evaluate AVX2 gather across output coordinates using
actual group-major codes. No hidden cache-residency assumption or parallel
all-expert scan. Count table construction,all six plane gathers/partial sums,
scale/output conversion,workspace and all source bytes. Padding final three
input coordinates zero;reuse exact qualified pattern convention.

## Source control and accuracy BEFORE cost admission

Original source F32 O with actual engine dotf/hsum256 AVX bodies,compiled under
the qualified original flags,provides source native output control from the
SAME input Z. Compare every control raw F32 score bit against stored original
native Y,without rerunning recurrence. New LUT raw output reference=(q_F64 Z)
timesF64 scale,using already qualified q/code/scales. Every actual-history
output/word/metric/source gate audited;scalar independent pattern-decoded
reference and mutation/digit controls must detect code/layout faults.

Proposed numeric raw relative1e-10 for F64 LUT vs integer-q F64 reference,
BF16 smooth1e-4 with every bit mismatch retained/rounding analysis where
needed. Original additional1% raw/BF16 and donor budgets unchanged. These
tolerances need explicit implementation protocol/review before launch.
No speed admission if numerical/quality gates fail;full audit still completed.

## Measure actual active cost on this accurate artifact

After correctness,bind bounded batch1 source-derived input-ready timings for
same selected p6 and exact original F32 operator. First warm controlled mode,
then explicit cold-source mode:qualify CPU CLFLUSH support,flush every packed
code/scale cache line (original F32 control likewise) and serialize before
timed call;include LUT construction/gathers/conversion in timed region.
Cache-flush work itself reported separately. Use fixed source-derived sample
positions/repetitions,all3 sites,CPU/core/environment identity. Avoid an hours
campaign evicting buffers before every one of all67k history positions.

Cold initial cache state needs explicit evidence;architectural eviction alone
is not a hardware counter measurement of total DRAM traffic. Report these
distinctions,source bytes/latency and shared-desktop limitations. Physical
expert-DRAM/n scaling remains a separate required admission. No inherited old
synthetic rates. This output function alone is not an end-to-end model token.

Prior M398/M400 full-width synthetic-code decoders rejected~.14/.12s fixtures;
reuse their blocking lessons,not unchanged campaign. If selected-p6 LUT is
slower than original dot,attribute builder versus gathers/scratch traffic
before deciding layout or precision changes. Do not claim speed from7.57MB
versus25.17MB counts. Initial implementation/run900s/audit1200s/OS6GiB/new
orderedE<=2GiB proposed;exact source/runtime/compiler/storage/timing bound
only after code/protocol freeze. No GPU/T4/hours allocation now.

## Keep complete conversion objective intact

This preserves trained readout geometry while testing an engine-compatible
primitive extension. Source input generator/SSM composition/attention/FFN/
tokenizer/categorical head/OWN-history chatbot and useful selective ternary
bank still need transfer and qualification. Compact shared causal core,
RAM-driven n/10B/100B/multiple source families/scales/conversion master/gradient
storage and SAME useful>=50 token/s are unresolved. No generic full-width
donor runtime or small operator result can complete that objective.
