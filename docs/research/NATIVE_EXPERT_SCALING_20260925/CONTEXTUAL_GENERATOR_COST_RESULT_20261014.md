# Actual BF contextual generators: cost and precision qualified

14 October 2026. **COMPLETE / full numeric audit and closure PASS. Dense
baseline exceeds remaining budget. Goal ACTIVE / INCOMPLETE.**

Executed actual source in_proj at sites1..23, sequential batch1 CPU calls,
affinity0/single thread. Shared read-only BF16 coefficient ranges directly in
the donor checkpoint, total631570432bytes/315785216 distinct parameters.
No expanded F32/F64 weight storage during timing. Each call decodes2048 BF h
once to8192-byte F32 stack and weights inside AVX2 registers, accumulates
eight FMA lanes, ordered reduction, rawF32=>GG BF, then actual6704-entry BF MuP
multiply inF32=>fields BF. First-layer source-only F0 lookup is excluded.

| Mode,64 positions each | Whole23 median ms | Min / p90 / max ms | Remaining recurrent room ms |
| --- | ---: | --- | ---: |
| Resident, no deliberate eviction | 52.15740 | 49.61060 /55.67900 /58.45640 | 3.18275 |
| All23 weight ranges +shared MuP evicted before group | 50.42935 | 49.54750 /52.35740 /55.48330 | 2.97615 |

Every observed generator group alone exceeds the20ms full-model target.
The frozen decision compares generator medians with the qualified recurrence's
remaining3.18275/2.97615ms: FAIL in both modes. This is a finite measured dense
baseline, not a lower bound on every quantized/conditional/nonlinear conversion.
Do not sum separately measured recurrence and generator medians into a claimed
end-to-end latency or useful tokens/s. CPU sharing/cache histories differ.

All four fixed shortest/longest FIT/DEV histories use final16 real positions,
from admitted own-history traces. Resident record medians50.21525..53.62640ms;
evicted50.16035..50.91060ms. No discarded tails/records.128 same-ABI no-op
groups median.01890ms, not subtracted; per-site clock/wrapper overhead retained.
Eviction median13.19275ms is outside generator timing, intentionally recorded.
Resident means no explicit eviction;631.6MB exceeds cache and is not claimed
resident in L3. The slightly faster evicted screen does not prove a causal
cache speedup. Shared desktop, no exclusive scheduler/hardware guarantee.

## Complete precision and stored audit

All2944 timed calls/8 output archives:19736576 words EACH of rawF32, GG BF and
fields BF. Auditor independently computes F64 products from exact BF source
values for every input; raw maxrelative1.1056500621896033e-7<=1e-4. Every stored
raw/GG/fields word exactly reproduced by C replay, and manual RNE(raw) plus
ordered F32 BF-GG*BF-MuP rounding matches all stored BF words exactly.
Finite inputs/source values/outputs and FE_TONEAREST/MXCSR0x1f80 enforced.

Recorded source-field and smooth F64=>BF=>MuP controls cover each16-position
record/site, overall and gate/x/B/C/dt slices. Max source slice relative
.00018069102220482023 (0.0180691%); max smooth slice.00021155238529844264
(0.0211552%), both below frozen1%. Source GPU rounding is not assumed bit
identical to CPU. Matrix ranges/shapes/addresses/bytes, all sample coverage,
no-op summaries, per-site/record/whole timings, output sizes and decisions
independently checked. Shared mapping/input/C replay/hash helpers are declared;
F64/RNE/statistics implemented separately. No independent clock authentication.

Exact algorithmic ledger:631570432 weight bytes per23-call group,
94208 supplied BF h bytes,315785216 scalar FMAs,1263140864 logical decoded
input read bytes,308384 logical MuP read bytes,1233536 output bytes. Input
stack/cache reuse and register decode differ from stored coefficients.
These are algorithmic operands, **not measured DRAM traffic**. Actual physical
traffic remains unresolved. Preceding decoder, recurrence, FFN, attention,
head, tokenizer and CPU/GPU transport are excluded from this screen.

## Custody and reproduction

Code/C/auditor/protocol frozen8ce16b9; binding/state committeda473270 before
timing. Per phase1800s/OS4GiB/new E192MiB. Every declared runtime/source/trace/
code/protected input hashed before/after; outputs exclusive. No preparation
fault or repair. All observed receipts exit0:

| Phase | PID | Seconds | OS peak bytes |
| --- | ---: | ---: | ---: |
| Preparation | 10584 | 17.360 | 44183552 |
| Producer | 28596 | 34.937 | 778592256 |
| Independent auditor | 25044 | 40.594 | 904724480 |
| Closure | 17808 | 5.109 | 28762112 |

Closure checks PID AND creation time and all descriptors. New E family10files/
157972256bytes:8 outputNPZ plus DLL/log. Output NPZ157898528bytes, numeric
payload157892608bytes. Original checkpoint/traces are reused, not copied.
GPU calls/source model forwards/optimizer updates/RESERVED queries all0.

- [Protocol](CONTEXTUAL_GENERATOR_COST_PROTOCOL_20261014.md), [state/reproduction](CONTEXTUAL_GENERATOR_COST_STATE_20261014.md).
- [Binding](contextual_generator_cost_binding_20261014.json),281715bytes,SHA256468a7321a2bba619fda55cb78aecb9ed769987ee6d3fb0906831ee297a53efcf.
- [Producer](contextual_generator_cost_result_20261014.json),150179bytes,SHA25655eac059d75ca1c3d3080f8caac5259af031c59bff53bf612de63b173b0e0dfa.
- [Full audit](contextual_generator_cost_adjudication_20261014.json),108869bytes,SHA25682de1ccda7750817f6cf3d0e3999c2725e925833152e7634c8d53d66e00e74e4.
- [Closure](contextual_generator_cost_session_closure_20261014.json),2644bytes,SHA2563ae0e12dbb5a0b523dd9ffab9ec97b244363dd5e3f554b2c3c359fa51b9d5de0.

## Decision toward the full goal

Close this baseline; do not polish another dense minor kernel or expand an
unpriced FFN rank grid. The next variable is **executed source block depth**,
with residual2048/source functions preserved at kept sites, identity for omitted
whole blocks, qualified native state and NEW causal hidden at every kept site.
[Frozen intervention](FIXED_DEPTH_PROTOCOL_20261014.md) separates FIT selection,
DEV/tasks and mechanical verification, explicitly not useful-n evidence.
This tests whether depth can be reduced before further width/routing/adaptation
work. It avoids the previous simultaneous residual256/random-core/operator
changes; no claim of function-preserving identity deletion is made.

Depth prototype is not a qualified chatbot. GPU quality worker/binding/full
auditor and sparse-KV continuation controls remain to implement. Any physical
CPU artifact must still eliminate source FFN/attention/head dependency, prove
quality/useful>=50 on the same artifact, useful n and measured DRAM, then
multiple families/scales and conversion training/storage cost. All remain open.
