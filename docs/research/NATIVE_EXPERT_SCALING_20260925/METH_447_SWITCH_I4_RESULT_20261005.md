# METH-447: row-I4 saves storage, fails the declared argmax gate

**Qualified local result; fixed recipe CLOSED. Goal ACTIVE / INCOMPLETE.**

## Reproduction and frozen definition

Source/math/[protocol](METH_447_SWITCH_I4_PROTOCOL_20261005.md) frozen in
c122074 BEFORE first import/quantization/forward. Operational command recorded
in8644148 before execution. One terminal session8604, exit0, no rerun or repair.

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth447_switch_i4_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth447_switch_i4_result.json

Raw record: [meth447_switch_i4_result.json](meth447_switch_i4_result.json),
SHA2563025bb7b383a631162c2eca803784e0c2cdc811c47a6abe1f73619a791e6dd29.
Controller meth447_switch_i4_pilot.py; encoding/primal meth447_switch_i4_math.py.
All source/helper/protocol and runtime binary hashes are in the raw record.

Same original source128 finalbank11, all128 REAL functions, WI3072x768 and
WO768x3072. Single calibration-free symmetric row encoding, levels-7..7,
represented F32 scales, nearest-even rounding, two signed nibbles per byte.
Original A16 activation quantization and native residual/final RMS/tied I8 head.
Exactly336 previously consumed original teacher-prefix positions, books18..23,
four cases/book,14 positions/case. Original core/input/p fixed; no fitting.

## Apparatus: ALL nine PASS

- Fresh source/manifest/descriptors, parents443/446/418/420 and retained outputs.
- ALL336 original native states, A16 scales/codes and full32128-vocabulary heads
  byte-exact, logit stream identical443 before quantized predictive conclusions.
- ALL256 complete matrices pack-inverse exact; saved full128 bank byte-exact.
- All225 weight pairs times9 activation extrema, zero rows/input, forbidden-8
  low/high nibble and nearest-even checks qualified.
- Width3072 integer bound704,621,568<2^31; ALL1344 actual I4 WI/WO projections
  have exact I32 versus independently decoded I64 sums and F32 outputs.
- Same original input/probability/final normalization/head for all controls.
- Full-vocabulary KL via CE-minus-entropy and independent log-probability
  difference agree within1e-10, finite and nonnegative within tolerance.
- Removal complete heads byte-exact443.
- ONE encoding; zero calibration/training/updates or data/ID selection.

These checks qualify Python arithmetic and the stored bank. No C I4 kernel,
whole-model export, generation, task-quality, accepted rate or DRAM measurement.

## Fixed gates and actual result

| Gate for original-ID I4 | Required | Observed | Outcome |
| --- | ---: | ---: | --- |
| Mean original-posterior KL, nats | <=.01 | .001377877644269 | PASS |
| Every book mean KL, nats | <=.05 | maximum .001999447593943 | PASS |
| Changed argmax fraction | <=.01, at most3/336 | 4/336=.011904761904762 | **FAIL** |
| ID+1 increases mean source KL, nats | >=.01 | .177920820995395 | PASS |
| Nominal bank coefficient/scale ratio | <=.51 | .501622323166775 | PASS |

Four of five pass. The predeclared argmax threshold is not relaxed because the
miss is one position. Close THIS symmetric full-row I4 encoding before C export
or native timing. No universal low-bit impossibility follows.

Correct-ID KL median .000140468241003, p95 .005800473006740,
maximum .026347964180613. Per-book correct-ID results:

| Book | Positions | Mean KL | Argmax changes |
| --- | ---: | ---: | ---: |
| 18 | 56 | .001970403602462 | 1 |
| 19 | 56 | .001204290273556 | 1 |
| 20 | 56 | .001468400033232 | 0 |
| 21 | 56 | .001999447593943 | 1 |
| 22 | 56 | .000884069199295 | 0 |
| 23 | 56 | .000740655163123 | 1 |

| Counterfactual at SAME original p | Mean source KL | Argmax changes |
| --- | ---: | ---: |
| I4, original ID | .001377877644269 | 4 |
| I4, ID+1 mod128 | .179298698639664 | 41 |
| Function removed | .032704093152633 | 21 |

Parent443 original-I8 ID+1 mean KL .176457652840979. Current I4 intervention
retains a substantial joint bank identity effect. This does not count useful
individual functions:106 naturally selected IDs and128 distinct packed expert
fingerprints are descriptive counts, not useful-n proof or scaling.

The four changed positions, reported from the frozen per-position metrics:

| Book/case/position | Original expert | Original p | KL | Relative down error | Relative head-input error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 18/2/2 | 33 | .2639080286 | .0007081529431 | .1477229552 | .0072312515 |
| 19/2/2 | 38 | .4000397623 | .0033928044278 | .1096714540 | .0147701516 |
| 21/3/7 | 72 | .9156016111 | .0263479641806 | .0603165401 | .0283959964 |
| 23/1/4 | 52 | .3398241997 | .0037058723198 | .1258743097 | .0150220840 |

No source-margin or semantic diagnosis is inferred from these rows. They remain
consumed diagnostics; a future representation must receive new full held-out
evaluation, not a quality claim from correcting these four examples.

## Geometry, information and the next variable

Full-row I4 coefficient Frobenius relative errors across all128 functions:
WI .13585755..18799329, WO .15466513..40298235. These are matrix metrics;
they do not equal native function error or posterior KL. Unlike446's forced
three-expert/all-input assay,447 uses the naturally selected functions and full
original readout. No ratio across unlike estimands establishes improvement.

For decoded logits z and perturbation delta, p=softmax(z), the exact identity is

    KL(p || softmax(z+delta)) = log(sum_i p_i exp(delta_i)) - sum_i p_i delta_i.

In the small-perturbation limit its quadratic term is one half of the p-weighted
variance of delta; a constant logit shift changes no posterior. Argmax additionally
depends on pairwise margins: a competitor wins when delta_j-delta_w exceeds its
original logit gap. Low mean KL alone therefore does not guarantee stable argmax.
This is an algebraic explanation of the separate gates, not a measured margin
diagnosis or a whole-model error-composition bound.

Next proposed NEW448 variable: replace one full-row scale with fixed64-coefficient
block scales, preserving all coefficients and original IDs, without fit/clipping
search. Ideal-real max quantization bounds then depend on each block maximum,
with represented F32 scales/rounding still needing separate qualification.
Choose64 prospectively as the SMALLEST power-of-two block that keeps this
geometry's nominal storage below60% of source I8:32 fails that byte budget,
64 passes. This is a byte-budget choice, not a validation-error/grid choice.

Keep447's KL/book/argmax/identity thresholds for the future diagnostic; explicitly
freeze the new block encoding, accumulation contract, storage gate and resources
before any new quantization. No448 source/protocol/output exists at retention.
It remains possible that this new encoding fails. Active C/LUT cost may also fail
even if local predictive quality passes.

## Resources and retained outputs

115.625s total:13.687s admission +101.938s conversion/numerics/reporting.
Peakprocess2,426,564,608B; freshly hashed11,548,503,277B. CPU0/BLAS1/Torch1,
zero optimizer updates, no GPU, source downloads, new corpus or C modification.

Bank nominal coefficient/scale bytes605,945,856 ->303,955,968. Actual bank.npz
303,956,990B; storage ratio is not latency or traffic measured at DRAM.
Nine outputs total519,318,028B, all paths/sizes/SHA256 in raw inventory:
full bank, complete original prefixes/scores/IDs, original full logits, three
counterfactual full-logit arrays and three complete state archives. All retained.
Original engine/source payload/374/389 binaries intact; session terminal.

Full goal remains open: useful RAM-scaled n, routing/normalization, CPU LUT,
actual DRAM, whole original-relative held-out/generation/task quality AND
>=50 acceptedIDs/s on SAME new artifact, additional families/actual~100B.
