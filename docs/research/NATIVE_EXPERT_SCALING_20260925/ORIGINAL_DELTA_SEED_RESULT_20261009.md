# Original-engine delta correction: actual27 durable, quality still fails

9 October 2026. Goal ACTIVE/INCOMPLETE. All owned jobs terminal; no T4.

## Decision and scope

The [selected protocol](ORIGINAL_DELTA_SEED_PROTOCOL_20261009.md) and
[missing-only completion](ORIGINAL_DELTA_SEED_FINISH_PROTOCOL_20261009.md)
are COMPLETE. Initial native function preservation, actual new delta gradients,
one full-history optimizer update, durable state/export and resource gates PASS.
Strict GPU/C numerical gate FAIL. Native source-relative quality remains poor.
This closes the initialization defect identified in
[actual26](ORIGINAL_WIDE_BRIDGE_RESULT_20261009.md); it does not demonstrate that
extra delta width causes recovery, or justify a large unchanged training dose.

The target is still the original LUT/ternary/AQ63/dReLU/SSM/SWA arithmetic:
D256/N96/DN1024/DT48/L6/n1152/k8/H128/V65537, five SSM sites and one SWA site.
No original computational kernel was changed. The 20 original native bodies and
their prior [cost envelope](ORIGINAL_NATIVE_ENVELOPE_RESULT_20261009.md) remain
the deployment foundation. Full-vocabulary raw fixture speed is not useful
chatbot speed and is not remeasured by this IO-heavy qualification.

## Actual initialization and update

Load actual26 real masters/Adam/RNG/ancestry; never reconstruct masters from
dequantized packed experts. FIT history magpie022 has 1507 positions. Five
postconv/SiLU X arrays, 1507x1024 F32 each, were captured from actual26 tensors
and qualified retained core_norm inputs. The F64 helper uses

\[
D_0=XU_0^T,\quad A=D_0^+X,\quad R=X-D_0A,
\qquad U_{new}=W-(WA^T)U_0.
\]

W contains 32 principal residual directions. Match old median response RMS;
cap each new row norm at 10 times the old median norm. At all five sites the
old response rank is 16, new response rank 32, and F32 cross-response relative
norm is 5.228e-9..7.550e-9 (required <=1e-6). Significant residual eigenvalues
and row norms pass. These are FIT activation-support facts, not preservation
of source history or a capacity theorem.

Only x_proj[16:48,:] is seeded: 163840 F32 coordinates. New dt_proj columns
remain zero, all old coordinates/moments/steps26/RNG exact. Packed header/table,
105 other fields and old rows in the five modified fields match bitwise.
All 3352 native complete head rows, routes/masses and 18432 integer witnesses
match actual26 bytewise. GPU initial KL is exactly 6.313983917236328.
Seeded26 is reconstructable from immutable actual26 plus saved seed rows;
there is no second full pre-update optimizer snapshot.

One NEW Adam step26->27 uses the same full 1507 FIT history and 256 cached
65537-class source distributions: lr5e-5, betas .9/.999, eps1e-8, no decay,
foreachFalse, global clip1, full core and streamed CPU bank adjoints.
Forward26.734s/backward82.016s/optimizer plus finite inspection8.594s;
gradient norm21.058893954, clip coefficient .04748587246.
Exposure unions[686,845,951,1004,948,911], 12056 selected pairs/site.
All92 gradients/states finite and all Adam steps27.

Every new V column has nonzero gradient and changes at all five sites:
32768/32768 nonzero coefficients/site, 163840 total. New V F64 rank32/site;
the full V matrix rank48/site. New-block L2 norms are
[.0001637204,.0018219927,.0021306513,.0022644875,.0023990644].
New U gradients are zero on this first step with V initially zero; seed rows
and their moments stay exact, as required by the product-rule algebra.
There was one new update, no new source call, no new FIT ID or source label.

## Native endpoint

Before metrics are adopted only after complete initial native identity.
All routes valid/unique, masses nonnegative and maximum normalization defect
1.9371509552e-7; full heads finite and updated18432 integer witnesses exact.

| Cached canonical history | Native KL26 -> KL27 | Argmax disagreement26 -> 27 /256 |
|---|---:|---:|
| DEV self_oss036, 357 positions | 9.0592552034 -> 8.9399476961 | 248 -> 247 |
| FIT magpie022, 1507 positions | 6.3128872933 -> 5.9208975771 | 240 -> 233 |
| DEV magpie036, 1488 positions | 9.1644050239 -> 9.0305439511 | 251 -> 250 |

GPU FIT KL after5.9210734367; native5.9208975771, delta .0001758596.
Three forced histories and two DEV cases are not the full24 DEV/domain screen
or fresh own-history chatbot evaluation. Improvements are observations from one
joint update, not a matched width or seed causal comparison.

Persisted full1507 GPU heads AND raw routes permit independent comparison:
20 head rows exceed1e-4 relative RMS; worst .01794141925; greedy mismatch356;
zero ordered/set ID mismatch calls; mass maximum delta .001996830106.
Strict numerical gate remains FAIL. Prior
[trace](ORIGINAL_WIDE_TRACE_RESULT_20261009.md) establishes the actual26
half-bin mechanism; this packet does not claim to trace every actual27 failure.
Neither small surrogate KL discrepancy explains the large donor-quality gap.

## Durable artifacts and first fault

| Artifact in original_delta_seed_finish_20261009 | Bytes | Raw SHA256 |
|---|---:|---|
| seed_rows.npz | 656592 | 0cec2cd9f1b7378a11d146ad54b07bad47cf1044cc18f738ef6cba3c7f1d60cc |
| seeded26.packed | 520029440 | c1e898e177fe1bc3a64c2e841babf3f32988e5d584d6386540bdb62f30fff500 |
| candidate_seeded27.pt | 8652210298 | ce1baeaf92795c84b88212260ffd8340d69b9fb85100c685c61df476791d7821 |
| candidate_seeded27.packed | 520029440 | 7d419a9ecb8fb5a9f3687388df5f286554166954ccc0c81a31fe866a5bdf2652 |

State schema ORIGINAL_DELTA_SEEDED_STATE_V1 includes model/Adam/RNG/source26/
inherited ancestry/seed ledger/extents/binding, counter27/new_updates1/
completed new FIT ID and optimizer_partial_possible false. Saved before later
evaluation. Independent CPU mmap deserialization confirms92 masters,
721008128 coefficients, all92 disk Adam steps27, new U moments zero,
all new V columns/moments active and packed organ bit identity.

First frozen worker83dd2b2/binding3499a1a0 exited1 after five X captures and
five successful scalar-qualified seed solves. Master names contain `.organs.`;
the seed dictionary uses packed wire names. The verification incorrectly
compared intentionally modified whole x_proj to its parent. Later post-update
U verification had the same key mismatch. No child or optimizer update occurred.
[First fault](original_delta_seed_first_fault_20261009.json),
[launcher failure](original_delta_seed_result_20261009.launcher_failure.json)
and original namespace/code/binding remain preserved.

New completion dbe80da9 fixes those lookups and adopts exactly five saved X
arrays. Five additional closed-form U solves are necessary because old U arrays
were not serialized. Totals: five GPU feature captures, ten closed-form solves,
two native runs and one optimizer update. No source/model prediction or update
replay. Do not claim bit identity to unsaved old U arrays.

Completion binding SHA d49fa8316a3ae164573767ec0de5d5169de0099198c022ea0035cfc132b58c64;
raw [result](original_delta_seed_finish_result_20261009.json) SHA
b71d31a3cd868aa79493ff8d6a1d5512eef37213125013f2569dd63922bb1b0b.
[Terminal](original_delta_seed_finish_result_20261009.terminal.json),
[worker log](original_delta_seed_finish_result_20261009.worker.log),
[stored audit](original_delta_seed_stored_adjudication_20261009.json),
[audit code](../../../benchmarks/native_expert_scaling/original_delta_seed_audit.py).
All56 inputs/25 completion outputs hashed; native metrics recomputed with zero
KL delta; all row comparisons/routes/seed geometry/disk rank independently checked.
Stored-only audit27.593s, no new prediction/refit/update and no separately held
family resource record; this elapsed is not a training benchmark.

## Resources and exact next

First launcher31320/worker17984/create_time1791569625.5206437/exit1 held26.734s,
OS worker12587532288B/launcher30597120B/GPU622361088/650117120B; six old files
30863805B. Completion launcher29596/worker16132/create_time1791569962.6372883/
exit0 held455.563s, raw worker415.984s. Native33064 and22636 both exit0,
held35.781s/34.219s, peaks528064512B/527462400B. No compiler.
Completion worker through-exit OS16866500608B, launcher31002624B;
conservative sum with direct child17425567744B (not simultaneous peak).
GPU allocator2262381056B allocated/2971664384B reserved; not full device residency.
25 new outputs11849242017B; combined31 files11880105822B/held482.297s.
Declared time/OS32GiB/GPU10/11GiB/combined12GiB/log caps PASS.
No own process remains; no source/RESERVED/T4 allocation.

[Exact next](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md): qualify actual teacher
residual boundaries for a finite joint full-history recovery pilot, rather than
another zero-factor repair or numerical trace. The unresolved transfer is the
composition of source history, residual coordinates, nonlinear functions and
selection into an affordable core. Original engine operators remain fixed.
Useful fresh chatbot + same-artifact >=50, useful large n, structured CPU
IDs/mass, physical DRAM and family/~10B/~100B evidence are still missing.
