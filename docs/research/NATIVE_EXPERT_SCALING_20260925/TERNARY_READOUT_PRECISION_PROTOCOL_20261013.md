# Fixed full-size readout balanced ternary precision preflight

13 October2026. Source original O2048x3072 BF16 weights decoded F64, sites
0/12/23. Actual qualified native own-normalized3072 BF16 Z/raw F32 Y/BF16
output baseline and original source BF16 output, all48 histories/site.
Source-weight-only p4/5/6 artifacts for ALL3 sites generated before consuming
any response. No calibration of scales/FIT optimization/new source/teacher/
GPU/T4/native call/RESERVED or activation quantization. Preserve original
global response dimensions while testing precision, not a small-rank fit.

## Frozen F64 source quantizer and exact integer representation

p4/5/6, Q=(3^p-1)/2=40/121/364. Row scale s=max(abs(O))/Q computed F64;
zero row means s0,q0 explicitly. Ratio O/s computed F64 with explicit zero-row
masked division. q=nearest-even of the actual F64 ratio, clipped[-Q,Q], int16.
This defines the numerical quantizer, not exact rational RNE of an unrounded
scale. Retain F64 ratio half ties/clipped coefficients/zero rows and coefficient
error. Bound max(abs(O-s*q)/(s/2))<=1+1e-12 on nonzero rows;no hidden epsilon.

Balanced digits by remainder mod3:0->0,1->1,2->-1,then integer carry(q-digit)/3,
p steps/exact remaining0. Group FIVE digit columns,zero padding final3 beyond
3072;index sum_k(digit+1)*3^k in[0,242],uint8 layout[p,615,2048]. Decode every
pattern/digit independently and check EXACT reconstruction all q coordinates,
all pad digits0 and plane digit histograms. No rank truncation constraint.
Store q/int16 and codes/scales/spectra for audit; q is a diagnostic redundant
copy, not part of prospective deployed packed-byte traffic claims.

Full direct SVD original O and each q*s, all2048 singular values, robust
numerical rank sigma>1e-10*maxsigma, no positive-mode floor or inversion.
Independent column-block Gram/eigvalsh comparison of every squared singular
value,max absolute discrepancy/maxsigma^2<=1e-8;raw Gram negative counts/minima
retained,negative/max<=1e-10. Direct spectra are finite numerical diagnostics,
not exact-real rank certificates. Source spectrum computed once/site and its
three stored copies compared exactly;avoid unchanged duplicate work.

## Actual input response and held-out candidate policy

Dense control=F64 O Z. Quantized reference=(F64 integer-q dot Z)*F64 row scale,
then explicit F32 cast/BF16 round-nearest-even. Do not fold scale before the
integer dot and pretend equal rounded contractions. All input BF16 values
preserved, no int8 activation step. Original native F32 score reference and
original mixer-output semantics (no extra source multiplier) unchanged.

Every arm/all histories: whole/centered/label relative RMS for raw_vs_native,
BF16_vs_baseline and BF16_vs_source;domain/worst/means/bit mismatches. Each
case raw extra error<=.01 AND BF16 extra error<=.01;original donor gates
mean<=.1/worst<=.2/mean-centered<=.1/worst-domain<=.2,FIT/DEV separate. Dense
real control vs native each case<=1e-6. Conservative output coefficient-bound
norm and coefficient-to-response amplification retained as cancellation
diagnostics. Quality failure never stops full evaluation/independent audit.

FIT candidate rule BEFORE observation: smallest p passing BOTH per-case extra
gates and all donor FIT gates across ALL3 sites. DEV verifies ONLY that candidate,
cannot upgrade after observation. Keep all p diagnostics and null selection if
none qualify. No whole-model/OWN-history/causal quality admission from these
source-input boundary tests. Full trained generator/recurrence/maps unchanged.

## Complete independent audit, storage and bounds

Audit does not import producer. Manual sign/floor/fraction/parity nearest-even
of independently decoded source ratios;all source/native parameter bits and
q/scales/packed digits/padding/histograms/metrics/spectra. Full32-row independent
q-digit reconstructed promotedF64 contractions, separate low-bit BF16 RNE,
ALL415,586,304 stored BF16 output words and EVERY unsaved raw-score metric/
bound/flag/aggregate/FIT selection/DEV decision. Raw F64 response arrays not
retained,no claim all producer raw-score bits are archived/verified. Array
scale reproduction<=1e-12;metadata abs(delta)/max(abs(ref),1)<=1e-10;
independently rounded BF16 smooth relative RMS<=1e-4,bit mismatch counts also
retained. No claim original AVX F32 reduction bit equality. Before/after full
extent hashes of source/code/protocol/runtime/data/foreign files.

Prepare/run900s/audit1200s/OS6GiB/new orderedE1GiB. CPU0..5,BLAS/OMP/MKL6
before interpreter. Principal actual resolved NumPy ABI/linalg/BLAS/Python/
psutil pinned (not a complete transient DLL tree guarantee). All source,
native/source-output packets/provenance/exports/old M398--400 evidence pinned.
New9 parameter NPZ+144 BF16 prediction NPZ ~1.001GiB DECIMAL bytes (~955MiB),
under1GiB binary cap;no duplicate raw floating output histories.

Prior M398 full-width input-ready synthetic-code decoder~.138--.165s and
M400 blocked variant~.119--.125s per measured fixture/rep median rejected
unchanged geometry. M399 says builder AND rows independently exceed14ms.
Different Q6 source/synthetic-code representation;fingerprint-only aspects
not exhaustive byte proof,not a learned chatbot or Falcon readout bound.
Use these as cost warnings,do not repeat unchanged old campaign. Source
engine.c actual2-trit int8/pshufb/nibble path differs from proposed5-trit
floating table backend. No borrowed throughput or bit qualification.

If accurate selected p passes DEV,freeze actual C LUT backend/accumulation/
source BF16 precision/native primitive controls and physical DRAM/cost on same
artifact. Full F32 group table597780B/multiple-plane lookup work must count;
smaller packed bytes alone do not guarantee speed. Source CHATBOT pipeline/
conditional useful ternary experts/compact causal core/other families/scales/
conversion masters/OWN histories/SAME useful>=50 remain required.
