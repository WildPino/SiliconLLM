# METH-395: full-dimensional I8/A16 router with exact candidate refinement

Prospective after394 rejected/retained2fa0cf2. New variable: precision and
addressed weight bytes, preserving ALL dimensions/ALL candidate normalization.
Reuse356 exact I8/A16 projection arithmetic and393 actual router inputs/full
scores.394 low-rank classifier lost competition/probability; no rank reduction
or source-only SVD here. Local quality thresholds remain exactly394; cost proxy
now measures changed weight bytes rather than claiming fewer coefficients.

Bind controller/protocol/helper/393/394/338/380 records before observations;
independent368 manifest parser and24 actual source F32 router segment hashes.
Freshly hash ALL384393 consumed-cohort teacher/natural traces and validate exact
headers/indices/phases/input/score shapes. Entire payload hashes inherited393,
size/mtime unchanged, actual router segments freshly match original source.
Independent full-weight actual-input score L2<=1e-6 and exact selected IDs.

Convert each original classifier row: F32 absmax/127, nearest-even I8[-127,127],
zero-row scale1; all scales finite positive. Actual F32 inputs: absmax/32767,
nearest-even A16[-32767,32767], zero scale1. SAME356 mathematical I64 integer
sum, F64 rowscale*inputscale order and final F32 projection. Independent Python
integer dots for12 deterministic query/row pairs per bank/mode; F64 integer-
valued BLAS sums exact within768*127*32767<2**32<2**53, every result checked
integral/bounded. Save/readback/hash actual I8 codes/F32 scales. No C change,
new training or altered expert output dimensionality.

Fixed m=0,1,4,8,16. m0 selects original lowest-index maximum approximate score;
others stable approximate ranking followed by exact original captured candidate
scores/lowest-ID ties. Replace ONLY those candidate scores; normalize over ALL
mixed scores. F32 subtraction/F64 exp rounded F32/F64 sum/final F32 probability;
stable maximum centering with selected numerator if an unrefined estimate
outranks exact candidates. Actual selected identities/probabilities compared
to original exact full-score router on its existing inputs.

EACH twelve banks/BOTH modes: identity agreement>=.9999; absolute-relative
selected probability p95<=.01 AND maximum<=.05; addressed WEIGHT-byte proxy
(n*768+4*n+4*768*m)/(4*n*768)<=.5. This replaces394's coefficient proxy because
format, not dimensionality, changes; report coefficients ratio1+m/n explicitly.
Includes full I8 codes/scales and candidate F32 rows, excludes activation reads,
ranking/normalization/dispatch; not physical DRAM or measured cost. Original
F32 routers remain stored with added I8 representation, both counted.

Only a source whose SAME m passes ALL24 bank/modes is eligible for separately
frozen native numerical/NEW original-primary whole-quality and actual rate
work. Diagnostic queries are already consumed, not new whole-model heldout
quality. No pooled override, optional grid/threshold adjustment or unchanged
rank trial. First apparatus failure retained; candidate rejection valid.

Budget MAIN20min/RSS<=8GiB, expected<=2min. NumPy2.4.6/OpenBLAS1 thread actual
readback/library SHA; all native jobs exited. Fresh representation/result paths,
no model/timing/download/GPU/T4 overlap. Five apparatus gates: trace/source
bindings, independent full score controls, representation readback and exact
integer controls. Source profile/whole-model quality/speed/useful>256/actual
LUT/DRAM/another family/~100B remain unqualified for this new router.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth395_switch_router_integer_screen.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth395_switch_router_integer_screen_result.json
```
