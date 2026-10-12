# Mixed correction at the already qualified BF16 boundary

13 October2026. Native V1 COMPLETE/full audit but NUMERIC FAILURE retained.
One control at broad_dev_smol_summarize_034:AVX_F32 normalized input relative
1.2641131e-4>original1e-4. All48 scan/finalstate/projection controls qualify,
all92,352,512 F32 raw score words bitexact scalar reference;controlled F64
normalized vectors/rawscores/BF16 outputs all reproduce direct CPU reference
exactly. Local source-quality budgets PASS,both paths/modes. Timing V1 not run.
This does not identify which individual F32 decode/gate/RMS rounding dominates;
the evidence isolates the discrepancy before the final output contraction.

Fixed correction,no weight/rank/refit/DEV model selection:use the already
qualified F64 decode/gate/RMS/normalized-BF16 boundary,then original AVX F32
dotf for output projection. It replaces only one controlled-branch contraction
expression in a separate C source;original V1 files/results remain immutable.
Float32 projected inputs/B/C and state384x96 remain. Full source gate/skip and
nonlinear generators/conv widths remain retained. Both fixed norm modes tested.

Let q=BF16(gamma*u64/r64). Proposed output isBF16(dotf_F32(O,q));O/source
coefficients BF16-promotedF32. This combines a qualified input boundary and a
separately bit-qualified F32 contraction. Their actual composition still
requires full qualification;separate passes alone do not admit the new artifact.
No attempt to relax1e-4/1e-3 thresholds or substitute successful histories.

All48 actual new native scans/final states/B/C/input projections,all two
normalized vectors/rawF32 scores/BF16 outputs stored. Fixed numeric controls:
projections1e-6,scan/finalstate1e-4,smooth normalized/rawscore/BF16output1e-4,
scalar score reference ALL words bitexact,old bridge1e-3 retainsall flags.
Independent exact exported bits/source/maps,every CPU384x96 state timestep,
all controlled physical inputs/output contractions/every rawscore bit,
metrics/domain/gates/hashes before-after. Audit complete even failed flags.

Local DEV budgets unchanged:mean/centered<=.1,worstcase/worstdomain<=.2,both
modes. Actual fixed mixed path only (flags1),unused original F32 branch remains
in the compiled derivative but is not called/admitted. No new fit/source/T4/GPU/
RESERVED/whole native chatbot. Native single thread,affinity0,OS6GiB,
prepare/qualify900s,audit1200s,compile45s/DLL,output2GiB fresh ordered E namespace.
Freeze/bind before execution;timing separate only AFTER full audit PASS.

Future timing flags5 selects mixed ownnorm-only (F64 norm,AVX output),not V1
F32-norm flags6. Previous timing source/protocol is frozen but NEVER LAUNCHED.
Count/measure full retained generator/gate/conv/skip/norm/out costs. No useful n,
causal chatbot,multiple family,actual expert DRAM or same complete artifact50.
