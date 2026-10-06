# METH495 — one fixed hybrid function/control fit and physical evaluation

Prospective, before any 495 control, source feature, fit or candidate result.
Goal ACTIVE/INCOMPLETE. Previous goal turn is PROGRESS: 494 compiled and
independently admitted all source-informed priors and physical cost contract.

## Decision and scope

Resolve whether the fixed 494 hybrid geometry can reproduce the actual 493
weighted targets Y=pF, with cheap learned choice. Reuse all original bank 11
UIDs/occurrences and priors, without replaying source FFNs/models or earlier
scientific namespaces. One fit, no hyperparameter/rank/update/ID/checkpoint
sweep. A failed fixed local gate closes this recipe. A pass permits composed
476 candidate evaluation, not whole quality/speed or goal completion.

All 17,540 UIDs, 19,962 occurrences, 128 IDs, 11,721 development inputs and
5,819 consumed validation inputs remain. Development is owner roleflags&5;
validation roleflags&2, disjoint. Validation is diagnostic, not fresh evidence.
All actual source controls are accepted; preserve this domain, do not narrow
to the old 107 ready IDs. ID 0 remains weight-derived, with no development
examples and explicitly unsupported global amplitude initialization.

## Fixed physical operators

494 packed A/scales and weighted quantized L stay fixed. Source F32 input
quantizer: F32 maxabs/32767 (zero ->1), F32 divide then RNE integer/clamp
[-32767,32767]. Native must reproduce admitted source qx/alpha BYTE for all
UIDs. Group4 table has 192*81 I32 entries; packed code trits are {-1,0,1}.
Sum table entries in I32; phi=abs(roundF32((F64(dot)*F64(sigma))*F64(alpha_x))).
Quantize phi by the same A16 rule. H=[F64(qphi)*F64(alpha_phi);1].

L dot has eight I32 SIMD lanes, each bounded by (768/8)*127*32767<2^31;
reduce lanes in I64 (the total can exceed I32). B at width 512 has both lane
and total I32 bounds; same I64 reduction is exact. Round each dot*rowScale*
activationAlpha from ordered F64 products to F32. Add L+B in F32, then bias
in F32. Rejected control skips the entire residual; actual source set has
accepted=1. G already estimates pF, with no second p multiplication.

Keys use raw source x and rounded phi, not dequantized H. Stored F32 key
coefficients; F64 eight-lane ordered dot over 1280 features, then reduce
sum_{j=0..3}(lane_j+lane_{j+4}), add F32 bias in F64 and round logits F32.
Row/column winner ties choose smallest axis index; ID=16*row+column.
Stable softmax: F32 exp(F64(logit-max)); ordered F64 sum of these F32 values;
selected axis probability F32 division; F32 product of axis probabilities.
This probability is candidate control diagnostics, not source-mass fidelity.

## Fixed development learning

For each nonempty expert, physical fixed L output determines R=Y-L, in F64.
Use mean data loss and prior penalty .01*trace((C-Cprior)Kreg(C-Cprior)^T),
Cprior=F32 amplitude times serialized F32 C0, multiplied in F32 then widened.
One Cholesky solve:

```
Gamma = H H^T/N + .01 Kreg
T = R H^T/N + .01 Cprior Kreg
C = T Gamma^-1.
```

Serialize C F32; weighted initial L is unchanged BYTE. Empty ID 0 retains
its entire physical B/bias prior BYTE. Quantize fitted B rows by F32 maxabs/
127 (zero ->1), F32 divide, RNE integer/clamp[-127,127]; bias F32.
No target val fitting or use of validation for any choice.

Normalize the 1280 key input features by development RMS in F64 (zero RMS=1),
append bias 1. Initialize F64 theta from 494 F32 centroid coefficients times
RMS, bias unchanged. Loss is mean(CE_row+CE_col) + (.0001/2)*||theta-theta0||²,
including ALL coefficients and biases. Thus the real objective is strongly
convex with modulus .0001; no optimum assertion from finite Adam.
Full development 128 Adam updates, LR .01, beta .9/.999, eps 1e-8; bias
correction by beta**step. Start m=v=0. Retain theta/m/v at initial and all
128 updates, plus loss/gradient diagnostics at final update. No checkpoint
selection; serialize theta_final/RMS to physical F32 coefficients (bias RMS1).
Report the computed real-formula gradient gap expression ||grad||²/(2*.0001)
with floating-point/finite-update limitations, not as donor quality.

## Wire formats

All integers little endian; arrays C-order. Headers <8sIIQ unless stated.
N=17540, D=768, r=512. Input rows M495INP1/4620/768/N:
expertU32,acceptedU32,alphaF32,x[768]F32,qx[768]I16.
Features M495FEA1/6148/512/N: phi[512]F32,qphi[512]I16,
alpha_phiF32,physical_L[768]F32. Predictions M495PRE1/6248/768/N:
chosenIDU32,selected_probabilityF32,logits[24]F32,oracleG[768]F32,
coupledG[768]F32. Native consumes both physical bank and source inputs; prior
features use M494BNK1, learned predictions use M495BNK1. Bank schema/size
otherwise exactly 494 (127232136 B).
Fitted real coefficients M495FIT1/1575936/513/128. Optimizer history
M495OPT1/737856/1281/129: theta,m,v [24,1281] each F64.
Key geometry M495KEY1/10248/1281/1: feature RMS[1281]F64, biasRMS1.

## Complete reports and eligibility

UID reports for development/consumed-validation, all expert exposure counts,
rare classes (development N=0,1..4,5..15,>=16). Occurrence reports for all
192 books*2 modes*2 controls (768 views), role/mode aggregates, all 128 IDs
with source/candidate exposures. All denominators derive from admitted
493 UID and occurrence links, not rounded-input duplicates. Report oracle-ID,
coupled, decision-only and source energy; zero denominators explicitly null.
Use squared-energy sums in F64 and relative RMS sqrt(error_energy/source_energy).
Recipe gates apply to each of six role/mode occurrence aggregates with data:
coupled RMS<=.01 and ID fidelity>=.999. Report all additional views regardless
of PASS/FAIL; do not choose a favorable subset. No source-probability gate.

Compiler retains full physical feature/prediction files, verifies them BYTE
with independent Python integer-as-exact-F64 products (all integer products
and intermediate sums below2^53), ordered keys and literal RNE arithmetic.
Auditor imports neither learner, its math nor native source implementation.
It reconstructs source input/feature joins and physical operations, audits all
128 fit normal equations and all 128 Adam transitions with distinct block
accumulations, recounts complete metrics, and checks every exported byte.
No prior Gaussian solve/control replay. Numerical audit tolerances fixed:
fit F32 equation envelope E32*abs(Gamma)+1e-7*max(1,abs(T)+abs(C32)*abs(Gamma));
E32=abs(C32)*2^-24/(1-2^-24)+2^-150. RMS audit uses blocks137 and abs/relative
tolerance3e-12. Optimizer transition abs/relative tolerance2e-10;
reporting sum comparisons relative1e-10/absolute1e-8. These are
numerical controls, not formal interval proofs or task quality certificates.

## Costs, resources and sole execution

Metadata builder90s/256MiB. Main1200s and audit1200s, each combined parent plus
native/descendant OS peaks<=1536MiB, CPU0/BLAS1. Each child<=120s within total.
Each new output directory<=768MiB, all495 outputs/records<=1GiB, raw<=8MiB.
No GPU/download/install/source FFN/model run. Existing qualified clang builds
only the new candidate physical evaluator; do not modify engine.c.
494 did not compile: re-admit the existing historical toolchain snapshot,
5353 compiler/header/library files (402562552 B), against their qualified hashes.
Unused source FFN payload extents remain historical dependencies, not current
495 numerical inputs; the qualified prior bytes are the current input.
Include hash admission, compilation, new controls, source joins, native feature
generation, fit, all Adam updates, native prediction, verification and complete
record writing in time/memory/output accounting. Watchdog through record write,
late resource receipt, actual sole tool/session/PID/create-time/exit and Windows
Event1000 evidence for parent/children. Retain first faults/partials before
numbered repairs; no namespace rerun.

New controls before source processing: all81 ternary group encodings with a
new fixed four-coordinate input, scale2 A16 half-even case, width768 I64
and width512 I32 endpoints at code32766/weight126, CPU/FPU requirements,
negative magic rejected without output. Main/audit independent exact integer
proofs and new ridge/Adam dyadic analytic fixtures qualify implementation.
No old completed control invocation. All code/schema/runtime/input/protocol
committed before metadata binding, and binding committed before main.
