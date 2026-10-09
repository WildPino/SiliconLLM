# True FFN operands and finite diagonal calibration

9 October 2026. Frozen before new operand capture or local responses. This
implements the [selected recovery investigation](CHATBOT_SOURCE_FFN_RECOVERY_NEXT_20261009.md).
The previous full ternary/AQ63 intervention failed; the F32-only control
preserved all16 source generation sequences. Keep the original engine target.

## Capture: missing evidence, unchanged source function

Original Falcon1.5B revision80ebc50d,all411 parameter objects unchanged,
original cached forced execution and qualified source SSD helper. Four adopted
trajectories chosen by metadata only:FIT rewriting008/magpie022 andDEV
rewriting035/magpie036,18/256/12/256 labels. Source-generations0. Observe sites0
and23 (largest preflight down-weight relativeL2). Save actual BF16 FFN x/y for
the last prefill position and every subsequent forced step,with positions/IDs.
542*2sites*2(x/y)*2048*2bytes=8,880,128B raw operands. Compare ALL542*65537
logit coordinates to original saved BF16 bits,stop at first mismatch. Hooks
must only read tensors. Keep completed case packets and first fault on failure.

Actual F32 control274rows+16generations cost75.829s. Four cases542 rows should
cost abouttwice its forced part,plus two longer prefills/source load/1084 hook
observations. This is an estimate,not captured timing. Fixed capture600s family/
60s reserve,OS8GiB,GPU10/11GiB,output512MiB/log4MiB,six cores/no overlap.

## Local response decomposition

Load ONLY selected original FFN tensors from pinned safetensors;source model
is not executed during calibration. Same captured floating x/actual BF16 y,
original MUP factors. Compare originalBF16,F32,fullF32weights+AQ63,tritweights
withoutAQ63 andtrit+AQ63. F32-internal variants finalcast toBF16 before scoring.
Trit linears multiply dot byrow scale outside the reduction;AQ63 scales follow
that dot/row multiplication,matching existing EngineFFN. OriginalBF16 local
batching may differ numerically from the cached model GEMM;measure that baseline
against actual captured y rather than assuming bit identity. Decomposition is
a common-operand FFN check,not whole-layer/state/generation equivalence.

Report normalized Frobenius L2,cosine,best scalar amplitude and its residual,
retain all computed finalBF16 outputs. No conversion of weightL2 into knowledge
fraction. Recompute baseline trits/scales with existing eight-round row rule
and require ALL selected codes AND scale bits match actual sector9aa6f319.

## Exact real-algebra freedom, finite numerical qualification

For F(x)=bD[(Ux) elementwise SiLU(aGx)]:x'=Sx,G'=G/S,U'=R*U/S,D'=D/R,
with positive diagonalS,R,preserves the real function. F32 comparison before
the finalBF16 cast must have relativeL2<=1e-5 on every observed trial FIT and
selected DEV response. This verifies the sampled finite transformation only,
not originalBF16 bit invariance or unknown inputs. Save FIT F32 identity outputs.

FITcase-balanced statistics inF64:inputRMS=sqrt(mean_cases(mean_labels(x^2))),
gate/up columnRMS=sqrt((mean_rows(G^2)+mean_rows(U^2))/2),hidden-productRMS
from original unquantizedF32 z,down columnRMS=sqrt(mean_rows(D^2)). Floor each
RMS at1e-12. logS=log(inputRMS)-log(gate/up columnRMS),logR=log(down columnRMS)
-log(hiddenRMS),subtract each vector's mean. For alpha,beta in{0,.5,1},
S=F32(clamp(exp(alpha*logS),1/16,16));R analogous. Normalize before clipping;
no second normalization. Nine candidates including exactidentity baseline.
Apply existing eight-round row calibration to transformed F32 coefficients.

FIT-only selection:argmin mean over two FITcases of ||candidateBF16-y||/||y||.
Ordered alpha thenbeta;strict less wins,first candidate wins ties. Do not evaluate
candidate DEV errors to select. Evaluate only the selected candidate on DEV.
Fixed gates per site:DEVcase-mean error ratio<=.90 andevery individualDEVratio
<=1.05 versus unbalanced full package;both sites required for overallPASS.
Preserve all FIT trial outputs/metrics andselected DEV/packed projection/scales/
S/R. No gradient/optimizer or original source generation. Do not rerun complete
48 whole conversion or16F32 generations for confirmation.

## Price, interpretation and next decision

Two selected FFNs:56,623,104 original coefficients total;F32 weights226,492,416B.
Original codes/scales are loaded/calculated sequentially per site;each local
batch is atmost256 rows,not a full source model activation graph.18 FITtrials,
four cases for five decomposition modes,selected DEV andfinite identity checks.
Use fixed300s calibration family/30s reserve,OS8GiB,GPU10/11GiB,512MiB outputs/
4MiB log. This is an independently bounded local computation,not a training
throughput certificate. Stop on resource/finite/baseline-sector/identity fault.

If both sites improve,price broader operand calibration plus NEW whole output/
generation before selection/core compression. If this finite construction fails,
close it in this scope andprice function-aware weight/scale recovery. Local
success does not preserve source chatbot capability. S addsD elementwise
multiplications before the shared AQ63/LUT;R folds into converted weights/scales,
no new dense matmul or more active FFN rows. Actual C/rate cost remains unmeasured.
No native/quality/T4/useful-n/DRAM/family admission from this local experiment.
