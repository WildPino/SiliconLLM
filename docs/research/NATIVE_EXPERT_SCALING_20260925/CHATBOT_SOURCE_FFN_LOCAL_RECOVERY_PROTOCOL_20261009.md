# Fixed local FFN function recovery after diagonal calibration

9 October2026. Freeze before learning/price observations. Positive andcorrected
input grids bothfail site0;site23 hiddenbalance improves23.54%. Common-operand
AQ-only error19-24% andternary-only52-56% motivate direct function-aware recovery.
No more diagonal grids selected. This is an offline source-width conversion
stage using original engine ternary/AQ63 work,not a final generic donor runtime.

## Actual initialization and finite forward

Sites0/23,same sourcecaptured FIT2/DEV2. Use selectedidentityS atboth sites and
R folded into site23 coefficients. Source parameter FILES unchanged. Learned
student owns three F32 master matrices andthree row-scale vectors per site:
28,322,816 trainable elements/113,291,264B. Original code/scales andALL initial
inference/STE BF16 outputbits must match the selected retained local sectors/
responses on allfour cases. Initial response is reused as an external witness.

Deployable inference:nearest-even ternaryround/clamp frommaster/positive scale,
AQ63 absmax/integerdot/F32 scales/MUP/SiLU/product/finalBF16. Same4608 FFN rows,
no routing/core/operator addition. Training uses a declared surrogate derivative:
x_proxy=x+detach(q*a-x),w_proxy=code*scale+(master-detach(master)). Return
detach(exact)+(proxy-detach(proxy)),whose finite forward isexact pluszero;
check against actual retained outputs beforeupdates. No dither. This is not
the true derivative of rounding. Signedzero differences would fail the witness.

## Fixed fitting, criteria and retention

Exactly256 updates per site,sites0 then23;no checkpoint/DEV selection. Alternate
FITshort/FITlong;at each visit take deterministic contiguous64 rows cycling
offset=(visit*64)%N,no shuffle. Shortcase uses all18. Lossnormalized squared
outputerror against actual captured sourceBF16 y. AdamWlr5e-5,betas.9/.999,
eps1e-8,weightdecay0,foreachFalse,globalgradclip1,scaleclamp>=1e-8 afterstep.
Allsix gradients positive finite;allweights/moments finite andAdamstep exact.
Save durable stateafter2 and256 withmodel/moments/RNG/history/hash provenance.
Packed final code/scale fields useoriginal pair/row layout,allpairs roundtrip.

Final DEVgates separately reported:equal-case errorratio<=.90,eachcase ratio
<=1.05,each absolute normalizedL2<=.10 andcosine>=.99. Bothsites required for
overallPASS. Relative recovery alone is not localfaithfulness or wholequality.
DEV neverupdates/selects a model. No source generation/forward/native/T4/RESERVED.
Localheld-development sample has two domains/four consumed prefixes,not fresh
generalquality. Wholecomposition andbroader inputs remain necessary afterward.

## Actual price and stop

Capturecost91.438s;localgrid25.656s,correctedgrid35.843s,available inference
prices. Training price is NEW:measure actual firsttwo completed optimizersteps,
persist state2 WITHOUT replaying updates;extrapolate remaining steps withtheir
maximum plus60s allowance forload/evaluation/checkpoints. Stop ifthe projected
family cannot fit fixed600s/60s reserve. This firstfinite learning run itself
prices local source-weight recovery. No fullmodel training feasibility assumed.

FixedOS8GiB/GPU10/11GiB/outputs2GiB/log4MiB/sixcores/nooverlap. Masters+grads+
Adam moments about453MB per site;sites sequential. Four statefiles about1.36GB
plus~28.4MB finalpairs/scales and~8.9MB afteroutputs;large tensors off-repo.
Retainfirstfault/completedsteps andprice state ifresource/finite/forward fault.
Successful local function learning leads to broader/all-site recovery pricing
andNEW whole source-width quality before selectedfunctions/core reduction.
Originalengine fresh same-artifact quality+50,usefuln/CPU LUT mass/DRAM/families
remain unqualified. Do not resume old compact286 orpartial512 contracts.
