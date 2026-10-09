# Next recovery: activation-aware FFN function calibration

9 October2026. SELECTED investigation/UNIMPLEMENTED/UNEXECUTED;not an acquired
conversion or a long training authorization/budget. [Complete arithmetic control](CHATBOT_SOURCE_FFN_CONTROL_RESULT_20261009.md)
fails before selection/core reduction. [F32 positive control](CHATBOT_SOURCE_FFN_CAST_RESULT_20261009.md)
retains ALL16 actual source generation sequences andhas twoFITKL<.0005.
This fixes the next uncertainty:can the ternary/AQ63 function approximation
be improved without adding dense deployed work or changing the other organs?

## 1. Capture the missing local operands,then separate package effects

Existing8808 logits andsix integer-dot witnesses do not contain full floating
FFN inputs/outputs. Do not invent them from logits or label a weight norm as
function error. First implement/price capture ofactual original-source FFN
x/y on four original FORCED cached trajectories:the same metadata shortest/
longest FIT2 plus metadata shortest/longestDEV2,sortingbyteacher-forcing length
thenID. No new originalsource reply generation/cropping/RESERVED query.
Actual selected cases:FIT rewriting008(18 labels),magpie022(256);DEV rewriting035
(12),magpie036(256). Observe site0 plus site23,the largest preflight down
weightL2(.4731844370),metadata-only choice.542 labels*2 sites*2(x/y)*2048*2BF16
bytes=8,880,128B raw hidden payload. Preserve oldsource parameter bytes/operator
order/casts andcompare newlyobserved source logits toretained source rows;
stop on an unexplained numerical mismatch. Store x/y,IDs/positions/dtypes/
source/capture hashes. Repeated forcedcalls acquire a NEW hidden-variable
packet needed for calibration,not repeat quality observations for confirmation.

At identical captured floating x,compare actual FFN responses fororiginal
BF16,F32-only,originalF32weights+AQ63,ternaryweights+unquantizedfloatingx,and
thecomplete trit/AQ63 package. Keep source MUP positions/finalcast explicit.
Use case-balanced normalized outputL2 plus direction/scalar residuals;retain
every output. This local decomposition is not a whole-layer or trajectory
certificate. Interactions are measured,not inferred by adding KLs.

## 2. An exact real-algebra freedom to test before broader recovery

For F(x)=b D[(Ux) elementwise SiLU(aGx)],positive invertible diagonal S,R give

    x'=Sx; G'=G S^-1; U'=R U S^-1; D'=D R^-1
    b D'[(U'x') elementwise SiLU(aG'x')] = F(x).

R commutes withthe elementwise product;the argument ofSiLU remains identical.
This is an identity inreal arithmetic,not guaranteed originalBF16 bit equality.
Verify the unquantized numerical response before ternarizing. Quantization is
not invariant under S,R:their choice may reduce action error even though
the exact unquantized function is the same. No success is presumed.

Candidate finite grid:input strengthalpha andhidden strengthbeta in{0,.5,1},
nine pairs including exactS=R=I baseline. FITcase-balanced channelRMS statistics
andoriginal columnweightRMS define normalized positive balances;clip factors
to[1/16,16]. Use S proportional to(inputRMS/gate-up-columnRMS)^alpha andR
proportional to(down-columnRMS/hidden-productRMS)^beta,each normalized by its
geometric mean. Bind exact zero-floor/order/dtypes/formulas beforeobservations.
Recalibrate trits/row scales ineach candidate using theexisting8-round rule.
Select each site's candidate by FITcase-balanced normalized outputerror only;
DEV does not select strengths. Fixed prospective local stop:>=10% DEV error
reduction against unbalanced package andnoDEVcase worsens by>5%. Treat a failed
finite grid as failure ofthis construction,not of all ternary conversion.

Deployment effect:S needs one elementwise D multiplication before the FFN's
AQ63;R folds into up-row scales/down coefficient conversion. Preserve the
shared input/LUT across experts at a site;do not silently give eachselected
expert its own inputquantizer/LUT. No new dense matmul or increased active row
count is proposed. Actual C integration/rate must still be measured if adopted.
Additional trit planes/shared-private functions are alternatives with separate
active-byte/work budgets,not automatic responses to this failure.

## 3. Price and decision before spending

First action:implement thecapture/local evaluator with immutable source and
sector reuse,calculate exact hidden-packet sizes from selected cases/sites,
andfreeze caps/commands/hashes before source calls. Proposed separate caps
capture600s andfinite local calibration300s,with60/30s reserves,8GiBOS,
10/11GiBGPU and512MiB outputs each. These are proposed envelopes,not a measured
price or a binding;the actual code/schedule must be priced before execution.
The completed cast control274rows+16generations cost75.829s,which supplies
an available starting inference price,not a training-throughput claim.

If localbalance improves held-development functions,extend calibration toall
FFN sites withbroaderFIT operands andthen run NEW whole prefix/generation/
own-history controls before composing selection/core compression. If itdoes
not,price function-aware FFN weight/scale recovery (one FFN/block atatime first),
using original teacher outputs andfinite retention gates. Source-shape full
FFN masters/gradients/Adam are large;no full model optimizer feasibility or
monthT4 recipe is established. Communicate actual T4 reason/budget/stops beforeuse.

Do not replay complete48 full package or16F32 generations;reuse their bytes.
No current worker is running. Existing compact broad24/Adam318 checkpoint and
allold native failures remain controls. Final useful conditional functions,
compact source-state transfer,new same-artifact C quality+>=50,useful n/CPU LUT
winners+mass/actualDRAM andother families/scales remain open.
