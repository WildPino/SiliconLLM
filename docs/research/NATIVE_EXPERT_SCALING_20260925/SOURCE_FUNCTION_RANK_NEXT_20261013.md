# Source-function active cost: FIT response-rank envelope

13 October2026. **COMPLETED / FULL INDEPENDENT AUDITPASS.**
[Qualified result](SOURCE_FUNCTION_RANK_RESULT_20261013.md),
[closure](source_function_rank_session_closure_20261013.json).
Original proposal and equations retained below. Fixed global low ranks show
large FIT residuals;next observable-route conditional output bank is proposed
in [the new plan](CONDITIONAL_OUTPUT_FUNCTION_NEXT_20261013.md). No compressed
model/OWN-history chatbot/quality-speed or useful-n admission.

[Actual-input source function/native temporal replay](TRUE_INPUT_FUNCTION_AND_TEMPORAL_RESULT_20261013.md)
is COMPLETE/BOTH full independent auditsPASS and all field/state/output gatesPASS.
Use actual qualified H and actual-input native own-norm BF16/raw-score packets
at0/12/23. Historical inferred-input and BF16 failures remain retained. No
compressed model/rank or OWN-history quality admitted. Before implementing,
recognize the additional initial-layer token-only compiler alternative below.

## Uncertainty and connection to the goal

The trained full3072/state96 native scan costs about59us/history position in the
qualified separate timing run;normalization+output about1175us,raw dense input
GEMV separately about2515us. Keep these workload boundaries explicit;they are
not a measured joined-layer or chatbot rate. The next conversion question is
how much of each expensive LINEAR input/output response admits a shared low-rank
factor before conditional residual functions are needed. Preserve the2048 hidden
interface,all3072 recurrent channels,fixed state maps and conv/gate/norm semantics
while studying operator rank. A rank reduction here does not prove a compact
causal hidden encoder or conserved pretrained conditional capacity.

Reuse the captured and qualified actual BF16 H and actual source in_proj,
plus the qualified own-norm BF163072-channel inputs/raw F32 output responses.
Only24 FIT histories/site determine covariance,normalization and bases.
Give each history equal weight:sum_i Gram_i/history_i divided by24;do not let
long documents dominate or use DEV to choose rank/scales. No source forward,
new teacher response,RESERVED,gradient fit or ternary quantization in preflight.

## Input generator geometry

H is AFTER source ssm_in_multiplier,with2048 coordinates. W has6704 rows
(gate3072,x3072,B256,C256,delta48),actual BF16 source coefficients. Uncentered
C_H = mean_case(H.T H/history) includes the mean as part of source response.
Choose a positive per-section scale from its FIT raw LINEAR response energy
E_s=trace(W_s C_H W_s.T),S_out[s]=1/sqrt(E_s). This balances sections in the
declared linear objective;it does not equate nonlinear postconv field errors.
Zero energy is an explicitly unsupported section,not a hidden epsilon/floor.

Let L=C_H^(1/2),J=S_out W L. The squared singular values of J are the eigenvalues
of the2048-square K=L W.T S_out^2 W L;no6704-square response matrix is needed.
For a rank-r operator approximation,finite-FIT weighted linear squared-error
minimum is the sum of the discarded squared singular values. Report full
spectra,relative tails at fixed ranks128/256/512/1024/2048 and numerical
residuals/orthogonality/negative eigenvalues. Do not silently floor negatives
or invert unresolved tiny singular values. Exact numerical qualification,
zero-mode policy and caps must be fixed in an implementation protocol.

If retained left singular vectors U_r are numerically qualified,one candidate is

    A_r = S_out^-1 U_r             (6704 x r)
    B_r = U_r.T S_out W            (r x 2048)
    approximate_raw_projection = A_r B_r h.

This minimizes the stated real linear objective on the observed FIT proxy;
explicit BF16 projection/MuP/conv4/SiLU/delta stages still need full replay.
Actual source MuP has two BF16 rounding stages;folding a scalar into W is
not an exact equality of those rounded computations. If an inferred-input control is evaluated,its covariance remains explicitly
approximate;the primary compression calibration requires actual qualified H.

## Output geometry

Let Z be own-norm BF16 inputs and O the actual source2048x3072 out matrix.
The stored raw AVX F32 responses Y permit equal-case2048-square C_Y=mean(Y.TY/n).
Its spectrum gives the optimal rank-r OUTPUT-SPACE projection of those finite
stored responses. With leading output eigenvectors U_r,the implementable
reference factors are U_r and U_r.T O. Because the target uses stored F32
contractions and implemented factors will introduce new rounded contractions,
the projection-error envelope is not an exact error certificate for that new
kernel. Record the difference via actual replay before admission.

Fixed ranks128/256/512/1024/2048 are diagnostics;DEV cannot select an alternative
after observation. FIT response rank and ordinary source-weight rank answer
different questions. These finite-response envelopes constrain only this
declared globally linear approximation class,not nonlinear/conditional experts
or future causal information generally.

## Cost and decisions

At rank256,input coefficients are256*(6704+2048)=2,240,512 versus13,729,792;
output coefficients256*(2048+3072)=1,310,720 versus6,291,456. These counts include
both factor matrices but not scales/workspace/gates/conv/state/head/glue. They
are arithmetic,not throughput:measure real new kernels and actual DRAM later.
Full-rank control is numerically useful despite a larger factorized layout.

If useful low ranks have small FIT response tails,freeze a FIT-only candidate
rule,explicit dtype/BF16 boundaries and per-case held-out field/temporal/source
budgets,then qualify that actual generator/readout and its native cost. If
tails require expensive ranks,document this finite-data linear-class limit
and move to input-conditioned low-rank residual functions/ternary expert
banks around a cheaper shared part. Selection/routing IDs AND masses and
physical expert DRAM must be counted;mere duplicated experts do not transfer
additional donor capacity. No endless unchanged linear fitting.

Proposed preflight producer900s/audit1200s/OS6GiB/new E arrays<=1GiB. Exact code,
spectral uncertainty controls,data/source/runtime hashes,storage/launch remain
UNIMPLEMENTED/UNBOUND. No GPU/T4 or hours allocation now.

After function compression,join source-informed layers and test OWN histories/
source categorical and chatbot tasks. Cheap local operators do not establish
the full pipeline. Useful n/multiple donor scales/conversionO(n) masters and
gradients/same useful artifact>=50 batch1 tokens/s remain explicit requirements.

## Initial discrete-input alternative,not a rank-class obstruction

The [complete token-input witness](source_token_input_witness_20261013.json)
confirms every repeated token has identical actualH0 BF16 bits across this
corpus (22547 rows/3816 words/18731 repeats/zero mismatches);pinned source
embedding->RMSNorm->SSMin graph is token-only atlayer0. A full65537x6704BF16
gate/raw-conv/delta table would878,720,096B/13,408B per row with32-bit IDs.
It trades redundant storage for direct known-input lookup,same trained capacity.
Full-vocabulary derivation/rounding/field/native equivalence/kernel/physical
DRAM/latency are NOT implemented;source causal conv/state remain. Later23
inputs are context-dependent,so no token-only extrapolation. Compare this
compiler path for layer0 with global rank compression;finite input dictionaries
are a different function class from a shared low-rank matrix. No synthetic n
or duplicate-parameter intelligence claim. This does not replace the general
cheap context-dependent/conditional ternary transfer problem.
