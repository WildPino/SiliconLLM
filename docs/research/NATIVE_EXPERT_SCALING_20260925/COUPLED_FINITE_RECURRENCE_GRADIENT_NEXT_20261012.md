# Coupled finite-output recurrence gradient and one bounded FIT update

12 October2026. **SELECTED / PROPOSED / UNIMPLEMENTED / UNBOUND.**
Selection follows the predeclared sensitivity rule on FIT:both site12 layouts
have first-order remainder>.25. Full sensitivity audit now qualifies all vectors/
statistics PASS. No unchanged corrected-average-GM fit selected.

## Finite objective with unchanged memory budget

Actual source histories/gates/postconv x retained. For each head h,source
recurrent output R_h=y_h-D_h*x_h,learn rank-fixed oblique projector P_h=V_h W_h^T.
Initialize declared F64 masters from the FIT recurrent-only balanced bases;
retain the previously diagnosed promoted-F32 projector as a separate control,
record any baseline difference and qualify conditioning/biorthogonality before
retraction. Do not silently treat these points as bit-identical. Same adaptive ranks and
total384,source state256,full source skip. This is a local source-injected
operator pilot,not history acquisition or a complete chatbot.

```
Y_h = (R_h W_h) V_h^T + D_h*x_h
U = SiLU(source gate)*Y
o_hat = F U/sqrt(mean(U^2)+epsilon_source)
L = mean_FIT_cases [sum_t ||o_hat-o_source||^2 / sum_t ||o_source||^2]
```

Teacher unroundedF64 o_source and case denominators fixed. Use FULL reconstructed
RMS and finite output objective;no independent-time trace or tangent metric
substitution. Source gate is an injected operand,not a trainable generator here.
Source column-space categorical moments remain available for later joint
causal/readout adaptation;this local output loss alone does not preserve chat.

## Exact backward and projector constraint

For G_o=dL/do_hat,readout F and current U/r:

```
G_U = (G_o F)/r - U * sum[U*(G_o F)]/(3072*r^3)
G_Y = SiLU(source gate)*G_U
G_Vh = G_Yh^T (R_h W_h)
G_Wh = R_h^T (G_Yh V_h)
```

Accumulate with the exact equal-case denominator and FIT-only weights. Both
center/normalization effects included;no half Jacobian or updated-case teacher.
Independent full derivative qualification and finite differences precede an
update. W_h^T V_h=I must be maintained,not presumed for generic factors.
One candidate retraction:V'=V-alpha G_V,W_raw=W-alpha G_W,
M=W_raw^T V';W'=W_raw M^{-T}. Verify rank/M conditioning/biorthogonality and
the first-order derivative along this retraction;freeze candidate schedule,
normalization and stops before observing loss. If conditioning fails,retain
that candidate as rejected rather than silently changing the manifold.

## Finite first implementation

Start site12 adaptive384,all24 FIT histories;fixed8/site0/site23 remain frozen
controls,DEV evaluation only. Freeze finite gradient/control/output formats,
resource caps and FIT-only step-selection policy BEFORE execution. One bounded
line-search update with full operator/gradient/parameter-bit audit. If no step
qualifies,retain the nonacceptance and analyze the derivative/constraint rather
than declaring a capacity floor. No hours-long plain-gradient epochs selected.

For a selected new projector,generate NEW actual compressed scans with new W
and verify full CPU sequential/source-projected histories;fixed within-head
decay makes channel projection commute,not conditional changes across heads.
ActualF32 maps/scans and BF16/native export remain separate machine gates.
Do not substitute source projected labels for actual compact scan qualification.

## Toward the full goal

This changes the mathematical objective where the averaged surrogate failed.
After a useful local representation,measure source-derived native active cost
including retained generators/gates/conv/fullskip;separately test state96
composition and whole causal own-history behavior. Site23 accepted adaptive
representation is a cost-pilot candidate,not a general donor porting success.
No original-engine change,quantization/router/n/DRAM/multifamily/same useful
artifact>=50 claim from the pilot. A required hours-long run follows the user's
announce/bind-launch/pause instruction;no Luna/delegation/T4 allocation.
