# Next conditional transfer test: source-linear sketch of neuron importance

13 October2026. PROPOSED/NOT EXECUTED. Implement only after the complete
per-token oracle audit and custody closure. No cheap routing evidence yet.

The provisional oracle curves motivate a precise question:can a small source
input sketch predict useful selected neurons without first evaluating all
4608 full input projections? This is a routing approximation;every chosen
nonlinear correction and every output coordinate remains full size.

## Algebra and FIT-only construction

At existing routed center mu,use delta=x-mu. Fit an orthonormal input basis
P_k on the token-weighted covariance of actual FIT deltas,with ranks
64/128/256/512. Compute the whole2048 spectrum once per site;save covariance,
eigenvalues and basis. Verify reconstruction/orthogonality/eigen residuals.
No DEV directions or center changes. This is a declared PCA heuristic,
not an optimal importance feature or an expressivity bound.

Offline form U_k=U P_k and G_k=alpha G P_k. At current input z=delta P_k,
estimate u_hat=uc+U_k z,g_hat=gc+G_k z,then the exact remainder expression
evaluated at those estimated source coordinates. Rank estimated
r_hat_i^2*||beta D[:,i]||^2,stable source-index ties. This evaluates all
projected neuron functions but uses k input coordinates,not2048. A full
source U/G calculation is allowed only in the diagnostic/reference path;
it must not be used to select deployed IDs.

For chosen IDs calculate actual U_i x,alpha G_i x and the exact combined
remainder using their full source rows. Candidate=full-output affine cell
plus beta D[:,S]r_S. The input sketch approximates importance selection,
not the accepted output or a hidden-state replacement. No old failed
fixed-output projection bank or token-only shared GG rank is revived.

Proposed active neuron counts by site:0=>512,12=>2048,23=>32. These are the
smallest tested noncontrol counts meeting the existing mean<=.10/worst<=.20
on FIT oracle curves;confirm that rule from the qualified audit before
binding. Oracle counts are a diagnostic guide,not candidate admission.
Select the smallest FIT-passing sketch rank;then evaluate the same choice
on DEV. No promotion through DEV. Frozen local and whole-model gates remain.

Save all input bases/projected source rows/ranks/IDs/full predictions;
compare each case with source BF and qualified full-source oracle. Complete
independent audit/source canonical identity/hash/terminal custody. No
RESERVED,GPU,optimizer or new whole-model substitution during the assay.
Freeze code,binding,exact payload bytes and bounded CPU budgets before use.

## Resource screen toward the real engine

Additional shared sketch coefficients:2048k+2*4608k=11264k. Selected source
functions plus a full affine expert:4194304+6144r. At k128,r512 this is
8,781,824 logical coefficients versus28,311,552 source FFN. At k128,r2048
it is18,219,008,so middle-layer savings are modest. Bias/center fields,sorting,
nonlinear evaluation and precision/memory traffic add costs. Do not equate
logical saving with CPU latency or actual DRAM.

Before expanding to all24 sites or allocating many physical experts,
compile/time a representative real operator with the chosen routing and
precision on OWN input;include its loaded bytes/routing IDs AND masses,
cache/DRAM behavior and exact array checks. A width-fixed predictor improving
choice only matters if its physical cost leaves room for shared core,
contextual generators,attention,head and the full20ms useful50 budget.
If cost cannot plausibly fit,change the representation rather than spend
hours on a larger compatible donor runtime. Any subsequent hours-long job
obeys the user's announce/bounded-launch/goal-PAUSE rule.

This is a new transfer tool in a still incomplete procedure. Expert-n
capacity/generalization/multiple donor families/10B and100B scales and same
artifact quality/TPS remain unproven. Fit master/gradient costs and all
shared resources growing with total expert count must be accounted for.
