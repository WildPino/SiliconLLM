# Exact nonlinear remainder at frozen full-coordinate FFN cells

Freeze before this diagnostic and only run after full affine-pilot auditPASS.
All48 histories/sites0/12/23,existing64 FIT cells/routes unchanged. No new
centers/DEV selection/source model/GPU/RESERVED/compiled experts. CPU6cores,
1200s/OS8GiB,small JSON output. Input/source/model/prediction hashes before/after.

Let u=Ux,g=alpha Gx,uc=Umu,gc=alpha Gmu,du=u-uc,dg=g-gc,s=SiLU.
First-order feature a= s(gc)u+uc*s'(gc)dg. Exact omitted feature:

u*s(g)-a = uc*[s(g)-s(gc)-s'(gc)dg] + du*[s(g)-s(gc)].

First term is pure gate curvature;second is cross interaction. Apply beta D
to each and require exact reconstruction relative<=1e-10 percase. Report
their norms AND cross inner product;do not add norms as squared energies or
interpret cancellations as independence.

Second-order correction = s'(gc)*du*dg +.5*uc*s''(gc)*dg^2,
s''(g)=sigmoid(g)(1-sigmoid(g))*[2+g(1-2sigmoid(g))]. Compare affine,
affine+exact cross,affine+bilinear,and complete quadratic proxy with real
source function AND stored BF output. All neurons/source matrices still used:
this is a diagnostic of the defect,not an efficient new candidate or a speedup.

Decide subsequent representation from the terms actually carrying error.
Possibilities include retaining a small selected nonlinear remainder on top
of a full-output affine cell,or another source-derived nonlinear representation.
No feasibility claimed until subset coverage/physical CPU cost/own-model
quality/IDs AND masses/actual DRAM are measured. Full prior failure audits and
frozen whole-model quality gates remain unchanged. No universal affine-cell
impossibility inferred from1/16/64 Euclidean centers.
