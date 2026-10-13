# Full-coordinate FFN affine-cell pilot

Freeze before results. Reuse qualified actual pre-FFN x/output at0/12/23,
all FIT24/DEV24/22547 histories. No arbitrary embedding input,source model/GPU,
new generation/RESERVED/optimizer or whole-model substitution. This is a new
nonlinear FFN operator family,not a token-only/shared low-rank GG retry.

F64 smooth proxy F(x)=beta D[(Ux)*SiLU(alpha Gx)],canonical BF coefficient
values,alpha/beta effective F32 Python scalar values. Differentiate the smooth
proxy,not the quantized BF step function. No output coordinate projection.
Capture reference source BF function unchanged. Every case proxy-vs-BF
relative RMS<=1% numerical control;record the numeric floor separately.

Fit Euclidean full2048 input kmeans centers ONLY on FIT,K1/16/64. Seed0,
kmeans++ initialization,25 Lloyd iterations maximum,tie first ID,empty cells
reseed farthest FIT point. K1 FIT mean. Routing full-coordinate nearest center,
one uint32 ID/mass exactly1,flat scan explicitly not scalable routing proof.
No DEV centers/precision/count tuning or oracle selection.

Cell is F(mu)+J(mu)(x-mu). Exact implicit evaluation from source U/G/D:
u=Ux,g=alpha Gx;uc=Umu,gc=alpha Gmu,s=SiLU(gc),d=SiLU'(gc).
Prediction=beta D[s*u+uc*d*(g-gc)]. It is algebraically the full2048x2048
affine operator;physical J bank is NOT formed/timed/exported by this pilot.
Its theoretical coefficient count D^2 versus3DI is not a performance claim.
Directional central finite differences at actual FIT direction,step1e-4,
relative<=1e-5 verify the derivative formula before interpreting cell error.

Local whole-model-test ELIGIBILITY gates:a priori FIT case-mean relative
FFN output error<=10%,worst<=20%;smallest passing K selected on FIT. Same
DEV mean/worst gates,each DEV domain case-mean<=20%. These are new approximate
FFN eligibility thresholds,not relaxed source numerical controls or a waiver
of rejected GG1% field gates. Frozen chatbot gates remain unchanged. Any
eligible operator still requires actual own-history/model quality and CPU cost.

Store every432 predicted trajectories as F32,exact routing IDs/masses/radii,
all9 center/feature/derivative/offset records and real F64 metrics. Output
~1.71GB<=2GiB;CPU6cores/producer and full audit1800s each/OS8GiB. No hours-long
GPU allocation. Source/runtime/field hashes before/after,real terminal receipts.
Complete audit even if all cells fail;no restart on observation timeout.

Separate audit verifies exact FIT center replay (shared clustering code,
no optimality claim),source feature/derivative/offset identities1e-10,finite
difference1e-5,independent direct Euclidean routing and chain-rule delta action,
stored F32 predictions1e-6,and all real metrics/selection metadata1e-9.
Shared NumPy BLAS primitives declared. Every case is audited. No expert-n/
DRAM/final engine>=50 useful acceptedTPS/general source-family admission.
