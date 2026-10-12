# Frozen exact whitening and matched-output-direction preflight

Site12 adaptive384,previous FIT-selected physical point;all48 source histories.
FIT-only source GY_rec from independently verified prior basis. PSD negative
relative<=1e-10,eigenfloor1e-6*largest eigenvalue,regularized condition<=1e6.
S symmetric square root,Si inverse;square inverse/projector relative<=1e-9.
Rbar=R Si,Wbar=S W,Vbar=Si V. Decode to physical channels BEFORE full skip,
gate,combinedRMS/sourceepsilon/readout. No transformation through SiLU/conv.
Initial physical projector/output/objective/ranks are unchanged within1e-9;
prior selected scalar within1e-8. No information/capacity gain from coordinates.

Full physical and whitened finite outputs on all48 histories;all24 FIT GY and
both coordinate parameter gradients. Full chain rule G_Vbar=S^T G_Vphys,
G_Wbar=Si^T G_Wphys relative<=1e-8. No DEV gradient/metric fit. Same exact
teacher/case-weighted finite objective. Full independent CPU autograd required
for BOTH physical/whitened computation graphs and all saved vectors/gradients.

Reference tangent from physical parameters versus tangent from whitened
parameters/gradients,with frozen biorthogonal Sylvester projection. Map whitened
direction back to physical channels. Normalize each direction by sqrt(mean_FIT
||J_current dY||²/||o_source||²),so alpha1 has equal first-order output RMS1.
J evaluated at CURRENT reconstruction,not source teacher. Save all FIT raw J
direction vectors and energies;independent radial/tangent action and tangent
qualification required. Source gradient/normalization never omitted.

Fine central difference of fullFIT loss along actual physical QR/biorthogonal
retraction at1e-5/1e-6/1e-7,ALL relative<=1e-6. Earlier coarse FD failures remain
historical false flags;different point/direction/new probes do not admit them.
Curve alpha[0,1e-4,3e-4,1e-3,3e-3,.01,.03,.1,.3,1],both directions. Same original
Mcondition1e8/Vrankratio1e-12/biorth1e-8,Armijo1e-4/minrelative improvement1e-8.
Report minimum feasible Armijo FIT loss per coordinate;reference then smaller
alpha breaks exact global ties. NO installed update/new scan or DEV curve
selection in this preflight. A recommended candidate requires independent full
audit,held-out/actual scan qualification before later admission.

Independent audit:all SPD/eigen/inverse/projector/coordinate/source outputs,
every physical/whitened autograd backward and chain rule;independent Kronecker
tangents/full J direction normalization/FD/candidate losses/selection/parameter
replay/all before-after hashes. Output/inverse1e-9,gradients/scalars1e-8. Every
history/position,not a small successful subset. Principal runtime/GPU pinned.

Producer900s/audit1200s,OS6GiB,GPUallocated4GiB/reserved5GiB,output2GiB on
authorized E:\SiliconLLM\results\native_expert_scaling\whitened_recurrence_preflight_20261012.
Full source generator/gate/conv/state256/skip retained. No T4/model/native/
RESERVED/longalign/new data/clean timing/complete chatbot/n/50 admission.
