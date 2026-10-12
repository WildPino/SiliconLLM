# Frozen finite recurrence gradient, tangent and one FIT-selected update

Site12 adaptive384,ranks fixed,all24 FIT histories for gradient/selection/all24
DEV evaluation. Full source gate/generator/conv/state256/fullskip retained.
L=equal-case mean sum||o_hat-o_source||²/sum||o_source||². Teacher and denominators
fixed,unroundedF64 combined reconstructed RMS. Source-injected local operator;
no BF16 derivative or complete chatbot admission.

Initialize F64 V/W masters. Per-head thin QR with positive R diagonal and
W'=W M^{-T},M=W^T Q,canonicalizes gauge/biorthogonality. Record raw masters,
canonical point,all conditioning and projector differences<=1e-9. Repeated
baseline differs from prior promotedF32 control by<=1e-5 RMS scalar.
Master biorthogonality<=1e-8,V minimum/maximum singular ratio>=1e-12,
M condition<=1e8. Reject ill-conditioned/nonfinite candidates,retain attempts.

Full exact backward:all FIT output and GY words plus all per-case/total V/W
gradients retained. Orthogonal tangent projection solves
(W^T W)Lambda+Lambda(V^T V)=G_W^T V+W^T G_V;
GprojV=G_V-W Lambda,GprojW=G_W-V Lambda^T. Direction=-Gproj times
||parameters||/||Gproj||. Tangent/slope identities relative<=1e-8,slope<0.
Central differences along actual QR/biorthogonal retraction at1e-3/1e-4/1e-5,
relative tolerances1e-4/1e-5/1e-6,on FULL FIT loss,not a selected tiny cohort.

Candidate alpha[0,1e-4,3e-4,1e-3,3e-3,.01,.03,.1,.3,1]. For each candidate,
Vraw=V+alpha dV,Wraw=W+alpha dW;QR Vraw=Q R;W'=Wraw(Wraw^T Q)^{-T}.
This gauge produces the same oblique projector as non-QR retraction. Store
every parameter point;feasible candidates evaluated on every FIT history.
Selection:positive-alpha Armijo c1e-4 plus relative improvement>1e-8,minimum
FIT loss;smaller alpha breaks exact tie. DEV excluded. If none,retain no update.
One selected update only,no long epochs.

Store baseline/selected full F64 outputs on all48 histories. If accepted,new
actualF32 compact scans with new W,all384 coords/state256;source projected
full/recurrent relative agreement<=1e-4. ControlledF64 decoding of actualF32
maps/scans then BF16 input/output rounding,both source/combined denominators;
complete output/scalar/label/domain gate audit. Old-F32 precision bridge remains
separate. Original local DEV gates .1mean/.2worstcase/.2worstdomain/.1centered.

Independent CPU audit:all FIT outputs/GY/parameter gradients via autograd AND
radial/tangent closed-form action;independent Kronecker Sylvester solution,
full FD/candidate losses/FIT-only selection/parameters/ideal outputs;all selected
actual sequential scans and BF16 outputs/gates. Parameter-bit replay from stored
qualified direction uses frozen retraction code;independent constraint/projector
equations are a distinct numerical check,not bit-exact independent-LAPACK claim.
Gradient/scalar relative1e-8,vector1e-9;input1e-6,scan1e-4,BF16output1e-4.
Full before/after input/output/code hashes,complete even if quality fails.

Producer1800s/audit1800s,OS6GiB,GPUallocated4GiB/reserved5GiB,output4GiB in
authorized E:\SiliconLLM\results\native_expert_scaling\finite_recurrence_gradient_20261012.
Storage/command/extents tracked in repo. No T4/source forward/native/RESERVED/
longalign/new data;no clean timing,n/50 or complete model quality admission.
