# Stored recurrent projection: coordinate96, orthogonal96 and dual96

9 October2026. Prospective before any projection observations. No source-model
forward/reply/head call, optimizer/native/RESERVED/T4. Uses newly captured actual
full-history operands. The endpoint still needs useful chatbot+accepted50.
After source process loss, consume the stored-only adopted48-case capture:
45 durable originals plus3 separately held completion cases and63 bit-exact
partial-prefix witnesses. Original source exit/time/peaks/final411 identity
aggregate remain missing; this assay does not retroactively qualify them.

## FIT selection and actual output experiment

Fit ALL24 site bases on ALL24 FIT histories only, equal case weight. For each
case, compute F64 B^T B/tr(B^T B) and C^T C/tr(C^T C), average equally over both
fields/cases. Top96 eigenvectors define dense basis R, orthogonality<=1e-10;
top96 diagonal coordinates define the coordinate selection. DEV never selects
basis/count/scaling. Record full Gram/eigenvalues/basis/coordinates plus joint
uncentered and separately trace-normalized centered retained energy. Centered
energy avoids mistaking preservation of the common mean for useful distinctions.
These moments are a selection rule, not an output-fidelity theorem.

Third FIT-only arm uses different read/write bases. Let G_B/G_C be the separate
case-balanced Grams above. Form H=sqrt(G_C)sqrt(G_B)=U Sigma Q^T, retain96 singular
directions. Set V=sqrt(G_B)Q96 Sigma96^(-1/2), W=sqrt(G_C)U96 Sigma96^(-1/2).
Require W^T V=I within1e-8 and retained minimum singular value>1e-12; record basis
norms/conditioning. Project B'=W^T B, C'=V^T C. Then
sqrt(G_C)V W^T sqrt(G_B)=U96 Sigma96 Q96^T. The discarded singular-value norm is
the optimal rank96 Frobenius error for the FIT-weighted CARTESIAN pair kernel.
This is finite matrix algebra; actual causal/decay/history weights differ, so
neither this optimum nor the common Gram energy proves real recurrent fidelity.
Scalar per-head decay commutes with this oblique state map. Saved Grams/bases/
singular values make the stated identities independently checkable.

Actual propagated local comparisons at source sites0/12/23, all48 complete
histories/all token positions, initial state zero. Baseline independently invokes
standalone source chunk128 SSD equations/reduction order with saved x/B/C/delta/
A/D, then source post-gate RMSNorm and BF16 out_proj. Compare to actual captured
F32 pre-gate y and BF16 source SSM outputs. Strict normalized RMS<=1e-4 per
case/site for scan/output; keep failures. Additional independent F64 sequential
state recurrence on eight fixed channels/whole1507-ID longest FIT at each site,
also1e-4, guards the reconstructed operator against self-consistent transcription.
No earlier source/native numerical criteria are changed or inherited as PASS.

Rank96 arms use SAME full x/gate/delta/48 heads/A/D/norm/out_proj, replacing only
B/C by orthogonal BR/CR, selected coordinates or dual BW/CV. Report whole-history and supervised
position output RMS, pre-gate RMS, recurrent-only RMS after removing identical
Dskip*x from both outputs, and centered output RMS after separately removing
each output's per-channel history mean. The latter diagnoses variation without
counting common mean preservation; it does not replace full output error. A large
Dskip component must not hide a poor recurrent approximation. Equal case split means/worst cases and exact
per-case records. There is no predeclared whole-chatbot quality PASS from local
error. Numerical reconstruction failure makes interpretation unqualified; do not
relax it or replay source captures. Record any finite local projection errors
as measurements, not universal rank/capacity lower bounds.

Both dense96 arms require carrying all256+256 source B/C SiLU features before projection;
coordinate96 needs96+96 features. Neither experiment removes source3072 x/gate
channels, donor48 delta functions, D2048 residual,24 blocks or parallel attention.
Thus neither is a DN512/1024 target, warm core implementation or useful chatbot.
It changes the decision about state compression; head/input/layer/operator
omissions need their own evidence and complete native cost before a candidate.

Also report exact shape accounting for D256/N96/L6/n1152/k8/H128/V65537,
DN512/1024 and DT rank16/48: core matrices, full head, selected experts, flat
router, F32 organ/state coefficients and untied state exponentials. Formula per
SSM matrix products=DN*(3D+2DT+2N), five SSM plus one4D^2 SWA. Counts exclude
conv/vector/window/norm/packing/physical DRAM and do not forecast timing or
qualify an implemented geometry. Repeated scalar source rates/delta permit
algebraic exponential hoisting, but no work is counted as skipped without code.

## Bounded execution and preservation

Held family1800s/reserve90/worker+launcherOS6GiB/GPU allocator4GiB/reserved5GiB/
outputs96MiB/log4MiB. Validate every consumed operand/source/code/config/runtime/
foreign raw hash pre/post; exact childless Windows held-worker instance through
exit. Full output hashes and first fault/current site/actual completed bases/
metrics/resource peaks retained. No overlapping owned jobs or retries because
observation yields. Pinned coefficient file is read directly with safetensors;
no Transformer model is constructed. Source code supplies pure shape/segment
helpers; forward methods are never called. Original engine unmodified.

[Worker/binder/launcher](../../../benchmarks/native_expert_scaling/original_falcon_recurrent_projection.py).
Bind only after capture is terminal, freeze this code/protocol/binding, then run.
