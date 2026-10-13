# Fixed failed-state mechanism;survival-sum independent audit

Run after full temporal audit numeric PASS. Select ALL and ONLY histories
whose fixed temporal own-output or final-state flags failed. Diagnosis only,
no DEV-dependent model/rank/recovery selection or fitting. Use the SAME
qualified original/regenerated F32 projected B and original/regenerated BF16
x/delta,actual source Alog. Compute a=exp(delta*(-exp(Alog))) in F64.

Paired real recurrence S=a*Sprev+(delta*x)B with zero state. Ordered forcing
expansion is fixed before contribution observation:

    Uhat-U = (dhat-d)*x*B + dhat*(xhat-x)*B + dhat*xhat*(Bhat-B).
    Edecay_t = ahat_t Edecay_prev + (ahat_t-a_t) S_original_prev
    Eterm_t = ahat_t Eterm_prev + corresponding forcing term.

Save complete F64 original/regenerated/error/four component final states and
4x4 contribution Gram matrix normalized by ORIGINAL native final-state norm.
Retain cross energies/cancellation. Contributions depend on this ordered
factorization;they are not unique causal fractions or Shapley attributions.
SiLU gate/C/output norm do not enter the state update;their current-output
attenuation is separately measured by the preceding temporal experiment.

Independent audit derives final states by survival-weight sums,without the
producer recurrence loop. w_t=product_{k=t+1..last}(a_k). Final originals/new
are sums w*(delta*x)B;decay component sum(w_hat-w_original)*U_original;
other components sum(w_hat*corresponding forcing). Contract per head using
matrix products. Full7 final F64 vectors/component identity/Gram/scalars/
hashes;relative component gate1e-10;identity error in original-state units
1e-10;paired original/new native-state and delta-error in original-state
units1e-4. Native arithmetic tolerance is unchanged;F64 identity is not
claimed bit exact for the rounded F32/FMA engine. Record all numeric flags.

Prepare/run/audit each300s/OS2GiB/newE<=128MiB,CPU0..5/sixBLAS. Bind complete
parent report/adjudication/binding and exact selected source fields/export/
recovery/baseline/new packets,resolved ABI/BLAS/psutil/Python/source/protocol/
storage. No source-model/GPU/T4/RESERVED/native rerun/optimizer/teacher call.
If forcing from approximate inputs explains the fixed temporal failure,retain
failure and prefer a bounded TRUE in_proj input capture/calibration proposal;
do not tune the inverse on DEV or waive the1% state gate. Spectral compression
proposal remains conditional until calibration-data fidelity is qualified.
