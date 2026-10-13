# Fixed finite FIT source-function rank envelope

13 October 2026. Implementation protocol before source freeze/binding/launch.

Only sites 0/12/23 and the 24 FIT histories/site (11148 positions) qualify
actual H and actual-input own-normalized native raw F32 scores. Equal history
weights: C_H=mean_case(H.T H/n), C_Y=mean_case(Y.T Y/n), both UNCENTERED.
H is actual BF16 after .625/before in_proj, decoded F64; W is original BF16
6704x2048 source in_proj decoded F64. Output Y is the stored F32 contraction,
promoted F64 for this finite proxy, not an exact real O*Z product. DEV arrays
must not be read or used for covariances, section energies, bases or ranks.

Input sections: gate 0:3072, x 3072:6144, B 6144:6400, C 6400:6656,
delta 6656:6704. Positive E_s=trace(W_s C_H W_s.T) REQUIRED; zero-energy
section unsupported, no epsilon. S_s=1/sqrt(E_s) balances these five LINEAR
response energies; it does not balance nonlinear/temporal quality. L=sqrt(C_H),
J=S W L, K=L W.T S^2 W L. Input squared singular spectrum from K, output
spectrum from C_Y. Fixed diagnostic ranks 128/256/512/1024/2048 only.
Squared residual tail divided by total spectrum, and its square root, report
the finite weighted globally-linear or output-projection optimum proxy.
These are not a lower bound on nonlinear/conditional experts or causal capacity.

## Numerical policy frozen before observation

F64 full dense matrices/eigensystems. Preserve raw eigenvalues, minimum/maximum,
negative counts/mass and exact zero counts. Symmetrize each Gram by (M+M.T)/2;
record pre-symmetry residual. Covariance root uses max(raw eigenvalue,0),
no positive-mode flooring, reciprocal or inversion. Report removed negative
mass; material negative eigenvalue INVALIDATES numeric qualification but does
not suppress the remaining diagnostics/audit. Tails use the same declared
nonnegative spectrum. Finite F64 envelopes, not interval-certified exact-real
lower bounds. No compressed factors/kernels are constructed in this run.

Gates: matrix reproduction 1e-12 relative Frobenius; asymmetry <=1e-12;
orthogonality max(abs(U.T U-I)) <=1e-10; eigen residual <=1e-10 relative
Frobenius; negative eigenvalue/max <=1e-10; root-square and balanced trace
abs(trace(K)-5)/5 <=1e-10. Independent energy/scales/K agreement <=1e-10.
Metadata absolute-or-relative error abs(x-y)/max(abs(y),1) <=1e-10.
Direct rectangular SVD squared spectrum max absolute disagreement/max sigma^2
<=1e-8, ALL 2048 modes for each input J and output weighted history stack.
No rank or threshold changes after results. Retain full audit even poor tails.

Independent auditor does not import producer: full H/Y moment sums in 32-row
blocks, root reconstruction/square, all eigenbasis residuals/orthogonality,
energies from ||W_s L||_F^2, K via 128-row J block Grams; direct SVD of J
(6704x2048) and equal-history weighted Y stack (11148x2048). Verify every
stored matrix/vector/spectrum/scale/energy/tail, metadata/gates, full before/
after hashes, canonical FIT selection and original data adjudications.

## Custody and caps

Prepare/run 900s each, audit 1200s, OS peak <=6GiB, new E array family <=1GiB.
CPU affinity 0..5, OPENBLAS/OMP/MKL threads 6 set before Python; no GPU/source
model/native recurrence rerun/teacher/optimizer/RESERVED/T4/children. Runtime
pins actual resolved NumPy ABI/linalg/BLAS, psutil binary and Python paths.
Hash source/checkpoint/config, scripts/helper/protocol/storage, provenance
reports/bindings, 72 FIT H and native packets, engine and foreign tracked files
before/after. Native packet hashing covers its extra arrays but only own raw
F32 scores are consumed. This is not a cold-DRAM timing/admitted model rate.
Source and audit separate freezes/launch receipts/terminal/PID+creation times.
Store seven 2048-square matrices/site plus spectra/energies/scales (~705MB).

Next: inspect fixed-rank tails to decide a FIT-only candidate rule and rounded
field/state/output gates, or move to conditional residual functions. Retain
the exact initial-token input compiler alternative. Cheap functions still
need source-informed whole layers and OWN-history/chatbot validation, useful
ternary n, actual DRAM, multiple donors/scales, and SAME useful >=50 token/s.
