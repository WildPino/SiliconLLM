# FIRST FIT-only source/core quadratic factor extraction

Prospective8 October2026. Implements [NEXT](CHATBOT_SOURCE_QUADRATIC_NEXT_20261008.md)
after independently supported curvature. New uncertainty: can ONE fixed
source-conditioned rank16 factor format be constructed and numerically qualified
before response fitting? No converted chatbot or response coefficients yet.

## Data and changed operation

Original FIT parent0..15 anchors only, current source/core BF16 G/U/D and
qualified saved gx/ux/Gv/Uv/source/core H. Entire old files hashed for identity;
novel rows NEVER used in selection/projection labels. Source slices/header only,
no full-model/source/core H/J/response/old feature experiment/optimizer replay.
Global mu/r/mass are not needed for this extraction; needed at later feature design.

Candidate IDs0..4863 source,4864..5375 negative shared512. For each FIT parent,
kappa_i[j]=a''_i*u_i*(Gv_j)_i^2+2*a'_i*(Gv_j)_i*(Uv_j)_i, negative for core.
New atom dictionary row is four output vectors kappa_i[j]*D[:,i], flattened
4x896=3584. Target is SAVED H_source−H_core, not a new H acquisition.
Explicit dictionary5376x3584 F64 per parent (154140672 payload bytes), streamed.
Normalize each row by its actual F64 Euclidean norm. Eligible finite norm>
1e-12*largest dictionary norm; other rows set0/excluded. Target energy>1e-12.

Exactly EIGHT greedy steps. Score is absolute normalized row dot current
orthogonal residual. Previously selected/ineligible IDs excluded. Choose highest
F64 score, lowest ID for exact F64 ties; require score>1e-12*||residual||.
Refit all selected normalized rows to original target with np.linalg.lstsq,
rcond=None; actual full rank k and condition<=1e8 required. No discarded modes,
regularizer, lower atom count, retry, alternate selector or independent geometry.
Any failed prerequisite closes acquisition with retained partials/fault.
Each stage projection normal and reconstruction relative to||target||<=1e-10,
residual energy cannot increase more than1e-12*target energy.
Save ALL scores/target+8 residuals/selected normalized atoms/coefficient and
singular witnesses/norms/kappas/IDs. Numbered per-parent witnesses are retained
before later fault, followed by complete concatenated arrays, without overwrite.

Each selected atom provides TWO forms, (g_i z)^2 and(g_i z)*(u_i z).
R[0:8]=R[8:16]=the EIGHT gate BF16 rows; T[0:8]=same gates,
T[8:16]=the corresponding up BF16 rows. Byte copies only, no re-encoding.
R/T[16 parents,16,896]<u2; exact source/core row IDs/hashes mandatory.
Input z/center/scale and C output coefficients belong to LATER feature/fit
protocol. Pure source-atom projection coefficient witnesses are NOT C.

## Decisions and scope

Numerical/byte/provenance/finite resource qualification plus FIRST independent
saved-score/formula/projection audit are prerequisites for one changed-feature
design. No absolute projection-quality gate because the later free C columns
define a broader output class than fixed source D_i/scalar atom projection.
Report every parent's source-atom projection RMS, norms/conditioning/diversity;
none establishes response fidelity, useful expert capacity or864-null coverage.
Original novel1%/ALL-category3% response gates and complete ALL24/export/native/
fresh own-history/tasks/accepted50 SAME artifact remain required. No rank ladder.

## FIRST audit contract before promotion

Independent BF16 exponent/mantissa promotion, scalar math.exp derivatives,
factorized candidate scores sum_j kappa_i[j]*(D[:,i] dot residual_j)/norm_i
(not explicit normalized dictionary matvec). Verify ALL128x5376 scores/choices,
lowest-ID ties under actual F64 audit arithmetic; mismatch is qualification
failure, no tolerance-based change of selected ID. Formula/norm/score numerical
gaps<=1e-11+1e-10*max norms. Verify original target/all eight saved residual
reconstructions, projection orthogonality<=1e-10, independent small-Gram
rank/condition and recorded singular values (rel1e-9/abs1e-12). Compare every
R/T byte with selected original source/core rows; ALL scalar energies/recorded
RMS/choice fields verified rel2e-12/abs1e-14. No pursuit/LS/source-H rerun.
Freeze audit source/binding BEFORE its FIRST run; retain faults/partials.

## Finite execution

Python3.12.10 isolated/no-site/no-bytecode; five restricted NumPy2.4.6/psutil7.2.2
roots, OpenBLAS1, CPU0..10/launcher11. Primary worker180s/family300s/conservative
OS4GiB/output64MiB/log2MiB through actual held-handle exit. FIRST audit proposed
same time/memory/log caps, empty directory/output1MiB. No Torch/GPU/T4/network/
new resources. Source extents before/after; common launcher source/runtime/foreign
SHA preservation/overlap guard. Launcher final receipt/stdout tail outside last
peak snapshot disclosed. Actual code/binding/full Git freeze BEFORE selection,
first independent audit/typedUTC known-instance closure BEFORE promotion.
Standard chatbot_directional_launch.py exact argv saved in terminal.command.
