# METH483: algebraic warning repair and exact partial resumption plan

6 October2026. Full goal ACTIVE/INCOMPLETE. Original483 apparatus fault retained
273a11d before repair; all134records bound by first-fault inventory
ceea4d1cd0bcc1b27327931e48aaa041068e258e694d4c7f584c1dc6242462b5.
This is a plan, NOT a frozen R1 science protocol or executed/admitted R1 result.

## What the retained bytes say

Read-only serialization diagnosis after retention, no NumPy/SciPy/model/LP:
bank7's first round proposed256rows. Their original input fields contain NO
positive/signed-zero coordinates. The explicit-zero warning hypothesis is
therefore excluded for these proposed feature rows. Exactly ONE nonzero F32
coordinate is <=the documented HiGHS1e-9 small_matrix_value default:

- UID223609, coordinate602, F32bits2944796779;
- value promoted exactly to F64: -2.4380439334059645e-10;
- minimum nonzero abs in proposed rows:2.4380439334059645e-10;
- maximum abs:276.5580139160156.

The [-h,h,1] row contains two entries of that magnitude. Bound runtime binary
has a warning template for packed matrix values <=a threshold being ignored.
This is a plausible explanation; original console/log was disabled and the
installed small_matrix_value option was not read. Exact original warning cause
has NOT been established. Preserve BOTH diagnosis JSONs, including the failed
explicit-zero explanation. No new geometric conclusion follows from these bytes.

## Why a small coefficient warning need not invalidate a witness

If model readback establishes h_tilde by deleting ONLY coordinates with
abs(h_j)<=tau, with the augmented bias/t/norm row unchanged, then

```
||h-h_tilde||_infinity <= tau
abs((h-h_tilde)^T theta) <= tau*||theta||_1 <= tau.
```

Thus the original and pruned active-set optimum margins differ by at mosttau.
This is an algebraic consequence under that readback condition, not a measured
claim about the failed call. For new arbitrary nonnegative dual weights, use
ORIGINAL h in the residual/sum interval and the certified upper bound remains
valid. For primal heads, verify ORIGINAL ALLdevelopment/consumedval inequalities
and norm/export arithmetic. Neither proof requires trusting optimality of the
pruned solver matrix. An API warning cannot be silently discarded or treated
as affine infeasibility.

## Complete prospective interface repair

1. Read/set/readback the installed small_matrix_value explicitly, fixed1e-9;
   no threshold tuning. Capture a new exclusive solver log and exact statuses.
2. On passModel/addRows, kError remains apparatus fault. kWarning may be admitted
   ONLY after full getLp readback: rows/columns/objective/bounds/norm row, every
   sparse index/value match the intended matrix after ONLY that documented
   coefficient deletion. Expected sparse format/order/dimensions predeclared;
   no arbitrary matrix differences or row loss permitted. Preserve dropped
   source indices/value bits and warning text/status/readback hashes.
3. New analytic interface control BEFORE unfinished source work: start with a
   source row(-1,0), append(+1,2^-32), labels(-1,+1), norm1<=1. The true normalized
   optimum is1. This specifically exercises tiny nonzero addRows coefficients
   and controlled readback, not an old LP/control replay. Verify primal/dual
   against ORIGINAL rows and physical arithmetic; require the planned diagnostic.
4. Keep all scalar/array mathematical checks on original source inputs. No
   normalization/acceptance threshold or root-label change; no longer LP budget.
   Complete new main/helper/reconstruction/protocol/runtime/Windows BEFORE first
   numerical import. Original83source/helper/protocol/controls/outputs immutable.

## Resume, do not repeat completed work

Banks0..6: ALL56root rounds and7final archives exist. Reconstruct selected
primal/dual, their raw provenance, exact source labels/ALLUID intervals and
complete reports from them in a separately frozen routine. NO new LP calls for
those banks and NO old4control LPcalls. Do not invent original unsaved LPseconds:
last-round clock and original bank-start/completion give bounds; original job
264.032s remains charged. New reconstruction cost separately measured.

Bank7: saved round0 includes active indices, proposed256new indices, valid
vectors/flags and ALLdevelopment signed intervals. Reuse it as the seed; build
the next active matrix from those saved rows/proposal. Original live basis was
not checkpointed, so rebuilding its basis is a declared apparatus difference.
Do not execute round0 again. Remaining at most7rounds; conservatively subtract
the original bank-start-to-failure elapsed1.828s from its25s search allowance.
No extra optimization budget hidden in the new controller.

Banks8..11: never attempted, first inquiries under fixed original algorithm/
25s/max8rounds. All source banks/labels/ties/sourceID/roles retained. Across both
controllers at most96root LPcalls; prior57retained plus at most39new calls.
Exactly4old controls retained plus1new warning control; no source capture/native
replay. Complete new global certificate/all12archives/ALL39-72-4608-9216views and
ALLold/new rounds require independent admission, without solver/helper replay.

Prospective R1budget600s/8GiB/new64MiB/admission180s CPU0/BLAS1; new audit budget
must be priced/frozen before its first import. Same-handle rule while live,
actualterminal then ONE frozen Windows query, new firstfault before any repair.

## Whole-project decision retained

482 admitted budget INCONCLUSIVE.483partial local false flags cannot reject
the affine class, general routing, LUT or whole model quality. After complete
recovery/remaining inquiries/audit, take the frozen decision: useful sufficient
heads lead to mass/C qualification; verified dual bounds reject only the uniform
margin requirement; unresolved brackets lead back to feature/partition/function
geometry and the wholeartifact budget, not a third equivalent longer LP.

Source-ID preservation is diagnostic. Final information loss depends on
|p_hat-p|*||f_e|| + p_hat*||f_ehat-f_e|| and propagation into predictive output;
fresh ownstate prediction/generation/task quality AND SAME>=50 remain required.
Core/head/selected functions, useful distinctn with RAM, physicalDRAM and CPU
LUT/mass must be included. Neither a software warning fix nor router success
alone completes pretrained knowledge transfer or proves scaling to100B.
