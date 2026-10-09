# Mean versus varying correction after saved geometry

9 October2026. Adaptive followup, frozen before its observations. Previous saved
SVD diagnostic51b1c97/b6290fe6 completed:site23 output energy99 rank29/stable1.078,
site0 rank203/stable32.269 on274 FIT rows. This is uncentered energy, not knowledge.
New uncertainty: is apparent final-site recovery predominantly a common offset?
This determines whether to seek a compact common response before private experts,
and prevents conflating normalized L2 gain with recovered conditional variation.

Reuse actual source y, selected before and learned after BF16 packets. No new
data, model calls, updates, native or RESERVED. F64 error r decomposes exactly as
norm(r)^2=N*norm(mean(r))^2+norm(r-mean(r))^2. Report both normalized components,
source mean energy fraction, error normalized to centered source energy, and the
signed fraction of actual squared-error reduction due to mean versus variation.
Verify identities/retained aggregate errors within1e-12. Negative gain is allowed.

Add ONE prospectively specified FIT-only constant correction:mean of the two
FIT per-case mean residual vectors, equal case weight. Evaluate before+c on all
four cases without using DEV to fit c. Save c with hash. This is a diagnostic
common-output control, not a deployed new bias/operator or a model candidate.
Its varying error must remain unchanged up to F64 roundoff. Report behavior even
if it worsens output. No selected learning/checkpoint/architecture thresholds.

Read the results as scope-limited evidence. Mean correction can be useful but
does not establish preserved input-conditioned computation. A centered error is
not a full-chatbot quality metric; source means are empirical and domain-limited.
No claim that the identified common component generalizes beyond these operands.

Same51 saved-input extents plus prior result/binding/receipt/new code/protocol.
CPU six cores,120s family/20s reserve,2GiB OS/zero GPU/8MiB outputs/log4MiB/no
overlap. Stop on nonfinite/bounds/hash/identity failure;retain original fault.
