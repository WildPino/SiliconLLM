# Stored response-onset diagnosis (frozen before observation)

10 October2026. Prior native fit/audit complete, no owned benchmark active.
Uncertainty: are new chatbot errors already present at the first response,
before any own-answer feedback, and how much loss weight does that onset get?
Reuse qualified new native result/audit/terminal and canonical old14/16 source.
No new model/core/history/head FG/optimizer/GPU/T4/reserved call.

ALL16 same-prompt first full-V source/native score rows: source and new native
first argmax IDs, KL(q_source||p_native) with independent logaddexp/math.fsum
algorithm (delta<=1e-10), source top1 probability and native probability of that
ID. Exactly14 source-correct cases. Gate at least8 first-ID disagreements on
those14 => FIRST_RESPONSE_CONDITIONAL_GAP; otherwise no onset dominance claim.
This is an operational diagnosis, not a representation impossibility proof.

ALL48 native forced-case first-label KL and continuation mean KL separately,
equal-case aggregates. Verify position0=len(original prompt)-1 in every case.
Whole fit objective gives onset total weight E_case[1/m] and continuation
1-E[1/m]. No changed fit launched. If onset gap confirmed, consider an explicitly
new temporal weighting (half per-case first response,half continuation), with
same decoder/native features/parameter domain; change objective alone rather
than calling this a replay or an information ceiling. Held-out/tasks stay fixed.

Seal script/protocol/original numerical parent result/audit/terminal/binding,
all16 original source case JSON/full score files and all16 native full score
files before/after. Reuse parent complete8808 native audit for forced-case KL
values; no duplicate6GB broad score audit.60s CPU/OS256MiB/output1MiB/no children,
one BLAS thread. Independent scalar checks complete regardless of quality.
Report actual costs/hash identities; full goal and all gates remain open.
