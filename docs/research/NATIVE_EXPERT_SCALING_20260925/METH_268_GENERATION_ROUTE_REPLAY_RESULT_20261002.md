# METH-268: generated-prefix drift includes a routing contribution

Fixed consumed sample:471 prefixes,all24 sources,both original BF16 E1280
and saved265 trajectories,prospective grid plus first-divergence neighborhoods.
Initial Python shadowing stop preserved; repair92c445d renames only the
exception variable. Repaired session14458 exits0,375.921s after imports,
RSS2,567,196,672 bytes,peakGPU4,317,503,488 bytes. No job remains active.

All apparatus guards pass: original265 own next choices exact,all same-input
routes/selected BF16 dictionaries exact,all oracle-replayed parentIDs/scores/
childIDs exact,finite logits/FFN values. Distinct stored alias encoding is
not the source of these finite same-input routing differences.

| Category | Prefixes | Candidate differs from BF16 E1280 | Oracle routing differs | Recovered / matching lost |
| --- | --- | --- | --- | --- |
| Code |175|18|12|6/0|
| Prose |154|14|9|5/0|
| Technical |142|8|4|4/0|
| Pooled |471|40|25|15/0|

Nondeployable reference-route replay removes15/40 next-choice differences
(37.5%) with zero newly lost matching choices. Actual last-position parent
order changes somewhere in24layers on467/471 prefixes; child order changes
on471/471. These are any-layer ordered-ID witnesses, not per-cell failure
rates or a learned large-n comparison. Replay replaces routing *and scores*
at every prefix position,while compact core/conditional inputs remain
candidate. It therefore isolates that combined routing/gating intervention
at these consumed prefixes, not every contribution to semantic errors.

Mean logit relative squared error is .000840118 ordinary and .000876964
under replay (+4.3858%). Better top1 agreement does not imply better full
logit fidelity. Mean single-reference-state FFN relative squared error is
.000198409 against the actual BF16 source FFN. Different states/reference
arithmetic prevent direct causal comparison with earlier FP32-source
component errors. The25 remaining choice differences require core/input-
function investigation; a routing-only fix is insufficient here.

Raw [repair1 result](meth268_generation_route_replay_repair1_result.json)
SHA256 `ed3d8289d5e729d59d4c545bff88ab0d294056d777e0b5689766ccb012fd1abe`.
No fit,new artifact,free generation,changed semantic verdict or new quality
promotion.259 remains closed by267. Next separate source BF16 intermediate
arithmetic,LUT and actual stored FFN error on fixed consumed states before
a changed source representation/route-robustness recipe. Useful RAM-scale n,
native CPU route/LUT/real DRAM/accepted rate and multi-family/10B/100B remain due.
