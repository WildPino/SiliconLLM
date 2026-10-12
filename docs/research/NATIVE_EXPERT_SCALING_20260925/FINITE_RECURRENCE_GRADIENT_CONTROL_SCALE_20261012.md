# Separate finer derivative qualification after full backward proof

Full CPU autograd/radial-tangent/Kronecker auditV3 qualifies every saved V1/V2
backward/direction;errors~1e-15,all input/output hashes exact. Original FD gates
remain FALSE at1e-3/1e-4/1e-5:relative.02245284/.000225185/2.251916e-6. Added
1e-6 control passes with2.252001e-8. Error decreases approximately quadratically
with step,consistent with central-difference truncation along the nonlinear
retraction;independent full autograd distinguishes this from a wrong backward.
None of the original failed gates is retrospectively admitted.

ProducerV3 records ALL original coarse probes/flags/case losses/parameter points.
It introduces a separate qualification at1e-6/1e-7/1e-8,ALL relative<=1e-6
(same most stringent original derivative tolerance). Only after those new
controls pass may the unchanged FIT candidate ladder/Armijo policy select ONE
update. Exact objective,initial/gauge/ranks/teacher/gradient/tangent/candidate
schedule/selection/actual scan and local quality gates/resource bounds unchanged.
New code/binding/storage/result namespace. No hidden scale or tolerance repair.
Complete independent full audit,including every original false flag,required.

Earlier failed partial audits are retained:missing tangent-report lookup then
NumPy bool JSON adapter;these did not produce a complete adjudication. Completed
auditV3 is the authoritative saved-backward/scale evidence. No model/native
quality admission and no unsupported claims about unstored failed-producer FD.
