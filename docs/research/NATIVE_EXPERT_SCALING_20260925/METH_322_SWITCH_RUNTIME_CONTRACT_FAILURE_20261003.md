# METH-322: direct router control fails before sparse/full forward

Freeze `73d7bdc`, exit1 on the first expected-mask/probability/third-output
shape assertion. No checkpoint weights read. Cases list remains empty: sparse
and full model forward contracts have NOT been executed by this run.
Do not claim a measured full forward crash or source quality failure.

Installed source computes max(keepdim=True) index, then one_hot: on3D inputs
this introduces an extra singleton dimension before cumsum(dim=-2). That
static axis derivation explains a plausible mask/capacity mismatch; the322
failure record does not capture actual tensors/shapes and cannot isolate
which part of its combined assertion failed.

Preserve this incomplete diagnostic before a NEW capture: retain same seed,
config/inputs/source code, record all router outputs and boolean controls,
continue ALL sparse/full forwards even if the first control fails. Separately
compare sparse output to a logical per-sequence top1/capacity masked expert
sum, with fixed1e-6 relativeL2 gate declared before observation. This is a
new diagnostic, not changing322's expected mask or making a failing reference
eligible. Source weights/reference acquisition remains stopped.
