# Scope clarification: native MoE family scaling is sufficient

15 October 2026. Latest owner clarification in the project conversation; this
supersedes conflicting interpretations of the preserved original goal and the
earlier pipeline reassessment. No experiment or model result is changed.

## Accepted scope

There is no mandatory dense-versus-MoE starting architecture. A native pretrained
MoE is a valid primary donor. Further partitioning its experts is allowed when
it contributes to the same useful-quality and active-cost objective.

A reproducible procedure for **one model family that includes very large
models** is sufficient. Demonstrating unrelated families is an extension rather
than a mandatory completion gate. Converting a dense70B donor is not a required
prerequisite for success on a native MoE family.

Dense conversion may be deferred or explicitly unsupported when practical
evidence makes it an unattractive route; a universal impossibility proof is not
required. An untested dense route must still be described as untested, rather
than proved impossible. A small native MoE can be the tractable first pilot,
but choosing a small donor alone does not demonstrate the requested large-model
pipeline.

## Requirements that remain

The procedure must preserve useful pretrained knowledge/capacity and execute
in the C engine, with a reusable core and useful selectively consulted capacity.
The same quality-qualified artifact must achieve at least50 useful accepted
batch-one CPU tokens/s on declared hardware and contexts. Actual routing IDs
and masses, selected coefficient and cache costs, physical memory behavior,
conversion resources and reproducibility remain part of admission.

Scaling must distinguish total distinct stored capacity, active work, router
cost, preserved behavior and actual memory traffic. A larger expert count,
duplicated banks, metadata forecasts or small-model speed alone do not establish
that a much larger pretrained family member remains useful and fast. The
family-specific method must identify how the always-on path and conditional
consultation stay within budget as total capacity grows, and demonstrate this
with actual larger-model evidence rather than extrapolation alone.

## Operational consequence

The independent strategy review has been given this corrected scope. It may
prioritize a small native MoE pilot if the experiment tests a credible procedure
for very large members of an identified family. It must explain the active-core,
expert granularity, selection/mass and conversion scaling limits. There is no
requirement to solve dense-to-compact conversion before pursuing that path.

The ongoing source-only capture and previous failed experiments retain their
original scope. This clarification neither launches another learner nor
promotes any current model to a completed pipeline result.
