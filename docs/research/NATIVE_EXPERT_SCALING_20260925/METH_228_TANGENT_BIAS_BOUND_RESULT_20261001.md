# METH-228: bias-only repair cannot rescue the frozen tangent functions

The [protocol](METH_228_TANGENT_BIAS_BOUND_PROTOCOL_20261001.md) and runner
were frozen at `24450d6`; session95537 completed,exit0. Every128 original
window energy and all three E16/E160/rotated SSE values replay exactly.
The recorded-error decomposition reconciles total SSE at relative0(E16)
and1.440460e-16(E160),within1e-12.

| Diagnostic | E16 | E160 |
| --- | ---: | ---: |
| Original normalized output SSE | .7889398394 | .5970355618 |
| Fraction of original SSE in per-cell mean residual | .3226493582 | .3387858525 |
| Optimal real-valued bias oracle normalized SSE | .5343889064 | **.3947683601** |

The oracle uses validation targets and exports no candidate. Even this
optimistic E160 error is far above the rounding-aware necessary upper
bound **.0100000144061** for any bias-only candidate passing .01 with
these fixed FP32 matrix products and declared add/subtract arithmetic.

**Decision: exclude bias-only repair of the frozen tangent products.**
The failure is not solely a per-cell mean offset. More distinct functions
and correct routing improved METH-227, but changing the non-constant
function response is necessary. No learned bias/checkpoint, coefficient
refit, new quality or native retiming is performed. This bound does not
exclude different Jacobians, other matrix products, altered routing or
nonlinear functions; it is not a universal affine-model impossibility.

[Raw result](meth228_tangent_bias_bound_result.json), SHA256
`103899a884a141f997dffe0c5036843c36363550766928916110acbac3d48de0`,
contains all replay rows, per-cell support/decomposition and bounds.
Runtime3.687s after imports,ending RSS2.657GB,GPU peak1.132GB,local3060,
no T4. No job is active. The prior goal turn plus METH-226/227/228 changes
authoritative evidence and next action; this is progress, not a wait.

Next: qualify the [affine plus nonlinear source-branch proposal](CONDITIONAL_SOURCE_CURVATURE_PROPOSAL_20261001.md)
natively before training a conditional bank. This changes active function
representation; no stopped tangent-only or stand-alone common is resumed.
