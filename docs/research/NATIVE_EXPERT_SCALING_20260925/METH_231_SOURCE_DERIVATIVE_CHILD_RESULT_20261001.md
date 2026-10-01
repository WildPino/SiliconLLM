# METH-231: source derivatives remove the copy, but nonlinear function quality fails

The [protocol](METH_231_SOURCE_DERIVATIVE_CHILD_PROTOCOL_20261001.md)
and runner were frozen at `aed767b`; session52391 completes,exit0.
The exact METH-230 repeated-input diagnosis passes. All11 reused parent/
input/selection/prior/route control tensors remain equal; E16 total fit
metric matches exactly. Every child receives actual source derivative
information at its frozen center with its inherited trained parent
nonlinear response. All176 fitted affine/down pairs now have distinct
BF16 bits. This is source information, not a perturbation/hash repair.

All160 center checks pass (worst relative L2=7.413132e-9), both complete
autograd checks pass and all160 normal-equation checks pass (worst
relative residual5.511938e-15). Every snapshot tensor reads back exactly.

| Function metric | E16 control | Source child prior | Fitted E160 | Rotated E160 |
| --- | ---: | ---: | ---: | ---: |
| Fit normalized output SSE | .0305380161 | .0628630547 | .0119604282 | Not scored |
| Consumed-validation normalized SSE | .0946990577 | .1020987242 | **.0938104642** | .1316283078 |

E160 validation is only approximately0.94% lower SSE than unchanged E16,
failing the fixed10% gain. Absolute error9.381% also fails<=1%; improvement
against its new prior fails10%. Rotated-choice gain passes. Paired-window
gain P05/P95=.000446012/.001336123 is positive, but cannot override the
failed absolute/relative gates.

**Decision: stop this fixed source-derivative nonlinear child geometry.**
No stored routed-bank C/full-model/generation/task/independent source is
opened. METH-229's8.875ms untrained shape fixture cannot be combined with
failed function quality into a model pass. The method improves the prior
affine-only function approximation and handles repeated-input slope
information, but does not conserve donor capacity to the required tolerance.

The fit/validation gap does not isolate feature coverage, regularization,
training support or domain variation. Do not claim a universal H2048,
nonlinear-sharing or source-derivative impossibility. The next proposed
representation increases **source nonlinear coverage** while changing
storage/operator granularity; it does not retry the stopped widths,
source selection or strength after seeing these results.

[Raw result](meth231_source_derivative_child_result.json),SHA256
`c83da761491f66473c3b64cf09539dadeef5cff865551677b871869a932a9664`,
retains all unchanged controls,source/solve checks,counts,histories and
every validation window. Local ignored snapshot
`results/native_expert_scaling/meth231_layer12_source_derivative_children.safetensors`
is1,389,422,328bytes,SHA256
`1f5d84c333054d5df5681e51852fe1242c052b7ddd949419d8d904b4dfb2b3e9`.
Runtime47.625s after imports,ending RSS2.986GB,GPU peak5.476GB,local3060,
no T4. Initialization warning and partial progress retained; no job active.

Next: [full-source row-Q8 qualification proposal](FULL_SOURCE_ROW_Q8_PROPOSAL_20261001.md),
first native codec/arithmetic/cost and scoped source-function error,
before any full-feature conditional readout fitting.
