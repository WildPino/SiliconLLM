# METH-227: distinct additional donor functions help, but absolute accuracy fails

The [frozen protocol](METH_227_CONDITIONAL_DONOR_TANGENT_PROTOCOL_20261001.md)
and runner were committed at `113be3f`. Session50719 completes,exit0.
All160 cells are occupied (18..1141 fit states,mean409.6); all176
full896-input function matrices have different BF16 weight hashes.
Every stored tensor reads back exactly; center worst relative
L2=2.339271e-8. Both additional centroid autograd checks pass.

| Consumed128-window function metric | Value | Fixed decision |
| --- | ---: | --- |
| E16 normalized output SSE | .7889398394 | Control |
| E160 normalized output SSE | **.5970355618** | **Fail <=.01** |
| E160 rotated leaf normalized SSE | 1.1167221063 | Correct route passes10% gain |
| E160 SSE relative to E16 | .7568,approximately24.3% less | Pass10% gain |
| Paired10,000-window-bootstrap gain P05/P95 | .185401 / .198475 | Pass P05>0 |

**Stop this fixed first-order tangent geometry.** More distinct functions
and correct choice improve this inaccurate component, but do not preserve
donor knowledge to the fixed tolerance. No routed-bank C/composition,
generation/task or independent source is opened. METH-226 operator cost
cannot be combined with failed METH-227 accuracy into a model pass.
No extra coefficients, seed/whitening/refit or threshold retry is made.

The stored functions are exact at their local center within declared
precision, but validation inputs remain far from those centers: median
squared distance/input squared norm=.589801(E16),.528732(E160);
P95=.781497/.717075. Mean squared distance decreases424.00->377.15.
These locality measurements do not prove which neglected nonlinear terms
cause the error; they motivate a fixed-output decomposition before more work.

[Raw result](meth227_conditional_donor_tangent_result.json), SHA256
`85f5b41e99850a2eeb32d0c071f32060f35d0e958db4729fffa115681a406fd2`,
retains all fit counts, every function hash/center check and each window.
Ignored local snapshot `results/native_expert_scaling/meth227_layer12_tangent_functions.safetensors`
is283,853,608bytes,SHA256
`c00e33f21eb5de7306586af38fc7f660797c742fe4362c3e12d6a4790e6868c2`.
Runtime16.453s after imports,ending RSS1.990GB,GPU peak1.635GB,local3060,
no T4. The cuBLAS initialization warning is preserved; execution/checks
complete. No project job remains active.

Next: [METH-228](METH_228_TANGENT_BIAS_BOUND_PROTOCOL_20261001.md),
a nondeployable bias oracle/decomposition on the frozen recorded functions.
It is a diagnostic bound, not a candidate refit or bias repair.
