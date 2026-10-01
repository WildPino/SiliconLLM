# Proposal: preserve the full parent and learn a tied latent child correction

**Status: historical proposal, implemented and rejected by METH-249;
the matched residual-basis variant METH-251 also fails useful count.
Private codec/new bank C work was not licensed. Original prospective
text below is not the current execution queue; use INDEX.md.**
METH-247's actual rank32 E160 loses2.26% on
consumed windows;METH-248 raw children lose27.78%. METH-242/243 preserve
source precision or center value only,with fixed source-reset slopes,and
fail. Change continuous parent-child coupling before another codec test.

## New information coupling

Retain the **whole actual METH-247 E16 parent function**,including unchanged
mixed base,learned BF16 rank32 factors and bias. Its consumed SSE/energy is
.0001176413. A child adds a latent correction through that parent's shared
left factor Lp,not an independently estimated child output basis:

`child(phi)=actual_parent(phi)+Lp*(delta_right_child*phi)+delta_bias_child`.

Rank remains32. This ties output directions across a parent's ten children
and protects parent adaptation over the whole input domain. Previous
METH-243 conserves only a center value;it does not test this full-function
anchor. No inherited source values/Jacobians reset all child slopes.
The parent's actually stored function is the reference,not a silently
substituted unrounded ancestor. BF16 source precision diagnosis remains
valid but no exact source-gradient/BF16 derivative claim is made here.

## Next exact action: METH-249 continuous pilot,then codec only if qualified

Implement and freeze a fit-bound **nondeployable continuous** comparison
before constructing a new private codec. Use same source/capture/route
identities,counts/keys and original parent feature variances,tau1024.
Compute actual parent predictions on each child fit support. Project its
target residual into the fixed Lp column space using a qualified FP64
left pseudoinverse. Fit centered delta_right with the existing fixed
variance-scaled ridge,zero prior in supported cells. A separately shrunk
full-output mean bias can retain the original n/(n+1024) rule. Parents,
output basis and prior strength stay fixed. No new source-point value reset.

For exactly zero centered feature support,labels contain no slope evidence.
Use only a source-derived **projected derivative** correction in that
unsupported cell,with its fit-point contribution canceled by bias. Derive
the source-minus-parent response through Lp and the unchanged analytic
feature Jacobian;qualify solver/projection/conditioning and coefficient
growth before using it. No ID-specific forced perturbation or copied-
expert counting. Fix this fallback and all gates in the protocol before
observing results;its source derivative is a smooth FP32 sensitivity,not
the mathematical derivative of BF16 rounding.

Require unchanged parent predictions/weights,real distinct decoded
functions,finite/conditioning/normal residual and raw-mean controls.
Only after fit prerequisites,compare continuous E160 versus actual E16,
rotated-child and source controls on the same consumed windows,with the
unchanged absolute1%,10% count/rotated/prior and positive paired-gain
requirements. No fresh/full quality or deployable acceptance follows.
Failure closes this fixed full-parent/tied-latent hypothesis before codec
or native work. Pass licenses separately frozen encoding and actual C
bank checks. Resources/budget/source fallback must be fixed before execution.

## Candidate private format if the continuous hierarchy passes

To avoid requantizing parent-right coefficients,keep its BF16 right array
and add **separate row-Q8/FP32-scale delta_right**,32x4864,plus child bias.
Latent is BF16-parent-right dot phi plus separately scaled int8-delta dot
phi,then one shared BF16 left projection. Use an actual fused/one-team C
operator with declared addition order;it is not the METH-246 executable.
Extra private payload `(32*4864+32*4+896*4)*24`=**3,824,640bytes** per child
across24 layers. Source/baseline factor-bias ideal553,738,244 becomes
**557,562,884 addressed bytes**,below560MB. These are shape calculations,
not quality/rate/DRAM measurements. Delta quantization must conserve its
small learned changes;do not assume low dynamic range guarantees that.

Future native source fixtures must exercise nonzero private deltas,
require real serialization/arithmetic and three fixed six-thread24-layer
passes<=10ms before encoded bank promotion. Learned n,private residency,
route cost/LUT and real DRAM need actual banks,not copied factors. Full
independent prediction/generation/tasks and>=50 acceptedtok/s on the same
artifact remain required. Sparse GigaChat needs its own source-bound
feature-sharing/operator variant;dense Qwen evidence does not transfer
automatically to actual10B/100B capacity.
