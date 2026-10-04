# M439: failed438 comparator cancellation diagnosed at exact same anchor

Frozen e609781, terminal exit0, no optimizer updates. [Protocol](METH_439_SWITCH_READOUT_FD_PROTOCOL_20261004.md),
[raw](meth439_switch_readout_fd_result.json) SHA256
530bb57fa82ca84d8fb52f0521fee8b4504d5ddab7f496654125860251e480f2.
ALL5 apparatus gates PASS, SAME failed438 C discrepancy reproduced exactly
3.690171289699145e-5. Immutable437 qualification prefix, identical actual own-
input fixture/seed437/rank32/zeroC/target/h1e-4/direction/offsets/reference.

| Same C directional derivative |Value / error |
| --- | --- |
| Ordinary F64 autograd |7.148142532626806e-6 |
| Native local approximate gradient |7.148143772870791e-6 |
| Old head-endpoint subtraction FD |7.148406311330292e-6 /3.690171289699145e-5FAIL |
| Rationalized RMS/head delta FD, SAME h1e-4 |7.148142532626803e-6 |
| SAME stable FD h/2 |7.148142532626800e-6 |
| Fixed Richardson |7.148142532626798e-6 /1.06647e-15PASS |
| Independent analytic |7.148142532626800e-6 /8.29479e-16PASS |
| Independent complex-step h1e-20 |7.148142532626790e-6 /2.25144e-15PASS |

Stable SAME h already agrees; no smaller training step or changed tolerance.
The failure comes from subtracting nearly identical full-head logit endpoints
inside the loss-difference checker, not demonstrated curvature/training instability.
Rationalized smooth RMS difference followed by head multiplication avoids this
cancellation, while fixed native offsets cancel without refresh/differentiation.
Base independent derivatives also agree<=4.18e-15; D/feature/scores structurally
zero at Czero and exact in all methods. ALLfive fields native/reference<=1e-3,
Richardson/analytic/complex/reference<=1e-5, same floor1e-8. Native C versus stable
FD1.73506e-7, base1.00803e-7. Full endpoints/gradients/offsets/inputs/directions/
parameters/projections retained, not a gradient through native rounding.

19.984s/1268330496B peak/14887817848B hashes. One full qualification/diagnostic
archive2269822B. Fresh original256 payload/manifest and needed captures/archive,
parent438/full helper bindings exact. CPU0/Torch1/BLAS1, noGPU/T4/network/fit/
C edit; original publisher daemons preserved.

Licenses ONLY NEW440 numeric wrapper/controller for the SAME fixed shared-readout
experiment. Tiny nonzero all-coordinate checker immutable437; actual zero-C
initialization uses immutable439 rationalized delta/analytic/complex/Richardson
checks on BOTH arms. Shared rank32/seed437/3passes/192updates/loss/matched control/
oracle thresholds/resources unchanged. First failures437/438 remain retained.
No model, quality, causal useful-capacity/rate or generalization result yet.
