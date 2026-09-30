# METH-202: amplifying trained child deviations does not help the viewed loss

The [frozen protocol](METH_202_CHILD_DEVIATION_GAIN_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth202_child_deviation_gain.py)
were committed at `168b9cf` before execution. The run bound the exact
METH-175 E12,800 BF16 bank, METH-173 24-document manifest and METH-179
exact-route result by SHA-256. Eight-prompt initial BF16 donor parity
passed; the unchanged λ=1 arm reproduced **every METH-179 shift-0
document NLL exactly**. Only the nine content-child B matrices per
parent were changed for the other arms; the structural child, mean,
shared A, parent route, gates, attention and core remained fixed.

On the 24 already-viewed documents (98,571 bytes), pooled BPB was
1.067704254 at λ=0, 1.067669147 at λ=1 and 1.067793026 at λ=4.
Thus the stored deviations have a tiny favorable effect relative to
collapsing them to their mean (0.000035107 BPB), but quadrupling them
is **0.000088772 BPB worse than λ=0** and 0.000123879 worse than λ=1.
The paired 10,000-resample document bootstrap, seed 202202, gives a
fifth percentile of -0.000238414 BPB for λ=0 minus λ=4 (median
-0.000090386). Both frozen directional gates fail: the required
λ=4 gain ≥0.0002 BPB is absent and the bootstrap fifth percentile
is negative.

The [raw result](meth202_child_deviation_gain_result.json), SHA-256
`803b6069ad44ea72a9f758314ec1d458a0b45826a74dfa1eff7330f6fc96fbf0`,
contains each document's NLL/BPB, bindings, parity, bootstrap, gates
and runtime. The local RTX 3060 run took 152.422 s, with 4.673 GB
peak GPU allocation and 15.659 GB ending RSS, within the frozen
budgets. No T4 or new source was used.

**Decision:** do not amplify or merely extend training of this same
fixed-route child-residual mechanism. The result joins METH-178/179/201
in motivating a change to route/training coupling that gives selected
specialists a measurable function. The diagnostic does not prove that
all larger expert pools fail, nor does it establish external quality,
CPU LUT scaling, 10B/100B transfer or full native accepted-token rate.
