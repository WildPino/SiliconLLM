# METH-221: actual Q6 float-input kernel is accurate but too costly

The [protocol](METH_221_Q6_NATIVE_PROTOCOL_20261001.md), exporter/oracle
and C kernel were frozen at `2bbd5da` before execution. All 72 actual
METH-186 Q6 code arrays survive the planar conversion exactly; every
code/scale segment is read back and hash-checked. Native codec checks
64 mixed patterns, exercising all unsigned values in every lane.

| Fixed component check | Result | Limit | Outcome |
| --- | ---: | ---: | --- |
| Median relative L2, 384 token/layer rows | 4.152442e-7 | <=1e-4 | Pass |
| Worst relative L2 | 6.355672e-7 | <=5e-4 | Pass |
| Median 24-layer FFN ms/token | **21.541930** | **<=10** | **Fail** |

The three six-thread native passes are **20.297998 / 21.541930 /
21.619639 ms/token**, all failing the feasibility gate. The stored FFN
binary is 245,145,624 bytes versus Q8's 323,592,216, but fewer bytes do
not imply a cheaper kernel. This run does not isolate decode/compute
versus memory as the timing cause, or establish a statistically matched
Q6/Q8 speed ratio. No kernel or layout retry is made after outcome.

[Raw result](meth221_q6_native_feasibility_result.json), SHA256
`a2c283b3a18e39753ea122ad878c510886687cace0807516845cf98467441600`,
contains all 72 segment identities, 384 distinct numerical rows, binary/
source/executable/output hashes and resource data. Runtime after imports
31.187 s, ending RSS 1.793 GB, GPU peak 87.3 MB, C peak working set
261.3 MB. Session 8661 exited 0. No T4 or new quality source was used;
no project inference job remains active.

**Decision:** stop this specific planar-Q6/FP32 kernel before recovery
training. Direct Q6 quality and global rank-correction failures remain
unchanged. This does not refute every Q6 implementation or conditional
correction, but supplies no credible native base for the proposed pilot.
Do not keep lowering bit count while assuming compute will track bytes.

The next geometric variable should reduce active FFN operations: learn
a smaller common function and conditional pretrained-output functions
from actual donor states. Review the hard-carve/shared-residual failures
in METHOD and paused donor branch; avoid reselecting donor neurons or
repeating global weight-SVD repair. A bounded local activation-function
screen can test whether genuine learned extra choices help before full
composition, large-n CPU cost, fresh quality or cross-family/10B claims.
