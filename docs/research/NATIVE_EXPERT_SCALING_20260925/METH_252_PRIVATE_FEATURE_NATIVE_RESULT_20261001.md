# METH-252: private source features restore fidelity but miss cost

Frozen bb3b920, session92140 exits0. Actual32 original BF16 gate/up rows
replace selected shared approximate features in each layer; all289 prior
base/LUT/readout segments remain byte equal. All source-only selector,
361 readback, private feature effect, native change and numeric gates pass.

Source256-state x24-layer FFN pooled SSE/energy falls from
.000137500193654236 to **.000026084689113638605**, an81.0293% reduction.
Worst layer .00028943036698593375. This is original BF16-weight/FP32
source-prefix function evidence, not full-model/independent quality or
useful conditional count. Native384-vector median relative L2
4.2104337507064835e-7, maximum9.404715464405225e-7, both pass.

Native passes10.299395/10.503606/9.787273ms, median **10.299395>10**.
Close this fixed branch-per-feature kernel before cell selection. Function
fidelity alone does not qualify its active cost. Do not simply retime it.

Physical fixture329,334,308bytes SHA256
`b90193f090812b9fd991616c0150f28f16a6553f1e2af2c1af8d8f5a489ba395`;
native check SHA256
`b9e1bb6c8a135c63ffcf5f78014a3bd9dc9693770602079aa134dc7fac867714`.
[Raw result](meth252_private_feature_native_result.json) SHA256
`7da69e9d44df64719fddbfb62a452a2b4afb3be7044511f1749d16305b6910ce`.
Runtime27.984s after imports, RSS1,703,059,456bytes, GPU peak223,927,296,
native peak345,726,976. Local RTX3060/six threads, no T4/download.

Freeze a changed execution mechanism with these same physical functions:
use the original unbranched shared feature workshare, then overwrite32
private features in a separate workshare/barrier before unchanged readout.
It computes the selected shared rows redundantly but removes the private
branch/map from all4864 shared rows. Numerical output/checksum must be
bitwise equal before prior fidelity can carry. All cost thresholds and
three-pass discipline unchanged. No learned private bank/large-n/route/
DRAM/full quality/rate is yet eligible.
