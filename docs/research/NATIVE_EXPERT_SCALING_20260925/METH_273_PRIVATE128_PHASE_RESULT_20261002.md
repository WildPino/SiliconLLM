# METH-273: actual private128 matrix phases dominate

Apparatus frozen `a0cf77d`,session74210 exits0; no repair/model job. All
source/fixture/helper bindings and361 segment readbacks pass. Entire384-vector
check including header is bitwise equal271;256-state checksum79.303890132
unchanged. Same arithmetic,added clocks/single/barrier instrumentation only.

| Instrumented phase | Median ms per24-layer token |
| --- | ---: |
| Input decode/team entry | .033983203 |
| Shared Q8 gate/up plus LUT | 6.729085937 |
| Private128 BF16 plus LUT | .511121094 |
| Concurrent mixed down/rank32 right | 3.672121094 |
| Rank32 left/bias/finite | .206849219 |
| Team exit | .134994922 |

Private plus entry/exit share6.0249%,below20%; shared plus concurrent
down/right share92.1427%,above80%. Frozen decision chooses matrix/operator
layout or memory execution next. Do not attribute all these intervals to
DRAM,expf or one organ: shared interval includes scheduling,LUT/product;
down/right remain concurrent. Added measurement/barrier overhead is retained.
Instrumented totals12.018791/10.902765/11.312503ms are diagnostic only and
cannot qualify rate or reclassify271/272 cost stops.

Runtime10.657s after imports,RSS42,008,576bytes,native peak354,271,232bytes,
CPU only,six threads,no active job. Raw [result](meth273_private128_phase_result.json),
SHA256 `d8918584b3c7d9ac323347e979600a15099559cacc3016c92c132f1144225643`.
All18 intervals,source/executable/compiler/check/helper hashes are retained;
command in [protocol](METH_273_PRIVATE128_PHASE_PROTOCOL_20261002.md).

Next prospectively test a changed shared-projection operator: dynamic I16
input representation with stored Q8 rows and exact integer accumulation,
preserving private FP32/BF16 features/LUT/readout. This changes input
arithmetic,so source-BF16 fidelity and matched GPU/native checks must precede
cost/full composition. It is not a claim that I16 is already sufficient
or that shared projections alone establish full>=50,useful n or transfer.
