# METH-331: actual probability mismatch has two contributions

Freeze211a1c1, observations85ca4b3. [Raw record](meth331_switch_router_diagnostic_result.json)
SHA256668a01a42276ee51955a9cde930f224cbb57180016bc7e88f1696226738948a1.
[Protocol](METH_331_SWITCH_ROUTER_DIAGNOSTIC_PROTOCOL_20261003.md), driver
benchmarks/native_expert_scaling/meth331_switch_router_diagnostic.py.

Trace source differs from330 only by explicit input/raw-logit logging. Both
full native output files **byte EXACT330**; source inputs/weights/config/math
unchanged. All78/84 actual route rows captured and original mismatch exactly
reproduced; signed replay decomposition identity EXACT zero for every row.
Complete original reference encoder/decoder/logit/routes and both native/
original router input/logit arrays saved as NPZ, exactSHA in raw result.
These are consumed numerical apparatus states, not untouched quality data.

Worst France row is encoder.block.1.layer.1.mlp.router, token7, expert30:

| Measured contribution | Signed probability error |
| --- | ---: |
|Total native minus original|-1.63912773e-6|
|Native classifier/softmax minus official replay at SAME native input|-7.45058060e-7|
|Official replay native input minus original input|-8.94069672e-7|
|F64 dot+softmax counterfactual at native input versus original|-1.50269013e-6|
|F64 dot+softmax at original input versus original backend|-8.07676060e-7|

Native normalized inputL2 difference6.99439e-7. Same-input native raw-logit
maxabs toF64 **1.17679e-6**, original backend **3.13430e-6**; native softmax
rounding contributes only1.85823e-8 at worstrow. Source F32 backend itself
has finite accumulation difference from F64 mathematical dot; neither is
exact real arithmetic. Counterfactual router-only F64 still exceeds original
1e-6, so this diagnostic does NOT justify isolated-router repair/promotion.

Experiment case worst row encoder.block.3.layer.1.mlp.router/token7/expert173:
total8.04663e-7=7.15256e-7 same-input+8.94070e-8 upstream, observed unchanged
PASS at1e-6. Its F64 router-only max7.59362e-7 stays below guard.

62.781s main excluding imports, max sampled combinedRSS6,775,226,368B,
end controller4,581,777,408B. Exec3260 exit0. No benchmark/GPU/quality result.
Original329/330 remain failed; no threshold/data/seed waiver.

Next specific variable frozen332: all matrix-vector products/sums use F64 on
original F32 operands then F32 output, addressing prior upstream projections
AND classifier together; prior330 normalization retained, attention dot F32.
Full actual source+Tiny composition guards unchanged, effect must be observed.
Different mathematical accumulation is numerical apparatus work, not final
compact representation or SAMEartifact>=50/quality/useful larger-n proof.
