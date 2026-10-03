# METH-333 result: independent-reference primitive gate FAIL

Frozen d28241f. New specified arithmetic estimand explicitly distinct from
original-donor quality; original329/330/332 failures unchanged. Command in
protocol. Authoritative exec63849 exit1. No model/source transfer, C compilation,
Tiny or source forward occurred; failure at bindings primitive assert.

Raw `meth333_switch_full_source_result.failure.json`, SHA256 `bbd5351b1a384b6112f1e3fa81c3e677c675e3f6a2547239c75e9b95d8bc3a17`.
Primitive seed333: matrix and attention F64/F32 rounded outputs EXACT NumPy;
softmax maxabs1.86264514923e-9 PASS; norm maxabs7.15255737305e-7 >2e-7 FAIL.
Main8.265s after imports; max sampled RSS not populated before primitive stop.
This is an unsupported independent reference primitive, not native C or donor
quality rejection. Preserve raw and frozen helper/controller/protocol unchanged.

Next prospectively implement an explicit correctly rounded norm primitive:
F32 variance-plus-epsilon, square root in F64 then F32, reciprocal of rounded
root in F64 then F32. Check SAME NumPy F32 oracle/2e-7 gate; no threshold/data
change. This addresses uncertainty in Torch F32 primitive implementation;
no causal primitive attribution inferred solely from this whole-norm mismatch.
Original333 native C can be reused unchanged if new reference qualifies.
All whole/source correctness, original-donor untouched quality/cost/accepted
rate/useful n/generalization remain unproven. No goal promotion.
