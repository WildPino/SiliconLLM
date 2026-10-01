# METH-217: five local parents expose a soft-to-hard calibration gap

Frozen protocol/runner `1ab7270`; original router and all model/factor
weights unchanged. Eight-prompt teacher/control logit parity is exact.
All 3072 raw fit sequences, totals and every metric/gate on layers
12/16/17/20 reproduce METH-216 within 1e-6. No reserved/source input
or ChatML fit inference is opened by this diagnostic.

| Layer / parent (zero based) | Selections | Exact-state floor | Hard max share | Soft max share at T=.0004 | Median top-two margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 / 359 | 2208 | 0.453% | 30.163% | 11.1123% | 0.00002006 |
| 16 / 600 | 2204 | 0.454% | 28.811% | 11.1124% | 0.00004876 |
| 16 / 889 | 2214 | 0.452% | 26.874% | 11.1121% | 0.00003654 |
| 17 / 158 | 513 | 1.949% | 26.706% | 11.1122% | 0.00005430 |
| 20 / 786 | 2216 | 0.451% | 29.874% | 11.1120% | 0.00001746 |

Every offending parent has balanced final soft counts and an exact-state
floor far below 25%. Exact top-two ties range 0–0.136%; margins <=1e-6
occur on 1.754–4.016% of selections. The failure is therefore the
specified soft-to-hard gap, not a >25% identical-state impossibility.
The local centroid rows are distinct, with median pair cosines
0.998789–0.999913. Their narrow score separations also explain why the
fixed final temperature remains soft relative to many deployed choices.
This interpretation does not prove that a static hard partition will
generalize or tolerate native floating-point differences.

Decision: freeze a hard-count calibration of these five raw parent bias
vectors, with all keys/projections/ChatML and other raw parent biases
unchanged. Reuse saved fit states, retain all original fit/reserved/source
gates, and stop on any failure. Do not add soft iterations or train B
from this result. Native cost/precision, useful specialists, independent
quality, 10B/100B transfer and accepted rate remain unverified.

Capture: 795,341,870 bytes, exact readback, SHA256
`ce3d11eec323e251deba08f7e70aef15d527f6881e3f35763b0ebcf2b4c541db`.
It saves full FP32 q, int16 parent IDs and sequence offsets for all four
layers, permitting subsequent fit diagnostics without model recapture.
Local RTX3060 runtime after imports: 543.437 s; end RSS 5.137 GB,
peak allocated GPU 2.945 GB. No T4. Session 23384 completed with exit 0.
[Raw result](meth217_local_key_concentration_result.json) SHA256
`25c79ebe507cb5650fbd434983ec888d678ef3166181f3f50728cf894edcc561`.

Command:
```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth217_local_key_concentration.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth217_local_key_concentration_result.json --capture results/native_expert_scaling/meth217_raw_four_layer_capture.npz
```
