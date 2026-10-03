# METH-330: stable RMS preserves all source bytes, probability gate still FAIL

Freeze a1548eb, observations f0a3129. [Raw record](meth330_switch_full_source_result.json)
SHA256b28dc0f5a8a289be7aa0c23270d3abba2f7b67d07904dc576153618e0bc46745.
[Protocol](METH_330_SWITCH_STABLE_RMS_PROTOCOL_20261003.md). New copy of328 C
verified EXACT with single norm replacement; original329/328 retained.
F64 sum of individually F32-rounded squares, F32 mean/rsqrt/scaled outputs.
No other arithmetic or weight change. New engine opt-in only.

All6392 actual native tensor hashes again EXACT327. Source mapping/spec
unchanged; same original14.664B/12x256 banks/tokenizer/contexts/config.
Both exact328 Tiny weight archives reproduced, full Tiny states/choices/
capacity/probabilities pass unchanged numeric gates.

| Actual source case | Encoder pooledL2 | Decoder pooledL2 | Full logit pooledL2 | Selected probability maxabs |
| --- | ---: | ---: | ---: | ---: |
|0 France|8.55104e-7|4.18859e-7|2.51667e-7|**1.63912773e-6 FAIL**|
|1 experiment|4.16529e-6|5.56869e-7|2.07603e-7|8.04662704e-7 PASS|

All states<=1e-4, exact route choices/capacity/greedy and zero-head faults
still pass. ORIGINAL probability1e-6 still FAILS case0. Stable RMS improves
pooled logit closure but cannot be claimed to repair the qualifier or prove
variance sum was sole cause. Unchanged330 is not promoted to precision/rate.

114.063s main excluding imports, max checked combinedRSS6,814,498,816B,
end controller4,601,856,000B. Within original20min/32GiB main guard. All
complete outputs/audit/cases/Tiny artifacts/compiler/runtime logs retained in
results/native_expert_scaling/meth330_switch_full_source. Exec11533 exit0.

Next concrete uncertainty: actual native normalized inputs versus source,
same-input router matvec/softmax versus official backend. Frozen331 captures
unchanged330 actual values, requires byte-identical outputs and decomposes
signed probability mismatch with official same-input replay and F64 diagnostic.
No blind next precision/summation sweep or tolerance waiver. Final source
quality/LUT/useful n/actual DRAM/accepted>=50 and cross-family/~100B remain open.
