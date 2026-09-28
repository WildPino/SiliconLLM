# METH-125: quality-valid centered E1280 factor access passes varied CPU component gate

**Decision.** The centered METH-123 E1280 bank passes the preregistered varied-token CPU component gate. Its 24-layer single-thread route plus selected-factor median sum is 2.510 ms/token-equivalent, 1.245× the matched METH-56 E128 control's 2.017 ms, below the fixed ≤4 ms and ≤2× limits. This is a component result; it does not establish LUT-coded arithmetic or full accepted-token throughput.

The [protocol](METH_124_125_CENTERED_FACTOR_CPU_PROTOCOL_20260928.md) and [METH-124 parity result](METH_124_CENTERED_FACTOR_CPU_RESULT_20260928.md) bind the quality-valid checkpoints. The [parent exporter](../../../benchmarks/native_expert_scaling/meth125_export_parent_control.py) produced a 93,732,904-byte E128 control bank, SHA-256 `4b9e3588f804f86751b46f651a917125fc8d48f4944e20c50fda3653bf0cac0b`. The [capture](../../../benchmarks/native_expert_scaling/meth125_capture_centered_varied_hidden.py) used all 24 METH-121 external prompts, selecting 11 evenly spaced positions in each of the first 16 and 10 in each of the last eight. The selection hash is identical for both arms: `7f951640ab6b5dbb6ea0c315487ff543f8470eefce1f5fc780306b1ad3240d3c`. Each model supplied its own actual BF16 pre-MLP states at those same source positions. The vector SHA-256 values are `525fe163656ed4a3098f1cd45cdc2bf6a6be1d6082bddae9465382becacdad0d` for E128 and `f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699` for E1280.

The [native benchmark](../../../benchmarks/native_expert_scaling/meth125_varied_centered_factor_cpu.c) verifies bank/vector formats and finite routes/residuals, then reports five repetitions in source order. It was committed at `72a7a09` with the vector binding before timing. The [E128 log](meth125_e128_native_raw.log) and [E1280 log](meth125_e1280_native_raw.log) retain every repetition and per-layer coverage.

| 24-layer single-thread measure | E128 | Centered E1280 |
|---|---:|---:|
| Distinct selected experts/layer, 256 positions | 81–120 | 252–402 |
| Sum of distinct selected factor rows across layers | 2,426 | 8,145 |
| Estimated unique BF16 A+B factor bytes | 69,558,272 | 233,533,440 |
| Nominal selected A+B bytes/token | 2,752,512 | 2,752,512 |
| Nominal router bytes addressed/token | 5,652,480 | 8,527,872 |
| Route median, ms/token-equivalent | 0.990 | 1.480 |
| Selected-factor median, ms/token-equivalent | 1.027 | 1.031 |
| Sum of medians, ms/token-equivalent | 2.017 | 2.510 |
| Resident process memory | 130,719,744 B | 930,131,968 B |

The factor working set of 233.5 MB is much larger than a typical last-level cache, and the bank is 9.53× the E128 control's physical size. The selected factor bytes/token remain fixed because four experts of rank eight are active in both arms. The nominal router count includes all parent projection/keys, the child projection and only the 40 child keys for four selected parents; it does not claim every byte of the 12.34 MB router bank is read each token. Five repeated passes over only 256 positions can still recycle rows in cache, so this is varied-token pressure rather than cold random DRAM. The 1.245× total cost increase is mainly router work; the factor medians differ by 0.0038 ms and do not support a meaningful factor-cost difference. Each native arm completed in under five seconds, below the two-minute limit. E128/E1280 capture peaked at 1.275/2.929 GB allocated GPU memory and used 2.896/3.281 GB RSS.

The CPU code reads BF16 factors and uses FP32 router dot products. It does not implement a compact LUT factor code or the donor core, attention, KV cache, logits or tokenizer. No full `engine.c` accepted-token rate or 10B/100B donor-family transfer has been measured.
