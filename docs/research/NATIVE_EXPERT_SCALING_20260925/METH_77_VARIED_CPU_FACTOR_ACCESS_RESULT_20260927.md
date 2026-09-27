# METH-77: varied learned E1280 factor access remains near E128 cost

**Decision.** On 256 actual, source-disjoint prompt positions per arm,
the trained METH-71 E1280 bank selects 421–485 different experts per
layer, touching an estimated 315.0 MB of distinct BF16 factor rows
across 24 layers. The single-thread CPU component route and selected
factor medians remain near the matched E128 control. This is stronger
cache-pressure evidence than METH-76's one repeated input, though
finite repeated positions still cannot prove cold random DRAM cost,
full-model rate or quality. METH-72/74 promotion failures stand.

The [protocol](METH_77_VARIED_CPU_FACTOR_ACCESS_PROTOCOL_20260927.md)
was committed at `4754352`. The
[capture](../../../benchmarks/native_expert_scaling/meth77_capture_varied_hidden.py)
and [C benchmark](../../../benchmarks/native_expert_scaling/meth77_varied_cpu_factor_access.c)
were committed at `4e1c789` before either capture. The same 24
METH-72 external prompts yield 11 evenly spaced positions in each
of the first 16 and 10 in each of the last 8, 256 total. The
position/source selection hash is the same for both arms:
`6dde11e289ba7c41c61f90abb618dac01382e731b729ac203a7d4a03f50be259`.
Each arm captures its own model's actual BF16 pre-FFN hidden state
at those positions for every layer. Source/checkpoint/bank hashes,
dimensions and complete vector readback are checked before timing.

| 24-layer single-thread measure | E128 | E1280 |
|---|---:|---:|
| Unique selected experts/layer, 256 positions | 119–128 | 421–485 |
| Sum of unique selected expert rows across layers | 2,967 | 10,987 |
| Estimated unique BF16 A+B factor bytes | 85,069,824 | 315,019,264 |
| Nominal selected top-four factor bytes/token | 2,752,512 | 2,752,512 |
| Route median, five repetitions | 1.043199 ms | 1.070901 ms |
| Selected-factor residual median | 1.128511 ms | 1.097234 ms |
| Route + residual median sum | 2.171710 ms | 2.168136 ms |
| C process RSS with bank and vectors loaded | 131,305,472 B | 923,734,016 B |

The E1280/E128 route median ratio is **1.027×**. The E1280 residual
median is 0.972× E128, but five-repetition ranges overlap (E128
1.098–1.217 ms; E1280 1.062–1.130 ms); no faster-factor claim is
warranted. The router, residual, finite-output and vector-format
checks completed without error on both arms. C elapsed time was
3.39/3.52 s, below the two-minute budget. Host: AMD Ryzen 5 3600X,
Clang 21.1.8, Windows. The
[E128 raw log](meth77_e128_native_raw.log) and
[E1280 raw log](meth77_e1280_native_raw.log) retain all five
repetitions, layer coverage and checksums.

The [E128 capture ledger](meth77_e128_capture_result.json) binds an
11,010,072-byte vector file with SHA-256
`260c11764bc96dfea36ad681ada64e00da365772ebbbf5c14cc4c86a5e03fb33`;
the [E1280 ledger](meth77_e1280_capture_result.json) binds its
same-size vector file with SHA-256
`2e9d733a8937a9522157d955e894952759c58127547c888e596cc784b728656d`.
These generated local files and the METH-76 banks are not committed
and can be reproduced from the bound checkpoints. GPU peak allocated
for capture was 1.282/2.918 GB, end RSS 2.888/6.842 GB; no T4.

The measured component still uses FP32 product keys and BF16 factors.
It has no compact LUT code, donor core, tokenizer, KV cache, logits or
sampling. The actual 10B→100B target also needs 10× **useful distinct
learned** experts on quality-valid donors, a RAM-sized bank with
acceptable varied-token factor traffic, and ≥50 accepted tokens/s
for the full native artifact. METH-76/77 support the CPU product-key
component cost at E1280; they do not establish those remaining gates.
