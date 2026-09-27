# METH-76: learned E1280 CPU route passes component parity and cost

**Decision.** The trained METH-71 update-64 E1280 product-key
router/factor component matches PyTorch on all 96 native fixtures and
passes the fixed 3 ms/token and 2× E128 route-cost gates. Its
24-layer single-thread route median is 1.005 ms per token-equivalent
pass, only 1.027× the matched E128 control. This validates the CPU
cost of the factorized learned component under repeated inputs.
METH-72's external route-utility failure and METH-74's semantic
failure still stop model promotion. No full-model accepted token rate
or LUT-coded factor bank was measured.

The [protocol](METH_76_LEARNED_E1280_CPU_COMPONENT_PROTOCOL_20260927.md)
was committed at `5c909e5`; the [exporter](../../../benchmarks/native_expert_scaling/meth76_export_learned_product_key.py)
and [C checker](../../../benchmarks/native_expert_scaling/meth76_learned_product_key_native.c)
at `fed2cbc` before export. Each bank is derived from the exact
METH-71 update-64 checkpoint. The source prompt is the first code
row of the frozen METH-72 external manifest. The exporter writes a
versioned header, FP32 rank-64 projection/product keys and BF16
forward-value rank-8 A/B factors for all 24 layers. It checks every
exported router/factor byte on readback. All 128/1,280 A+B pairs
per layer are distinct by content hash; this does not mean every pair
is useful or visited.

| Matched component measure | E128, 8×16 | E1280, 32×40 |
|---|---:|---:|
| Physical bank bytes | 93,732,900 | 886,751,268 |
| Router bytes addressed / token, 24 layers | 5,652,480 | 5,947,392 |
| Selected top-four factor bytes / token | 2,752,512 | 2,752,512 |
| Route median, five 1,024-token repetitions | 0.978312 ms | 1.004900 ms |
| Selected-factor residual median | 0.989202 ms | 0.998079 ms |
| C process RSS after bank read | 98,148,352 B | 890,576,896 B |
| Route / gate / residual fixture failures | 0 / 0 / 0 | 0 / 0 / 0 |

The E1280/E128 route latency ratio is **1.027×**. The bank grows
9.46× physically, while the selected factor payload remains fixed
because top-four and rank-8 are unchanged. Each arm has 96 fixtures:
the first and last actual BF16 hidden states of the same prompt in
each layer, plus two fixed-seed synthetic states. All unordered
top-four sets, BF16-rounded gates and BF16 residuals match exactly
at the saved precision (maximum observed gate and residual error
zero). A +1 mutation of one fixture residual makes the checker exit
1 with one residual failure in each arm.

The [E128 export ledger](meth76_e128_export_result.json) binds bank
SHA-256 `baea213c65fa31c81f56c4453fd209f51fb51de13f855ccb39cf9d2fb77fbcf2`
and [fixture](../../../benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth76_e128_fixtures.bin)
SHA-256 `a51d07c2570ecebdaf0642ddf885e988d66cb1a8455fbd528d44a51cf2ac1588`.
The [E1280 ledger](meth76_e1280_export_result.json) binds bank
SHA-256 `026298e4dc83ffec8c4cb8a3f80249b836a428d561b5658237e877643f94c06a`
and [fixture](../../../benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth76_e1280_fixtures.bin)
SHA-256 `e180f1f31fe2da1122bd124a6aabc74a9f4efe70795d553ab7bb33b0b2a96fbd`.
The large local bank is reproducible and is not committed.
[E128](meth76_e128_native_raw.log) and
[E1280](meth76_e1280_native_raw.log) native logs retain per-case
comparisons and timing; the negative-control logs are retained too.
Clang 21.1.8 compiled the C checker at `-O3` on Windows. Export
peak allocated GPU was 1.237 GB/2.873 GB; end RSS 2.975 GB/8.590 GB.
No T4 was used.

Reproduce the C checker with:

```powershell
clang -O3 -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth76_learned_product_key_native.c -o results/native_expert_scaling/meth76_learned_product_key_native.exe -lm -lpsapi
results/native_expert_scaling/meth76_learned_product_key_native.exe benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth76_e128_bank.bin benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth76_e128_fixtures.bin
results/native_expert_scaling/meth76_learned_product_key_native.exe benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth76_e1280_bank.bin benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth76_e1280_fixtures.bin
```

**Limit.** The 1,024 passes reuse one actual hidden state per layer
and therefore repeatedly select the same factor rows. Their factor
working set can remain in cache; this is a hot-input component cost,
not a varied-token DRAM-traffic result. The FP32 projection and keys
are not LUT-coded, the BF16 bank is not a compact one-byte or ternary
representation, and the donor core, tokenizer, KV cache, logits and
sampling are absent. A varied-activation factor-access benchmark
and a quality-valid packed full model are still needed before
claiming the CPU LUT and ≥50 accepted tokens/s scale with RAM.
