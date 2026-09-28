# METH-124: quality-valid centered E1280 factors pass native CPU parity

**Decision.** A versioned native bank containing the METH-121/123 quality-valid effective E1280 A/B factors and unchanged hierarchical router passes all 96 parent-route, child-route, BF16-gate and BF16-residual fixtures exactly. A one-byte expected-residual mutation makes the checker fail. Hot repeated-input component timing is 2.450 ms/token-equivalent for 24 layers; varied-token factor traffic and the matched E128 control remain METH-125 work.

The [protocol](METH_124_125_CENTERED_FACTOR_CPU_PROTOCOL_20260928.md) was committed at `b3e4164` before export, and the [exporter](../../../benchmarks/native_expert_scaling/meth124_export_centered_factor.py) and [C checker](../../../benchmarks/native_expert_scaling/meth124_centered_factor_cpu.c) at `82e7e1f`. The exporter binds the METH-56 parent, METH-107 child, BF16 donor and first METH-121 external prompt. It saves FP32 parent/child router tensors and BF16 forward-value A/B factors after applying `parent_B + child_B − mean_10(child_B)`, checks every tensor byte on readback, and captures four real/synthetic hidden-state fixtures per layer.

The [export ledger](meth124_centered_factor_export_result.json) binds an 893,141,032-byte bank with SHA-256 `8e4ef1f8abea39d9562e8b617f460e89591fa2b2d7fa7478fac0901804064580` and fixtures with SHA-256 `d4a25de436db554744288011379daf3b0672e5ca25e8d05acb80ad533ef54677`. The local generated bank is ignored by Git and reproducible from the bound checkpoints. The [native log](meth124_centered_factor_native_raw.log) records 96/96 exact matches, maximum gate error 0 and maximum residual error 0; the [negative log](meth124_centered_factor_negative_raw.log) records one failure after flipping one byte of the first expected residual coordinate.

| Single-thread Ryzen 5 3600X, 24 layers, hot repeated state | Median ms/token-equivalent |
|---|---:|
| E128 parent route | 0.960 |
| E1280 parent plus child route | 1.444 |
| E1280 route plus selected BF16 A/B residual | 2.450 |

Five repetitions of 1,024 token-equivalent passes give combined range 2.438–2.506 ms. The bank resident process used 897,560,576 bytes RSS, with 2,752,512 selected A/B bytes nominally addressed per token across 24 layers. Export peaked at 2.907 GB allocated GPU memory and ended at 4.687 GB RSS. Clang `-O3 -std=c11 -Wall -Wextra` emitted no diagnostics. This is a hot-input component test; no factor LUT code, varied-token/cold DRAM guarantee, dense donor core, attention, logits, tokenizer or full `engine.c` accepted-token rate is included.
