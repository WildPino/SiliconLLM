# Stage 1 (c): the existing ternary kernels at Ling-flash's shapes on the 3600X. RESULT

2026-11-03. Protocol: `S1E_TERNARY_KERNEL_PROTOCOL_20261103.md` (pushed `fae1305`, before the timed run). Records in
`benchmarks/stage1/results/`:
- `s1k_run.json`: every launch, with its command, CONFIG, REP and RESULT lines, CPU load before it, binary and TL2
  weight hashes;
- `s1k_verdict.json`: from `s1k_verdict.py`, selftest PASS;
- `s1_ceiling.json`: from `s1_ceiling.py`, the desk ceiling with the measured rates (DESK-MODEL).

## Outcome in one line

**BETWEEN.** The best checked ternary kernel is ik_llama.cpp's **IQ2_BN at 113.34 [112.65, 113.72] G weights/s** on
the routed experts, at 28.6 GB/s: 8.7 under the 122 the success line needs, 40 above the FAIL line. On the attention
shape the same kernel streams **155.63** G weights/s (39.1 GB/s), above the 128 line.

## The run

- 2026-11-03, 09:22 to 10:02, on an idle machine with the user's go-ahead. Pinned binaries, hashes checked.
- 26 cells × 3 rounds = 78 launches: 59 COMPLETE, 19 VOID busy (load 10.3-24.7% before the launch). Of the 19 retries,
  14 COMPLETE and 5 VOID busy.
- Every key has 3 rounds except five with 2 (TL2 experts2d stream and resident, IQ1_BN experts, IQ2_BN attn, mainline
  Q4_K experts). The rule needs 2.
- Correctness: every checked launch has rel_err ≤ 0.0098 and wrong-rows rel_err ≥ 1.215. IQ2_BN_R4 is reported, not
  judged.

## Controls

| Control | Rule | Measured | |
| --- | --- | --- | --- |
| C1 the wall | Q8_0 experts at 29-45 GB/s | 37.17 GB/s | pass |
| C2 streaming | TQ2_0 resident / streaming ≥ 1.15 | 1.35 | pass |
| C3 correctness | rel_err < 0.02, wrong rows > 0.5 | ≤ 0.0098, ≥ 1.215 | pass, every judged format |

## Rates (G weights/s, median of rounds [min, max]; streaming unless marked)

| Format (library) | bits | experts | experts2d | attn 4096² | resident experts |
| --- | ---: | --- | ---: | ---: | ---: |
| IQ2_BN (ik) | 2.0 | **113.34 [112.65, 113.72]**, 28.6 GB/s | 77.92 | **155.63**, 39.1 GB/s | 146.33 |
| IQ2_BN_R4 (ik), unchecked | 2.0 | 115.31 | | | |
| TQ2_0 (mainline) | 2.06 | 107.01 [104.99, 107.54], 27.6 GB/s | 58.44 | 110.41 | 144.69 |
| TQ1_0 (mainline) | 1.69 | 107.93, 22.8 GB/s | | 112.08 | 101.46 |
| IQ1_BN (ik) | 1.625 | 53.73 | | 61.09 | 80.30 |
| TL2 (bitnet.cpp + glue) | ~1.69 | (2-D only) | 56.98 | 117.75 | 55.05 (2-D) |
| I2_S (bitnet.cpp) | 2.0 | (2-D only) | 19.38 | 48.82 | 26.01 (2-D) |
| Q2_K (mainline), control | 2.625 | 88.56, 29.1 GB/s | | | |
| Q4_K (mainline / ik), control | 4.5 | 60.18 / 61.59, 33.9 / 34.6 GB/s | | 61.03 | |
| Q8_0 (mainline), control | 8.5 | 34.99, 37.2 GB/s | | | |

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| BETWEEN, more likely than PASS; FAIL very unlikely | **hit**: BETWEEN |
| best ternary on the experts a 2-bit format, 105-128, at 26-33 GB/s | **hit**: IQ2_BN 113.34 at 28.6 |
| 1.6-bit formats below the best 2-bit one | **hit**: TQ1_0 107.93, TL2 56.98, IQ1_BN 53.73 |
| I2_S the slowest, under 73 | **hit**: 19.38 |
| attention faster than the experts, same format | **hit**, all six formats |

## Read here, not registered

- **The gap to 122 is in the expert dispatch, not in the RAM or in the arithmetic.**
  - The same IQ2_BN kernel streams 155.63 on 4096 × 4096 (39.1 GB/s, above stage 0's 32 GB/s wall) and runs 146.33
    with the experts resident. On the routed experts it gives 113.34.
  - An expert matrix holds 4 M weights; each product pays a thread barrier and the `mul_mat_id` bookkeeping. Streaming
    costs IQ2_BN 23% of its resident rate, TQ2_0 26%.
  - A kernel of ours would therefore work on the dispatch: gate and up fused in one pass, fewer barriers per layer.
    ik_llama.cpp's fused up-gate path was not in the registered cells.
- **Fewer bits do not buy speed on this CPU.** TQ1_0 reads 22.8 GB/s against TQ2_0's 27.6: at 1.69 bits the unpacking
  is the limit. TL2 runs the same resident and streaming (55 and 57): its one-thread activation tables bound it.
- **The 2-D path pays a per-product overhead.** One plain product per matrix costs IQ2_BN 31% and TQ2_0 45% against
  `mul_mat_id`. The ceiling below charges the shared expert at this rate, so it is conservative on that organ.
- **The resident cell carries its own overhead.** TQ1_0 runs slower resident (101.46) than streaming (107.93): one
  layer per graph puts more barrier per weight. Resident rates are a floor on the arithmetic, not a ceiling.
- **Q2_K, the best non-ternary format, streams 88.56**: 78% of IQ2_BN's rate.

## Decision (the registered rule)

- **BETWEEN.** The measured rates enter the desk model (`s1_ceiling.py`); attention at its measured rate.
- **The gap to 122:** 8.7 G weights/s on the routed experts. The resident rates (146.33 IQ2_BN, 144.69 TQ2_0) leave
  room for it: the arithmetic is above the line.
- **Scale of a stage 2 training:** IQ2_BN keeps one scale per row, (b)'s format one per 256 weights. TQ2_0 carries
  (b)'s format exactly, at 107.01.

## Thesis line: the desk ceiling with the measured rates

Ling-flash, context 2048, decode-only k = 4 (stage 1 (a)), every organ costed at its measured rate (DESK-MODEL; byte
map, wall and fixed term from stage 0):

| Scenario | tok/s |
| --- | ---: |
| all organs ternary, at the wall, ring-linear attention (stage 0's scenario) | 25.0 |
| all organs ternary, at the wall, full q8_0 KV | 26.1 |
| all organs ternary, IQ2_BN measured, ring-linear | 24.5 |
| **all organs ternary, IQ2_BN measured, full q8_0 KV** | **25.6** |
| all organs ternary, TQ2_0 measured, full q8_0 KV | 21.4 |
| experts Q2_K measured, the rest ternary IQ2_BN, full q8_0 KV | 23.3 |
| experts ternary IQ2_BN measured, the rest as stored (Q4_K_M, f16 KV) | 13.6 |
| k = 4 alone, Q4_K_M as stored | 10.8 |

- **Ring-linear attention costs speed at context 2048.** Each linear layer reads and writes its fp32 state per token:
  117 MB against a 71 MB q8_0 KV cache. Full KV is 1.1 tok/s faster here.
- **Every weight-bearing organ has to be ternary.** With IQ2_BN and full KV, leaving any one organ at Q4_K_M drops the
  ceiling under 24.7: dense FFN out gives 24.6, shared out 24.1, head out 19.4, attention out 18.1, routed out 18.3.
- The success line is reachable on the desk with the kernels that exist, at 1.04× margin. Only the routed experts have
  been tested on quality at ternary.

## Next

The stage 1 delivery: `S1_DELIVERY_20261103.md`.
