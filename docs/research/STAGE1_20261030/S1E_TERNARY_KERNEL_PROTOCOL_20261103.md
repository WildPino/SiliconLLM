# Stage 1 (c): the existing ternary kernels at Ling-flash's shapes on the 3600X. PROTOCOL

2026-11-03. Follows stage 1 (b) (`S1D_TERNARY_EXPERTS_RESULT_20261103.md`): ternary experts are undecided on quality,
and (c) stays at ternary. The test is the one the stage 0 map (`WALL_MAP_RESULT_20261029.md`, §5) put first: the
ternary kernel's rate. This part measures the kernels that already exist. A kernel of our own comes only if none of
them reaches the line.

## Question

At Ling-flash's decode shapes, with 6 threads on the Ryzen 5 3600X, how many weights per second does the best existing
ternary kernel process, with the weights streaming from RAM?
- **PASS:** 128 G weights/s or more, the RAM wall at 0.25 B/weight.
- **FAIL:** under 73, the top of Q4_K's rate in stage 0 (A1: 35-41 GB/s at 0.5625 B/weight). Below that, ternary would
  not beat Q4 on this box.

## Thesis line (rule 4)

The k = 4 ceiling of stage 1 (a) is 25.1 tok/s on the desk model: Ling-flash at context 2048, 1,141 MB per token at
32 GB/s, plus 4.24 ms. It assumes every organ ternary and every kernel at the wall.

At k = 4 the routed experts read 1.56 G weights per token (390 MB). With an expert kernel at R G weights/s instead of
the wall, the ceiling becomes 1 / (39.9 ms + 1,560/R ms − 12.2 ms):

| R, G weights/s | 128 | 122 | 115 | 105 | 73 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ceiling, tok/s | 25.1 | 24.7 | 24.2 | 23.5 | 20.4 |

**The success line of 24.7 needs R ≥ 122.** The measurement runs at Ling-flash's own shapes, so it is at the thesis's
scale. Only the matrix products are timed; the rest of a decode step comes in the end-to-end of stage 3.

## Literature and the kernels that exist (read in the papers and the code)

| Source | What it measured or is |
| --- | --- |
| bitnet.cpp, arXiv 2502.11880, Table 7 (i7-13700H, tok/s) | at 100B: TQ2_0 1.61, TL2_0 1.69, I2_S 1.65, TQ1_0 1.48, T-MAC 0.73; at 7B: 19.92, 20.72, 20.62, 17.96, 12.29. TL2_0 beats T-MAC "up to 2.32x" on x86 |
| bitnet.cpp, §5 and conclusion | "the maximum bandwidth limits the potential of ELUT"; TL2 needs K a multiple of 3 (otherwise a two-weight tail) |
| T-MAC, arXiv 2407.00088, §5.2 | multi-threaded mpGEMV "primarily constrained by memory bandwidth"; tuned with TVM per device |
| llama.cpp TQ2_0 / TQ1_0 | 2.06 / 1.69 bits, one fp16 scale per 256 weights: the exact export format of stage 1 (b) |
| ik_llama.cpp IQ2_BN / IQ1_BN / IQ2_BN_R4 | 2.0 / 1.625 bits, one scale per row; _R4 interleaves 4 rows |
| bitnet.cpp I2_S / TL2 | 2.0 / about 1.69 bits, one scale per tensor; TL2's kernels are generated per shape |

Known ways to fail:
- At batch 1 the kernels are bandwidth-bound, so a 2-bit format at the wall gives 128 at 32 GB/s, and only fewer bits
  can pass it.
- TL2 builds its activation tables on one thread before every product.
- Small matrices pay a thread barrier per product: an expert matrix of 4 M weights takes about 33 µs at the wall.

## Found before registration (builds and a dry run)

- **bitnet.cpp's x86 TL2 does not run as shipped.** Its current ggml (`390c307`) calls `ggml_bitnet_mul_mat` for TL2,
  but defines it only for ARM TL1. The x86 body survives as a fragment without its batch-1 branch.
  - `s1k_tl2_glue.cpp` is that fragment at batch 1; the kernels it calls are bitnet.cpp's.
  - Two more changes in the BitNet clone: `codegen_tl2.py` gets the three shapes, and its generated header drops a stale
    `tensor->backend` check.
- **bitnet.cpp builds with gcc** (its setup asks for clang; the CMake file handles GNU).
- **ik_llama.cpp's IQ1_BN/IQ2_BN** keep a per-row scale in a row prefix that their `to_float` does not apply. The check
  applies it.
- **The dry run** (`results/s1k_dry.json`, pinned binaries) is not a timing: small shapes, 2 tokens, 1 rep, CPU
  load 8-28%. On the experts, in G weights/s:

  | | TQ2_0 | IQ2_BN | IQ2_BN_R4 | TQ1_0 | IQ1_BN | TL2 | I2_S | Q4_K |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | experts | 60 | 77 | 81 | 73 | 65 | 50 (2-D) | 24 (2-D) | 56 |

  - An earlier dry run of the same kernels gave TQ2_0 96 and IQ2_BN 116: these runs differ by up to 40%.
  - Q8_0 read at 35.8 GB/s.
  - Every checked cell had rel_err ≤ 0.011.
  - These numbers were seen before the predictions below.

## Design

**Bench (`s1k_bench.c`).** One C program, compiled against each library's own ggml (`s1k_build.sh`), so that each
format runs its library's kernel.

| Build | Commit | Formats |
| --- | --- | --- |
| llama.cpp, the build of stage 1 (a)/(b) | `5b335f4` | TQ2_0, TQ1_0; Q4_K, Q2_K, Q8_0 as controls |
| ik_llama.cpp, stage 0's reference | `d4c1f75` | IQ2_BN, IQ2_BN_R4, IQ1_BN; Q4_K |
| bitnet.cpp, gcc | `0b341e5` (llama.cpp `390c307`) | I2_S; TL2 with `BITNET_X86_TL2` and the glue |

**Cells.** One token at a time, 6 threads:
- **experts:** Ling-flash's routed experts. Per layer, `mul_mat_id` of k = 4 experts' gate and up (4096 → 1024) and
  down (1024 → 4096), drawn at random per token from 16 experts stored in 3-D tensors; 16 layers with distinct weights.
  That is 805 M weights per token from a 3.2 G-weight footprint, far above the 32 MB L3.
- **experts2d:** the same matrices as 2-D tensors and one `mul_mat` each, for the kernels that only take 2-D weights
  (I2_S, TL2), and for TQ2_0 and IQ2_BN as a comparison.
- **attn:** one 4096 × 4096 product per layer over 64 layers (1.07 G weights).
- **resident:** one layer with the same 4 experts every token (13 MB at 2 bits, inside the L3): the compute ceiling.

**Timing.**
- Graph compute only, with expert draw and graph build excluded. Each launch: 1 warm-up rep, then 9 reps of 128 tokens
  (64 for attn); its number is the median.
- 3 rounds over all 26 cells, interleaved. A key's rate is the median of its rounds, reported with min..max.
- Weights are drawn ternary (P(0) = 0.4); speed does not depend on the values.

**Idle machine (`s1k_run.py`).** Before each launch, total CPU load is sampled 3 times. Above 10% the launch is VOID
and is retried once at the end. Binary hashes are pinned. The machine is idle with the user's go-ahead (one hour).

## Controls before any label (`s1k_verdict.py`)

| Control | Rule | Dry run |
| --- | --- | --- |
| C1 the wall | Q8_0 experts, streaming, reads at 29-45 GB/s (A1: 35-41) | 35.8 |
| C2 streaming | TQ2_0 runs ≥ 1.15× faster resident than streaming | 1.64 |
| C3 correctness | a format is judged only if each of its rounds has rel_err < 0.02 against a double dot product on the dequantized weights, and > 0.5 against the wrong rows | all ≤ 0.011 and ≥ 1.2; IQ2_BN_R4 cannot be read row by row: reported, not judged |
| verdict selftest | 7 planted reports | PASS |

A failed C1 or C2 makes the run VOID.

## Predictions (registered after the dry run, before the timed run)

| | Predicted |
| --- | --- |
| outcome | BETWEEN, more likely than PASS; FAIL very unlikely |
| best ternary on the experts | a 2-bit format (IQ2_BN or TQ2_0), 105-128 G weights/s, at 26-33 GB/s |
| 1.6-bit formats (TQ1_0, IQ1_BN, TL2) | below the best 2-bit one: compute-bound |
| I2_S | the slowest ternary format, under 73 |
| attention (4096 × 4096) | faster than the experts for the same format (fewer barriers per weight) |

## Decision rule (`s1k_verdict.py`, separate from the runner)

- A format's expert rate is the best of its streaming experts and experts2d cells. R is the best checked ternary format.
- **PASS** R ≥ 128; **FAIL** R < 73; **BETWEEN** otherwise.
- Reported, not judged: the attention shape with the same lines, GB/s, resident rates.

## What a result changes (rule 5)

- **PASS:** stage 2 uses that kernel (llama.cpp, ik_llama.cpp and bitnet.cpp are MIT), and no kernel of ours is
  written. If the format carries one scale per row or per tensor (IQ2_BN, I2_S, TL2), stage 2 trains with that scale
  instead of (b)'s scale per 256.
- **BETWEEN:**
  - The delivery puts the measured R into the desk model (table above), with attention at its measured rate.
  - The gap to 122 is what a kernel of ours, or a 1.6-bit format at the wall, would have to buy. The delivery states
    whether the resident rates leave room for it.
- **FAIL:** the per-weight route is closed on this box (stage plan).

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| T-MAC measured | not measured | bitnet.cpp's Table 7 has it below TQ2_0 and TL2_0 at every size on x86 (100B: 0.73 against 1.61 and 1.69); it needs TVM to build |
| bitnet.cpp as shipped | TL2 through a batch-1 glue; codegen for our shapes; gcc | the shipped TL2 path does not link on x86 |
| (b)'s format, one scale per 256 weights | carried exactly by TQ2_0/TQ1_0 only | IQ2_BN keeps one scale per row, I2_S and TL2 one per tensor; their rates are upper bounds for (b)'s format |
| our own LUT kernel (`--lutblk`, E13: 67.5 G weights/s at 512 MB) | not re-measured | it lives in another engine's container, and its record is under the FAIL line; a kernel of ours is the step after this one |
| a full decode step (router, norms, attention core) | only the weight products | the kernel rate is the stage plan's question; the end-to-end is stage 3 |
| prefill | not measured | the thesis is decode tok/s |
| an independent reviewer | not needed by rule 6 | no GPU, no new primary metric (G weights/s is stage 0's) |

## Cost

CPU only. Builds and dry run: done. The timed run takes about 1 h on an idle machine, which needs the user's go-ahead.
