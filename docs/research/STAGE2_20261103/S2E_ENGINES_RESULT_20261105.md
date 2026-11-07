# Stage 2, tests 5 and 4: the idle session. RESULT

2026-11-05, 12:02-13:48, idle machine. Protocol `S2E_ENGINES_PROTOCOL_20261103.md`, amendments 1 and 2. Records in `benchmarks/stage2/results/`:
`s2e_run.json`, `s2e_verdict.json`, `s2e_cells.json`, `s2t_gather_time.json`, `s2k_cells.json`, `s2k_report.json`, `s2f_rate.json`, `s2f_feasibility.json`,
`s2_ceiling_measured.json`. A first launch stopped before measuring anything: `git` was not on the launcher's PATH.

## Outcome

**Engines: LARGER.** c50_engine is 1.543× ik_llama.cpp's best cell, CI95 [1.534, 1.550], and 2.15× llama.cpp's. On the same Ling-mini file, context 2048.

**The desk's KV row was wrong by four to five times.** At flash's shape the engine's attention takes 8.96 ms in f16 and 11.67 ms in q8_0; the desk charged 2.23.
**Cell 1 goes from 25.40 to 21.27 tok/s in q8_0, or 22.57 in f16: under the 24.7 line.** This is a sum of cells, not a run on flash. Rows not measured stay desk.

## Engines (5 rounds, tok/s, min..max)

| Arm | tok/s | Predicted |
| --- | --- | --- |
| c50_engine, S0's binary | 38.77 [38.38..38.96] | 35-38: above |
| ik, best cell fa0_moe | 25.13 [25.05..25.26] | 24-26, fa0_moe: hit |
| ik fa0 / fa1 / fa1_moe | 24.26 / 23.41 / 24.23 | |
| llama, best cell fa_off | 18.00 [17.87..18.12] | 17-19: hit |
| llama fa_on | 16.70 | |

C1 planted ratio 0.803 (0.78-0.84 predicted): pass. Label predicted CONFIRMS 70%, LARGER 10%. The same binary read 35.94 on 19 October; the 7.9% gap
between sessions is not explained, which is why the ratio comes from one session.

## Amendment 1: VOID by its own rule

bar3 > bar1 did not hold: 31.35 against 32.63 ms. Processes fall in two groups about 1.7 ms apart (base 30.92..32.71, k4 25.40..27.07), which is more than the
barrier's cost. The barrier cell says nothing. Kept as VOID, not rerun: the fixed term below answers the same question.
Read only as an indication, on this VOID run: `--kv q8_0` adds 1.58 ms, the opposite of the −0.4 to −1.9 predicted, and agrees with cell 5. k = 4 saves 5.64 ms and
k = 3 saves 6.97, above the predicted −4 to −4.5 and −5.2 to −6.6.

## Amendment 2: every control holds (ms at 32 layers, 3 rounds)

| Cell | Measured | Predicted | Desk |
| --- | --- | --- | --- |
| 1. 512 × 4096, Q4_K / IQ2_BN | 0.93 / 0.43 | 1.1-2.0 / 0.5-0.9 | |
| 2. QKV fused / two calls / 1024 in Q4_K | 4.21 / 4.21 / 7.90 | 4.4-5.2 / 4.8-5.6 / 6.0-8.0 | |
| 3. G weights/s: IQ2_KS, IQ3_KS, Q3_K, IQ2_KL | 143.8, 96.8, 93.3, 77.8 | 70-110, 55-80, 55-80, 45-70 | |
| 4. Fixed term; shared expert | 2.41; 5.25 | 3.5-6; 4.9-5.5 | 4.32; 5.01 |
| 5. KV f16 / q8_0 (f16 at 512: 2.38) | **8.96 / 11.67** | 10-17 / 7-15 | **2.23** |
| 6. Router, Q8_0 | 0.88 | 1.1-2.0 | 1.08 |
| 7. Four codes over three | 0.999, 1.002 | 0.97-1.03 | |

Selection and gather at K = 1,024, idle, p50: 0.195 and 0.177 ms, against the desk's 0.2 and 0.11. Contended they had read 1.2-1.9 and 0.34.

- **Splitting QKV costs nothing,** and small matrices keep IQ2_BN's rate (156 G weights/s). My small-matrix penalty was wrong: most kernel predictions were too pessimistic.
- **A Q4_K slice inside QKV costs 2.6 ms more than its parts** (3.36 + 1.87 against 7.90). Not explained. Reported only: the two input formats are one candidate, not tested.
- **The four codes cost no time.** Bal's −0.5·s·sum(x) correction is not in this cell.
- **IQ2_KS runs at 3.73 ms per 4096², against IQ2_BN's 3.36.** A better 2-bit format is available for 11% more time, if an organ needs it.
- **q8_0 is slower than f16:** 6.1 GB/s against 15.0. The KV is arithmetic-bound in this code, not byte-bound: the desk's assumption, bytes at the wall, does not hold.
- **The fixed term ran empty jobs,** so 2.41 ms leaves out waiting on uneven work: it is a floor.

## The ceiling (`s2_ceiling_measured.py`; cell 1, K = 1,024, k = 4, context 2048)

| Row | Desk ms | Now ms |
| --- | ---: | ---: |
| KV, q8_0 | 2.23 | **11.67** measured |
| Fixed term | 4.32 | 2.41 measured |
| Shared expert | 5.01 | 5.25 measured |
| Router | 1.08 | 0.88 measured |
| Selection, gather | 0.31 | 0.37 measured |
| Routed 13.77, attention 7.76, preview 4.14, dense 0.73, small 0.04 | 26.43 | 26.43 desk |
| **Total** | **39.37, 25.40 tok/s** | **47.01, 21.27 tok/s** |

| KV | tok/s | With test 3b's x15, +1.03 ms |
| --- | ---: | ---: |
| q8_0, as measured | 21.27 | 20.81 |
| f16, as measured | 22.57 | 22.06 |

**The line needs a KV of at most 5.15 ms,** 4.12 with x15. That is 1.7-2.2 times faster than the engine's f16, or 2.3-2.8 times faster than its q8_0. It is not against arithmetic:
537 M multiply-adds per token, on 6 cores at 16 per cycle and 3.6 GHz, take 1.6 ms at peak. Flash's 8 query heads per KV head are what mini does not stress: it
has 4. Any kernel written for this is outside this stage.

## Test 4: training rate at flash's shapes

RTX 3060, valid at both shapes, 9.05 GB at flash's. One block, per token and layer: 0.560 ms; frozen 0.063; scales 0.186.
Sequential blockwise training on 1.5 M positions takes 7.7 h; on 16.8 M, 86.4 h. With 4 downstream blocks, 11.1 and 124 h. Full QAT needs 412 GB of state and does not fit.
Those hours sit inside the budget, and the rates are measured; flash's quality under them is not.

## Parts specific to Ling

The shapes, 32 heads on 4, k = 4, and the 1.54× on this file. That this attention code is arithmetic-bound at this ratio of query to KV heads should hold on other models with the same ratio; it was not measured on any.

## What this changes

- **The stage's delivery counts the engine's lead as measured,** as registered for LARGER.
- **The line is not reached by quality work alone.** The next limit is the attention kernel. The stage's delivery will carry it with one recommendation.

## Amendment 3: ik's attention at flash's shape (2026-11-06, ms at 32 layers, 3 rounds)

Controls hold: rel_err at most 0.0063, the wrong KV head 1.3 or more, context 512 at 0.44 of 2048, fa_f16 under the NULL-mask time. Records: `results/s2l_cells.json`, `s2l_report.json`; the first run, on ggml's generic path, is `s2l_cells_nomask.json` (33.56, 47.06 ms).

| Arm | ms, min..max | Engine | Predicted |
| --- | --- | ---: | --- |
| ik flash, q8_0 | **5.95, 5.94..6.24** | 11.67 | 5-14, hit |
| ik flash, f16 | 7.24, 7.22..7.28 | 8.96 | 5-14, hit |
| ik without flash, f16 | 9.00, 8.98..9.06 | 8.96 | 7-16, hit |
| ik flash, f16, context 512 | 3.16, 3.15..3.22 | | control |
| −0.5·s·sum(x), one thread | 0.25, 0.24..0.26 | | 0.2-0.6, hit |

- **A known kernel gets within 0.80 ms of the line.** By the registered reading, 5.95 falls between 5.15 and 8.96: it says how far a known kernel gets. It is 2.0 times faster than the engine's q8_0 and 1.5 times faster than the engine's f16.
- **The ceiling with ik's q8_0 row in place of the engine's:** 47.01 − 11.67 + 5.95 = 41.29 ms, **24.22 tok/s**, against 24.7 (40.49 ms). In f16, 42.58 ms, 23.49 tok/s. This is a sum of cells: ik's kernel ran in ggml's thread pool, not the engine's, and nothing was ported.
- **In ik q8_0 beats f16, the reverse of the engine.** At 11 GB/s neither reads at memory speed; why q8_0 is faster is not explained.
- **The correction costs 0.25 ms on one thread,** an upper bound for a term folded into the jobs.
- **ik's flash path needs a mask, and crashed here when built with plain GCC flags.** It ran once the assembler option for unaligned moves was on. The comparison's best ik cell, 25.13 tok/s, had flash off; its flash cells read 23.41 and 24.23 at mini's shape, and whether they took this fast path was not checked.
