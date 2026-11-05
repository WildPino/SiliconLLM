# Stage 2, part 1, test 5: c50_engine, ik_llama.cpp and llama.cpp in one session. PROTOCOL

2026-11-03. A timing: it runs on an idle machine with the user's go.

## Question

On the same Ling-mini Q4_K_M file at context 2048, how fast does each engine decode in the same session? The ~1.45× of
the engine over ik_llama.cpp comes from two sessions ten days apart:
- S0, 19 October: c50_engine 35.94 tok/s (`../CHATBOT50_ROADMAP_20261016/C50_S0_V0_RESULT_20261019.md`);
- the wall map, 29 October: ik_llama.cpp 24.71, llama.cpp 17.49 (`../WALL_MAP_20261029/WALL_MAP_RESULT_20261029.md`).

## Thesis line (rule 4)

The stage keeps c50_engine.c as the engine and takes the ternary kernel from ik_llama.cpp. Whether the engine's lead is
real decides how much of the desk ceiling's 24.7 tok/s comes from the engine and how much from the kernel. This test
measures Ling-mini only, and its file is Q4_K_M, not ternary. It moves no quality number.

## Design (`benchmarks/stage2/s2e_engines.py`)

Each part is reused as it was run, so each number is comparable with its own session's.

| Arm | Binary | Cell | Measure |
| --- | --- | --- | --- |
| engine | S0's `c50_engine_s0.exe` (sha pinned, source blob of `78356877`) | A1b's arm B0: `--mem map`, normal priority | decode after a fill of 2,048 document tokens, 128 steps, median tok/s |
| ik | the wall map's ik_llama.cpp `d4c1f75` | fa0, fa1, fa0_moe, fa1_moe | `llama-bench -t 6 -p 0 -n 128 -r 5 -gp 2048,128`, mean tok/s |
| llama | the wall map's llama.cpp `5b335f4` | fa_off, fa_on | `llama-bench ... -d 0,2048`, the depth-2048 row |

- **File:** `ling-mini-2.0-a810f64-q4_k_m.gguf`, sha256 `44de7315…`. It is S0's T0 and the wall map's mini.
- **Order:**
  1. the pins;
  2. the wall map's perplexity check: every llama/ik cell within 6% of llama.cpp fa_off, and the planted 4-expert model
     beyond it;
  3. a 45 s pre-flight (foreign occupancy < 4.39%);
  4. 2 planted pairs;
  5. 5 rounds, each visiting the 7 arms once, rotated.
- **Processes:** a process at or above the occupancy bar is VOID and replaced, at most 5 per arm. Each process's
  CONFIG or JSON must confirm its cell, or the run stops.
- **Fill tokens:** the engine fills with document tokens; llama-bench fills with its own tokens. The routed experts
  read the same bytes per token whatever the tokens are, so the decode rates compare. The fill's own speed is not
  compared.

## Dry run (before registration, `benchmarks/stage2/results/s2e_dry.json`)

- **The check reproduces the wall map's:** every cell within 3.4% of llama.cpp fa_off (3.6899), and the planted
  4-expert model at +14.4%.
- **Plant:** engine with the head twice 15.3 tok/s against 21.1.
- **Occupancy:** every process VOID at 16-20% foreign occupancy, as it should be: test 2's training was running.
- **Duration:** about 2 min per process.

## Control (`s2e_verdict.py`, selftest PASS, 5 cases)

**C1 planted:** S0's binary that reads the head twice, paired with the engine.
- The median of the pairs' ratios must be ≤ 0.90.
- S0 measured the head at 7.07 ms of 27.8 ms per token, so reading it twice predicts 0.80.

A failed C1, or a run that is not COMPLETE, gives VOID.

## Decision rule (registered)

The best cell of ik_llama.cpp (and of llama.cpp) is the one with the highest mean in this run. That choice favours them,
not the engine. The ratio engine / ik's best cell gets a CI95 from 10,000 draws over the processes.
- **CONFIRMS ~1.45×:** the ratio is in [1.38, 1.52], 1.45 ± 0.07, about A1b's 5% stability bar;
- **SMALLER:** the CI95's upper end is under 1.38;
- **LARGER:** the CI95's lower end is over 1.52;
- **UNRESOLVED:** otherwise.

Engine / llama.cpp's best cell is reported, not judged.

## Predictions

| | Predicted |
| --- | --- |
| engine | 35-38 tok/s |
| ik best cell | 24-26 tok/s, fa0_moe |
| llama best cell | 17-19 tok/s |
| label | CONFIRMS 70%, SMALLER 20%, LARGER 10% |
| C1 plant ratio | 0.78-0.84 |

## What a result changes (rule 5)

- **CONFIRMS or LARGER:** the stage 3 recommendation counts the engine's lead as measured. Ternary is then the kernel's
  question alone.
- **SMALLER:** the delivery reports the engine's lead at its measured size, and the desk ceiling's engine share is
  recomputed with it. Whether c50_engine.c remains the base is for the delivery's recommendation.

## Parts specific to Ling

c50_engine.c runs only Ling-mini's architecture (bailingmoe2: shapes, the sigmoid router with group selection, the
q/k norms). Porting it to Ling-flash's shapes is outside the stage. The two llama.cpp engines are general.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| the ternary file the stage targets | Q4_K_M | the stage's test 5 is on Q4_K_M; the engine has no ternary path yet |
| Ling-flash | Ling-mini | the engine does not run flash's shapes (outside the stage) |
| the same fill for every engine | each engine's own | llama-bench has no document fill; decode reads the same bytes |

## Cost and session

About 1.5 h on an idle machine:
- the check, about 7 min;
- 37 processes of about 2 min each.

In the same idle session, after this run: test 4's training-rate microbenchmark (`s2f_rate.py`, about 2 min of GPU,
dry run in `s2f_rate_dry.json`).

## Amendment 1 (2026-11-05, before any run of the session)

Three cells close the desk budget's uncertainty, on one build of `c50_engine.c` with default-off options
(`bin/c50_engine_s2e.exe`, `s2e_cells.py`, clang 19.1.7, because gcc cannot compile the file's target attributes). They run
after the seven arms, in the same session: 6 arms, 5 rounds, rotated; a `base` arm of the same build is every cell's reference,
so the compiler does not enter any comparison.

| Cell | Options | Closes | Predicted |
| --- | --- | --- | --- |
| Barriers | `--barrier-extra 1`, `3`: empty `run()` after each job | the desk's fixed 2.70 ms per 20 layers | 2-6 µs per job, 0.3-0.8 ms of 27.8 ms |
| KV in q8_0 | `--kv q8_0` | S0 read the KV at 16.1 GB/s while the desk charges 32 | KV organ 5.36 ms → 3.5-5 ms |
| 4 and 3 experts | `--experts-used 4`, `3`: time only, weights not renormalized | how the routed read scales with k | −4 to −4.5 ms; −5.2 to −6.6 ms |

- **Reported, no label.** VOID unless bar3 > bar1 > base > k4 > k3 in ms per step. Occupancy, replacements and CONFIG as above.
- **Check before registration:** on 499 positions the build with options off, and with one extra barrier, gives logits equal to
  S0's byte for byte. `--kv q8_0` differs, as it must: top-1 0.970, KL 0.0195 against f16. It is a timing cell only.
- **Also in the session:** `s2t_gather_time` (`--threads 6 --reps 200`), the selection and the gather of test 2b. Contended, they
  read 1.2-1.9 ms and 0.34 ms against the desk's 0.2 and 0.11.
- **Takes from the design:** q8_0 here quantizes on write in the engine's own rows, not the kernel the ternary engine would use;
  k = 4 and 3 change every position's read, as decode-only k would.
- **Parts specific to Ling:** the job count and the 256 experts; the method needs only a thread pool and a top-k MoE.
- **Session:** about 1 h more, 2.5-3 h in all.

## Amendment 2 (2026-11-05, before any run of the session)

Seven cells for what the desk budget still assumes at flash's shapes (`s2k_cells.py`; `s2k_bench.c` on ik_llama.cpp's kernels; `s2k_engine.c`,
`c50_engine.c` itself built with `C50_FLASH_SHAPES`, no model read, default build unchanged). They run last in the session: 18 arms, 3 rounds,
rotated. ms = 32 layers, streaming. Predictions are mine; the dry run is contended and only proves the arms run.

| Cell | Arms | Closes | Predicted |
| --- | --- | --- | --- |
| 1. K and V alone | Q4_K, IQ2_BN on 512 × 4096 | what exact K and V cost | 1.1-2.0 / 0.5-0.9 ms |
| 2. QKV in two calls | 5120 × 4096 fused; 4096 + 1024 in IQ2_BN; 1024 in Q4_K | the price of a mixed attention | 4.4-5.2 / 4.8-5.6 / 6.0-8.0 ms |
| 3. Formats above 2 bits | IQ2_KS, IQ3_KS, Q3_K, IQ2_KL on 4096 × 4096 | what an organ pays one step up | 70-110, 55-80, 55-80, 45-70 G weights/s |
| 4. Fixed term, shared expert | every weightless step, at 32 layers; shared's three matrices | the desk's 4.32 ms; 5.17 ms | 3.5-6 ms; 4.9-5.5 ms |
| 5. KV, 32 heads on 4, context 2048 | f16, q8_0; f16 at 512 as control | the desk's 2.2 ms | f16 10-17 ms, q8_0 7-15 ms |
| 6. Router | 256 × 4096, Q8_0, the budget's int8 | the desk's 1.05 ms | 1.1-2.0 ms |
| 7. All four codes | IQ2_BN written with four codes, against three, on 4096² and the shared expert | whether the code costs time | ratio 0.97-1.03 |

- **Already measured, not redone:** Q4_K on 4096² (61.03 G weights/s). IQ2_BN on 4096² (155.63) and the 2-D shared expert (77.92) enter only as
  cell 7's same-session reference, and repeat stage 1's numbers.
- **The risk is cell 5.** The desk charges bytes at the wall. S0 read mini's KV at 16.1 GB/s, under the wall, and flash does 3.2 times mini's
  attention arithmetic on 1.6 times its bytes. Above 3.3 ms in q8_0 the room of 1.12 ms is gone, and the ceiling needs a faster attention
  kernel. That row goes in the delivery either way. The 2.2 ms stays a desk number until this cell reads.
- **Reported, no label.** VOID if a kernel's check fails (rel_err ≥ 0.05, wrong rows ≤ 0.5), if the KV at 512 does not take under 0.6 of the KV at
  2048, or if the fixed term did not run 7 jobs per layer. q8_0 slower than f16 is a result, not a VOID. Occupancy, replacements and CONFIG as above.
- **Dry (`s2k_cells_dry.json`, contended):** 18 arms run, rel_err ≤ 0.0098, wrong rows ≥ 1.33, 224 jobs per step.
- **Takes from the design:** weights in the kernels' cells are random, not Ling's, since the rate does not depend on the values (E45). The fixed
  term runs empty jobs, so it does not see barriers waiting on uneven work. The KV cell reads a filled cache without writing it. The 2-D chain has no
  routed experts beside it, while the engine runs the shared expert in the routed job.
- **Parts specific to Ling:** every shape, 32 heads on 4 and k = 4. The method needs only a kernel library.
- **Session:** about 10 min more, about 3 h in all.

**Correction to the budget's 27.8 ms (S0, `c50_s0_run_1.json`, context 2048).** The organs sum to 28.80 ms; the rate rows give 27.82 ms. They are
two quantities. The organs are means of steps in the 5 processes with laps (mean step 28.82 ms). The rate is the median of the 10 processes
without laps (27.83). Of the 0.98 ms, about 0.70 is mean against median (the laps' processes have median 28.12) and about 0.30 is the laps' own cost.
Organ shares stand; organ milliseconds read about 3.4% high against the rate. The desk takes neither, it uses bytes and rates. Above, 7.07 of 27.8
mixes the two: on one scale, 7.07 of 28.80 predicts 0.80 as well, so no line moves.
## Amendment 3 (2026-11-05, a second idle session, not launched)

Before any attention kernel of ours: what ik_llama.cpp's attention does at flash's shape. And the fourth code's correction, in time. `s2l_cells.py`, `s2l_attn.c`, `s2k_engine.c --cell corr`; 5 arms, 3 rounds, reported, no label.

| Arm | What | Predicted (ms per token, 32 layers) |
| --- | --- | --- |
| fa_f16, fa_q8 | `ggml_flash_attn_ext`, 32 heads on 4, context 2048, f16 and q8_0 | 5-14, 5-14 |
| mm_f16 | ik without flash attention, f16 | 7-16 |
| fa_f16_512 | the control, context 512 | under 0.6 of fa_f16 |
| corr | −0.5·s·sum(x) on every expert row, one thread: an upper bound, the engine folds it into its jobs | 0.2-0.6 |

- **Check:** seven heads of the first layer against a double attention on the dequantized cache, and against the wrong KV head. Smoke: rel_err ≤ 0.006, wrong ≥ 1.28. VOID if rel_err ≥ 0.05, wrong ≤ 0.5, or the control fails.
- **What it changes:** if ik reads under 5.15 ms, its kernel is the route, with its licence, and no kernel of ours is needed. If between 5.15 and the engine's 8.96, it says how far a known kernel gets. If at or above the engine's, the kernel is ours to write.
- **Takes from the design:** no mask, one token, random cache: a real graph adds the mask and the KV write. The heads are ik's graph op, not its whole server path.
- **Parts specific to Ling:** the shape. The method needs a ggml build.
- **Session:** about 10 min on an idle machine.