# Per-organ byte map of Ling-mini-2.0 and Ling-flash-2.0, and their clean reference speeds

**Status:** PROTOCOL, 29 October 2026. A desk study with one measurement. No GPU.
**Code:** `benchmarks/wallmap/` (`w0_byte_map.py`, `w0_desk.py`, `w0_family_map.py`, `w0_ref_speed.py`,
`w0_ref_read.py`).

## Question

For Ling-mini-2.0 (16.3B parameters) and Ling-flash-2.0 (102.9B): where does every byte read per decoded token sit,
organ by organ? How many bytes would each organ read in the target form? Which organ is the wall?

The target form is:
- a small always-on core;
- experts in RAM, read in contiguous blocks;
- low-bit LUT kernels on CPUs without VNNI;
- a recurrent state in place of a growing KV cache.

## What it produces

| # | Item | Source | Status |
| --- | --- | --- | --- |
| 1 | Bytes per token by organ, today (Q4_K_M as stored, F16 KV), at contexts 2048, 8192 and 32768 | `w0_byte_map.py` on the GGUF headers | SOURCE-DERIVED, run |
| 2 | The same in the target form, by scenario | `w0_desk.py` | DESK-MODEL, run |
| 2b | Active parameters by organ for other MoE families (Ling-1T, gpt-oss 20b/120b, Qwen3 30B/235B, Qwen3-Next 80B), from their config.json, with the all-ternary desk speed of each | `w0_family_map.py` | SOURCE-DERIVED + DESK-MODEL, run |
| 3 | Per organ: what the literature reports about converting it with training, and its known failures | papers, read in their tables | desk |
| 4 | Per organ: the cheapest test that could answer "impossible", with its cost | desk | desk |
| 5 | The desk ceiling: Ling-flash tok/s if every organ succeeds, and if only the likely ones do | `w0_desk.py` | DESK-MODEL |
| 6 | **Reference speeds:** llama.cpp and ik_llama.cpp on the two original Q4_K_M files, idle box | `w0_ref_speed.py` | **the only measurement** |
| 7 | Proposed thresholds | the result | proposal |

**Control on item 1.** The script reproduces the Ling-mini figure of ROADMAP §3 from the same file: 867.2 MB/token
at 2048, against 867.3 recorded on 17 October.

**Control on item 2b.** The config-derived counts for Ling-mini and Ling-flash equal the GGUF tensor counts of item 1,
organ by organ, within 0.1%, or the script stops. It passes. The gpt-oss and Qwen3-Next rows are approximate (biases,
convolutions and norms left out) and are flagged.

## The measurement (item 6)

**Engines.** Both are built from source in Release with GGML_NATIVE=ON (Zen 2), gcc 15.2 (WinLibs) and ninja:
- llama.cpp `5b335f4` (the pinned tree of A2b), `build-native`, with `_WIN32_WINNT=0x0A00` and the server off, as
  its `build-generic` (the vendored HTTP library does not compile otherwise);
- ik_llama.cpp `d4c1f75`, `build-native-ua`, with `-Wa,-muse-unaligned-vector-move`. Without it its
  flash-attention kernel crashes (access violation) under MinGW on both files: on Ling-mini at prompt batches of 32
  tokens or more, on Ling-flash even at 16. With it, every cell runs on both files (smoke, 29 October).

**Files.** The same two GGUFs for both engines:
- Ling-mini-2.0 Q4_K_M, sha256 `44de7315…`;
- Ling-flash-2.0 Q4_K_M without imatrix, sha256 `5348c73f…`, read from the SSD copy on C:.

**Command.** `llama-bench -m FILE -t 6 -p 0 -n 128 -r 5 -o json` at depths 0 and 2048:
- llama.cpp writes the depth as `-d 0,2048`;
- ik_llama.cpp has no `-d`; it writes it as `-gp 2048,128`.

Both time only the 128 generated tokens, after the 2048-token prompt is in the cache.

Over that, a grid fixed here:
- llama.cpp: `-fa off`, `-fa on`;
- ik_llama.cpp: `-fa 0` and `-fa 1`, each with its CPU MoE options off (`-fmoe 0 -rtr 0`) and on (`-fmoe 1 -rtr 1`).
  Fused MoE is on by default in ik_llama.cpp, so the plain cells turn it off explicitly. The JSON must confirm both
  flags. `-rtr` turns mmap off in ik_llama.cpp's loader even though the JSON still prints `use_mmap: true`; the
  repack flag is what the runner checks.

**Correctness check, before any timing.** A fast cell that decodes wrongly must not become the reference. For each file
and cell, `llama-perplexity` with the cell's flags runs 4 chunks of 256 tokens at one token per ubatch (the decode
path) on a fixed text (the imatrix calibration text, sha256 pinned in the record). Every cell must land within 6% of
llama.cpp `fa_off` on the same file.
- The 6% bar sits above the spread of valid configurations measured on Ling-mini on 29 October: 3.564 to 3.690, so
  3.4% from the reference. llama.cpp alone moves 3% with `-fa` and the ubatch.
- A planted wrong model must fire the check: Ling-mini with 4 experts per token instead of 8 measured +14.4%.
- If a cell misses the bar or the planted model does not fire, the run stops.

**Schedule.**
- Before every process the runner reads the model file end to end (page-in). `-rtr` keeps a private 62 GB copy of
  Ling-flash, which evicts the page cache, so the next process would otherwise start cold. The page-in time is
  recorded.
- Each grid cell runs as 3 separate processes, in round-robin order across the cells.
- Each process prints its CONFIG (engine commit, flags, file sha256). The runner refuses a cell whose CONFIG does not
  confirm it.

**Contention.** E44's split of the process's own and foreign CPU occupancy, as in A2b and S0:
- a 45 s pre-flight whose mean foreign occupancy must stay under 4.39%;
- a process at or above that bar is VOID and is replaced, at most 5 times per cell.

**Reading** (`w0_ref_read.py`, a separate script). For each engine, file and depth:
- per process, the median of repetitions 2-5. The first is dropped: llama-bench's warm-up is one token, so the
  first repetition pages in experts. The dry run shows it: ik `fa0` 21.69 against ~24.0 for the rest;
- the median of the 3 process figures, with their min..max;
- the reference is the best cell of each engine, and every cell is reported.

**Cost.** About 4-5 h of an idle machine: the check ~25 min; then ~100 s per Ling-mini process and ~8-10 min per
Ling-flash process, plus page-in. **The run needs the user's go**, and is pushed before it starts.

## Trivial control (rule 3)

The desk model's own prediction for the unconverted Q4_K_M file sits next to each measurement. On Ling-mini:
- the desk model gives 33.6 tok/s at 2048;
- llama.cpp measured 14.39 (A2b);
- the C engine measured 35.9 on the same file (S0).

**What this control reads.** It separates how much of any speed-up over llama.cpp is engine efficiency on unchanged
weights, and how much is conversion. A threshold against llama.cpp counts only the second part, so it is set against
the faster of the two engines.

## Desk model (item 2)

`t = Σ over organs of max(bytes / 32 GB/s, weights / R(form)) + 2.70 ms × layers / 20`. The two constants are the
A1 measurements on Ling-mini shapes.

**Kernel rate R** (added after the independent review). The ternary forms are compute-bound on this box. The operative
packed ternary kernel processes 46.16 G weights/s at the goal shape T10 (E28's session, packed arm). At 0.25 B/weight
that is 11.5 GB/s, a third of the wall. `--lutblk` reached 61.64 G weights/s, but E32 revoked its quality licence.
- The model prints three figures: bytes only (R infinite), packed, and lutblk.
- The stored Q4_K/Q6_K/Q8_0/F32 forms and int8 keep R infinite: A1 timed them streaming at 35-41 GB/s.
- trit5 and sub1 have no CPU kernel, so they get the ternary rate. QMoE's 0.8 bit is its ternary model,
  entropy-coded.
- Ling-flash's expert matrices (1024 × 4096) are not T10's shape, and the rate there is not measured.

**Recurrent state of the ring scenarios.** Each linear layer keeps heads × 128² fp32 values (from the Ring-linear
configs), read and written once per token. In flash that is 117 MB per token, more than the KV it removes at 2048.
Extra projections of the linear layers, if any, are not counted.

**Scenarios.**

| Scenario | What changes |
| --- | --- |
| `today_q4km` | nothing |
| `ptq_safe` | BF16 router (lossless on Ling-mini), q8_0 KV |
| `ring_linear_ptq` | the KV cache only in 1 of 5 (mini) or 1 of 8 (flash) layers, plus the recurrent state, as in the producer's Ring-linear models (600B-1T tokens of continued pretraining) |
| `all_ternary_nibble` / `trit5` | attention, shared, dense, experts and head ternary at 0.25 / 0.2 B/weight; int8 router; ring KV and state |
| `likely_experts_2bit` | experts at 0.25 B/weight priced with the ternary kernel (QMoE's 2-bit has 4 levels; no CPU 2-bit LUT kernel exists, and Q2_K is 0.33 B/weight), int8 router, q8_0 KV in every layer, the rest as stored |
| `experts_sub1_rest_nibble` | experts at 0.1 B/weight (QMoE's 0.8 bit), the rest at 0.25; ring KV and state |

**Not modelled (scoped out of every per-weight claim):** structural levers that change how many weights a token reads.
These are fewer active experts, vocabulary trimming, low-rank attention (H4), layer dropping, and speculative
decoding with Ling-mini as the draft. The result lists them with their byte effect.

**Not measured:**
- the ternary kernel's rate at Ling-flash's shapes;
- the fixed term at Ling-flash's width.

The model prints a second figure with the fixed term doubled.

## What it takes from the design (rule 1)

| The design asks | Here | Why |
| --- | --- | --- |
| Reference speeds "configured well" | a fixed grid of 4-8 cells per engine, not a search | a search over flags on the same runs would select on noise |
| Long contexts | depth 0 and 2048 only | flash at depth 8192 costs about 10 min per process; the KV term at any context is in item 1 |
| The C engine on Ling-flash | not measured | it runs only Ling-mini's shapes today |
| KV in q8_0 for the reference | F16, as both engines default | q8_0 KV is a lever of the target form, not of the reference |
| A small core that sits (almost) in cache | not modelled as a cache hit | the flash core in nibble form is ~600 MB against 16 MB of L3 per CCX; the result reports the size it would need |

## Literature (rule 2)

Read in the papers before this protocol, with the numbers taken from their tables:
- LoLCATs (2410.10254);
- Mamba in the Llama (2408.15237);
- MOHAWK (2408.10189);
- Llamba (2502.14458);
- Ring-linear (2510.19338);
- QMoE (2310.16795);
- BitNet b1.58 (2402.17764);
- BitNet b1.58 2B4T (2504.12285);
- BitNet Distillation (2510.13998);
- Turbo Sparse (2406.05955).

Their numbers enter the result with their table.

## Thesis line (rule 4)

This is a map: no outcome moves the thesis directly. It decides which organ is tested first, and whether the
Ling pair can show speed independent of total size at all.

## Independent review (rule 6)

Done on 29 October, before this push. A fresh agent received the thesis, the map, the code and this protocol. Its
objections, and what changed:

| # | Objection | Answer |
| --- | --- | --- |
| 1 | The desk runs ternary at 128 G weights/s; the operative kernel measured 46.16 (E28) | **Accepted.** Kernel rate added; flash all-ternary falls from 19.2 to 7.8 tok/s |
| 2 | The ceiling is not given as a multiple of the reference | Accepted: the result does it after the measurement |
| 3 | Structural levers missing; both headline claims true by construction | Accepted: claims scoped to per-weight forms, levers listed in the result |
| 4 | The in-cache core is not modelled | Accepted: row in "What it takes from the design" |
| 5 | Ring scenarios leave out the recurrent state | Accepted: state added (2 × 4 B per value) |
| 6 | The literature reading overstates; sub1 and the 2-bit form are mispriced | Accepted: reading rewritten in the result; both priced at the ternary rate |
| 7 | `-rtr` turns mmap off and evicts the page cache; `avg_ts` includes the cold first repetition | Accepted: page-in before every process, first repetition dropped |
| 8 | `-ub 16` hides a broken kernel; nothing checks correctness | Accepted: the alignment build replaces `-ub 16`; perplexity check with a planted wrong model |
| 9 | `-gp` against `-d` | Checked: equivalent (both time only the generated tokens) |
| 10 | The item-1 control is not independent; the desk is not a ceiling | Partly: the control stays as it is (it checks the arithmetic, not the rule); §4 of the result is labelled an estimate with a band |
| 11 | MTP layers unchecked | Checked: `num_nextn_predict_layers` is 0 in the Ling-mini and Ring configs and absent from Ling-flash's |
| 12 | The llama.cpp mini prediction is already exceeded on a busy box (16.75) | Kept as registered; see the note under the predictions |
| 13 | Anything citing internal documents | None found |

## Predictions (scored, not gated)

| Prediction | Value |
| --- | --- |
| llama.cpp, Ling-mini, depth 2048 | 13.5-15.5 tok/s (A2b: 14.39 with the generic build) |
| llama.cpp, Ling-flash, depth 2048 | 3.0-4.5 tok/s (0.37-0.55 of the desk 8.2, as Ling-mini ran at 0.43 of its desk) |
| ik_llama.cpp over llama.cpp, best cell against best cell | ≥ 1.3× on both files |
| flash/mini with the faster engine, depth 2048 | 0.20-0.30 |

These were written before the dry runs and stay as written. The dry runs ran on a busy box (foreign occupancy ~10%)
and are not timings, but they were seen before this push:
- Ling-mini at depth 2048: llama.cpp 16.75 (`fa_off`) and ik_llama.cpp 24.0 (`fa1_moe`);
- a one-repetition Ling-flash smoke at depth 2048: llama.cpp 5.3-5.9 and ik_llama.cpp 7.3-7.8.

The predictions are therefore not blind.
