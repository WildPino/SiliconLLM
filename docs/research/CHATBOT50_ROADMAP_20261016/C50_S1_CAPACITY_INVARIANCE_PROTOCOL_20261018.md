# C50-S1 protocol: speed invariance with capacity only

18 October 2026. Cell S1 of the [CHATBOT50 roadmap](ROADMAP.md) §5, Phase 1. Pushed before any S1
process runs, with its runner, report and binaries. Only the expanded artifacts' hashes come later,
in an addendum pushed after their build and before the session.

## Question

- **The question (ROADMAP §5 S1):** when T0's stored capacity grows at fixed active bytes, does the
  engine's time per token stay at T0's, apart from the priced bytes of the router?
- **Why it matters:** the Phase-1 claim is that capacity can grow without the speed paying for it
  beyond a known, priced term. S4 and S5 build on that claim.
- **What S1 does not do:** it reads no quality, because the added experts are byte copies and add
  no knowledge.

## Preconditions

- **The A4 qualification of T0** (A4 + A4-bis). If A4 fails, T0's replacement (Q5_K_M, then Q6_K)
  is expanded instead, by an addendum before the session.
- **S0 done.** S1 does not reuse v0's number: every ratio here uses the m = 1 arm measured in the
  same session.
- **A1b PASS (attempt 2), arm B0:** every process runs with `--mem map --prio normal`.
- **The session runs outside the host's memory reaper:** either the owner starts it from a terminal,
  or the session runs with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`. Two S2 builds were
  stopped by that reaper on 18 October.

## The artifacts

- **The expander,** `c50_s1_expand.py` (committed in `1eb08f9`), grows each MoE layer's pool by byte
  copies.
  - **Layout:** 8 groups of E/8 slots. Slot g·E/8 + 32k + j holds copy k of T0's expert 32g + j.
  - **Router rows:** copy 0 is T0's row. A copy k ≥ 1 is the row times (1 + σz), rounded to BF16,
    with z ~ N(0, 1) seeded per (layer, slot).
  - **Re-read checks:** after writing, every slot is checked against its source.
  - **Parameters fixed now:** **σ = 0.05, seed 20261016.** σ is set well above BF16's relative
    step (2⁻⁸ ≈ 0.004), so that the copies' scores differ.
- **The pool sizes,** computed from T0's header:
  - each extra expert adds 36,208,716 bytes over the 19 MoE layers;
  - the router rows take 2048 × 19 values per expert.

  | Arm | E | Stored GB | m | Router MB/token, F32 / BF16 |
  | --- | ---: | ---: | ---: | ---: |
  | m1 (T0) | 256 | 9.91 | 1.00 | 39.8 / 19.9 |
  | m2 | 536 | 20.05 | 2.02 | 83.4 / 41.7 |
  | m4 | 1,088 | 40.04 | 4.04 | 169.3 / 84.7 |
  | m6 | 1,600 | 58.58 | 5.91 | 249.0 / 124.5 |

- **Correction to ROADMAP §5 ("≈ 7 here").** m6 is the largest pool that stays resident with
  ~10 GB free. 2,048 experts (74.8 GB) does not fit: 71.7 GB of the 80 GB are free at idle
  (measured 18 October). The roadmap's "≈ 7" did not subtract what the OS already uses.
- **Staging:** all four artifacts go on the NVMe C:, 128.6 GB together. F0 leaves C: after S2.
  The owner is asked for ≥ 160 GB free on C: before the session (COMMUNICATION).

## Engine and known positives

- **The engine:** `c50_engine.c` at `1eb08f9`, the S1 tools commit (LF blob sha256 `68c947d9…`). The
  binary `bin/c50_engine_s1.exe` (sha256 `79e2e4f4…`) was rebuilt from it on 18 October with
  `clang -O2 -mavx2 -mfma -mf16c`. It does not equal the earlier `c50_engine_s1dev.exe` byte for byte,
  so it got its own parity check (below).
- **The options,** each default off and printed on the CONFIG line:
  - **E** is read from the header;
  - `--router bf16` converts the F32 router at load time. The load exits 5 if any value is not
    exactly BF16. The dot expands back to F32 in the F32 kernel's order;
  - `--hash-copies` (arm H): the router reads only copy 0 of each group, i.e. T0's 256 rows, and
    selects as T0 does. Each pick then goes to one of its byte-identical copies, by splitmix64 of
    (layer, position, expert).
- **Two known positives, run first in the session:** `--score` with `--dump-logprobs` on the
  parity document used by the A1b, S0 and S1-tools parity checks: `par1.tsv`, the first 200 IDs of
  SEL item 0 (`c50_holdout_ext_v1/sel_ids.tsv`, line 1), sha256 `373b4886…`; the runner rebuilds it
  and checks the hash. The position records and the log-probability rows are compared with `cmp`
  against the A3 binary on T0.
  - **K1:** the BF16 router at m = 1 is byte-identical to T0. If not, every BF16 arm is VOID, and
    S1 runs on F32 only (ROADMAP §5).
  - **K2:** arm H is byte-identical to T0 at m2, m4 and m6, in both formats. If not at some m, arm
    H is VOID at that m.
  - **Already seen, on T0 (E = 256) only:**
    - the S1 tools check of 17 October (§12): five combinations byte-identical to A3;
    - **the rebuilt `c50_engine_s1.exe`, on 18 October, before this push:** default, `--router bf16`,
      `--hash-copies`, and both together. All four are byte-identical to the A3 binary
      (`d14ea9a5…`) on `par1.tsv`, in the position records and in the 199 × 157,184
      log-probability rows.
  - K1 and K2 on the expanded pools are new.

## Arms, schedule and instrument

- **Instrument:** `--bench` at context 2048, 128 timed steps after a 2,048-token fill, one fresh
  process per visit. These are S0's rules:
  - the A4 FIT document 0;
  - the E44 occupancy split, bar 4.39%;
  - a process at or above the bar is VOID and replaced, at most 5 replacements per arm;
  - a failed process or an unconfirmed CONFIG line makes the run VOID.
- **Arms:**

  | Arms | Processes each | Role |
  | --- | ---: | --- |
  | E-F32 and E-BF16 at m1, m2, m4 | **20** | gated |
  | H-F32 and H-BF16 at m2, m4 | 10 | reported (pool effect; router cost per byte) |
  | E and H, both formats, at m6 | 10 | reported |
  | `--organs` E-F32 and E-BF16 at m1 | 6 | prices the router (below) |
  | `--organs` E-F32 and E-BF16 at m2, m4, m6 | 3 | the router's own time at each m, reported |

  At m1 arm H is T0 itself, so it is not run.
- **Why 20 processes.** A1b's B0 arm gave a CV of 2.05% over 10 processes. A ratio of two
  10-process means then has a CI95 half-width of ~2.1%, which leaves only ~0.9% of τ for the
  residual itself. With 20 processes per gated arm it is ~1.45%.
- **Order (fixed now):**
  1. Model hashes, then the pre-flight (45 s, mean below 4.39%), then K1 and K2.
  2. **Twenty rounds.** Round r visits the artifacts rotated by r (m1, m2, m4, m6, then m2, m4, m6,
     m1, and so on). At each artifact it runs that artifact's arms for the round, also rotated by r.
     The 10-process arms run in rounds 0-9, and the organ processes in rounds 0-5 (m1) or 0-2.
  - All of an artifact's arms run in one visit, so a reload happens only between artifacts.
  - In rounds 10-19 m6 has no arm left, and the round visits m1, m2 and m4 only.
- **Cost (ESTIMATE):** 230 processes (120 gated, 80 reported, 30 organ) at ~65 s, mostly the
  sequential fill, plus ~70 artifact reloads from the NVMe. That is **~4.5-5 h of empty machine** (owner's go, rule 7).

## Gate (frozen now)

The §1 equivalence test, in each router format f separately, at m ≈ 2 and m ≈ 4.

- **The rate of an arm:** the mean of its per-process median tok/s.
- **The router's price per byte:** c_f is the router's lap time per router byte, taken as the mean
  over the `--organs` m1 processes of format f, with the bytes as the engine counts them. It is
  measured at m = 1 only, so no number at m > 1 enters the prediction.
- **The prediction:** v_pred,f(m) = 1 / (1/v_f(1) + ΔR_f(m) · c_f), where ΔR_f(m) is the router's
  extra bytes per token from the table above: (E − 256) × 2048 × 19 values of 4 bytes (F32) or 2
  (BF16).
- **The residual:** ρ_f(m) = v_E,f(m) / v_pred,f(m).
  - **Its CI95:** a bootstrap of 10,000 draws, seed 20261016. It resamples the process medians of
    each arm and the m1 organ processes independently.
  - **PASS in format f** iff the CI95 of ρ_f lies inside [0.97, 1.03] (τ = 3%, the roadmap
    default) at both m2 and m4.
  - This is an equivalence test: noise widens the CI and fails it.
- **Outcomes:**
  - **Both formats PASS:** the Phase-1 invariance claim holds at m ≈ 2 and m ≈ 4.
  - **One format PASSes:** the claim holds in that format only; if that is BF16, it is the format
    S4 and S5 use (ROADMAP §5).
  - **Neither:** the rule-3 diagnostic budget applies before any T4 spend (ROADMAP §5 S1, stop).

## Reported, not gated

- **The raw ratio** v_E,f(m) / v_f(1), against the m1 arm of the same format, in the same session.
- **The BF16 expansions against the F32 T0,** labelled "includes the router upgrade".
- **The pool effect:** v_H,f(m) / v_f(1). The router bytes do not grow in arm H, so this ratio
  isolates the TLB, page and L3 effects.
- **The router's real cost per byte, from arm E minus arm H:** (1/v_E,f(m) − 1/v_H,f(m)) / ΔR_f(m),
  with its CI, at each m and in each format. Its constancy across m is checked against c_f.
- **The router's own lap time** at each m, from the `--organs` processes.
- **m6,** with every statistic above.
- **The selection spread** (`sel_distinct_frac`, `sel_copy0_frac`) of every arm.
  - If arm E's copy-0 share at m2 exceeds 0.75, arm E is labelled "unspread": its router bytes
    still grow, but its picks do not cover the pool.
  - The selections of arm H follow the fixed hash and spread by construction.
- **Arm M,** below.

## Arm M: the memory-bank lookup bench (reported, not gated)

- **Why:** it prices the conditional-memory route before S4 chooses between it and the expert
  pool.
- **The bench:** `c50_s1_membank.c`, pinned in the addendum.
  - **Tables:** synthetic tables of 1, 4, 16 and 48 GB in private memory, with rows of 4,096
    bytes (2,048 BF16 values).
  - **Lookups:** k ∈ {8, 32, 128} rows per token. The row addresses come from splitmix64(seed,
    token, j) modulo the row count: deterministic and spread over the whole table.
  - **The work per row:** its dot with a fixed F32 vector, in the BF16 router kernel's order.
  - **Timing:** 2,000 timed tokens after 200 warm-up tokens, on 6 threads with the engine's core
    map. 3 fresh processes per (table, k).
- **Reported:** ns per token and per row, the effective GB/s against A1's measured bandwidth, and
  the slope against table size at fixed k (the O(1) claim of ROADMAP §3).
- **When:** it runs after the 20 rounds in the same session, ~15 min (ESTIMATE).
- **Clock resolution:** the performance counter ticks at 100 ns, about 5% of a token at k = 8 on a
  small table. The per-process median over 2,000 tokens is what is reported.

## Tools (pinned now)

| Tool | Role | Pin |
| --- | --- | --- |
| `c50_s1_run.py` | runner: hashes, pre-flight, K1/K2, the 20 rounds, arm M; numbers only | this commit; it refuses other binaries, another engine source or unpinned artifacts |
| `c50_s1_report.py` | the gate and the reported statistics; chooses nothing | this commit |
| `bin/c50_engine_s1.exe` | the engine | `79e2e4f4…` |
| `bin/c50_engine.exe` | the A3 reference for K1/K2 | `d14ea9a5…` |
| `bin/c50_s1_membank.exe` | arm M, from `c50_s1_membank.c` (this commit), `clang -O2 -mavx2 -mfma` | `81a4f9cd…` |
| `c50_s1_expand.py` | the expansions | `1eb08f9` |

- **Schedule check:** `schedule()` yields 230 processes in 70 artifact visits, with the arm counts
  in the table above.
- **Logic check of the report, on synthetic runs** (scratchpad; no engine ran; not evidence):
  - **Router priced exactly, CV 2.05% per process:** both formats PASS. The CI95 half-widths of ρ
    are ~1.0% at m2 and m4, and the E − H estimate recovers the planted price per byte within
    noise.
  - **A planted 10% pool effect that the price does not include:** both formats FAIL, with ρ ≈ 0.90
    and arm H's pool effect ≈ 0.91.
- **Arm-M smoke** (0.25 GB, contended box; mechanics only, no rate): it runs and its checksum is
  deterministic.

## Build and session commands

The expansions are built on E: (USB), then staged on C:. Then come the addendum with their hashes,
and the session.

```
python -I benchmarks/chatbot50/c50_s1_expand.py --src E:/.../gguf/ling-mini-2.0-a810f64-q4_k_m.gguf --experts 536 --sigma 0.05 --seed 20261016 --dst E:/.../s1/t0-e536.gguf --out benchmarks/chatbot50/results/c50_s1_expand_e536.json
(the same for 1088 and 1600)
python -I benchmarks/chatbot50/c50_s1_run.py --out benchmarks/chatbot50/results/c50_s1_run_1.json
python -I benchmarks/chatbot50/c50_s1_report.py benchmarks/chatbot50/results/c50_s1_run_1.json --json benchmarks/chatbot50/results/c50_s1_report_1.json
```

## Registered predictions (ROADMAP §5; scored, not gated)

- At m ≈ 4 the raw slowdown is 10-16% with the F32 router and 5-9% with the BF16 router.
- The residual ρ lies within 3% in both formats.
- The BF16 router is byte-identical to T0 at m = 1 (K1).
- Arm H's pool effect at m ≈ 4 is below 3%.

## What this cannot say

- **One box, one artifact family, context 2048.** Other machines are Phase 3.
- **Copies are not knowledge.** S1 measures speed only. The quality half is S4 and S5.
- **Arm E's spread is the perturbed router's,** not a trained router's. A trained pool may select
  differently, and so touch memory differently.
- **The organ laps price the router at m1** and include its selection step. The E − H estimate is
  the independent check.
- **Page-cache residency is the B0 arm's.** `--mem map` keeps the weights in the file cache, so a
  pool that does not fit in RAM is out of scope here; m6 is the largest that fits.
