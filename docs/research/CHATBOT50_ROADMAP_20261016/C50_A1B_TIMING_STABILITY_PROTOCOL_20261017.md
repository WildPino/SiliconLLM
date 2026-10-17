# C50-A1b protocol: timing stability of the engine across fresh processes

17 October 2026. Cell A1b of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed before any A1b
process runs.

## Question

- **The problem.** The complete Qwen-0.5B runtimes showed process-to-process spreads of
  1.14-1.63 (ROADMAP §2). Until the spread is small, a rate gate on a lower confidence bound
  measures the spread and not the rate.
- **Gate (ROADMAP §5 A1b, unchanged):** across 10 fresh processes, max/min of the
  per-process median tok/s ≤ **1.06**.

## What the engine does now, and the levers

`c50_engine.c` (the A3 engine) already pins one thread per physical core, on the first logical
processor of each core: `cores=0,2,4,6,8,10`, three per CCX on the 3600X. Its waits are spin
waits (`_mm_pause`), with no sleep. Two causes remain plausible, neither yet measured.

- **The weights are a read-only file mapping.** Their pages are faulted on first touch, they
  can be trimmed from the working set, and they sit wherever the standby list left them.
- **The process runs at normal priority.**

**Levers.** These are new options, default off, printed on the CONFIG line, and built as
`bin/c50_engine_a1b.exe` (sha256 `6cbafb90…`). The A3 binary is not rebuilt.
- `--mem private`: the whole GGUF image is copied into private memory and locked in the
  working set (`SetProcessWorkingSetSizeEx` hard minimum + `VirtualLock`). Every pointer is
  rebased.
- `--mem large`: the same copy on large pages. This needs `SeLockMemoryPrivilege`; without it
  the engine exits 4 and prints why.
- `--prio high`: `HIGH_PRIORITY_CLASS`.

**Parity, measured before this push** (a correctness check, not a timing):
- **Method:** a 200-token SEL document (199 positions) was scored, and the output compared
  with `cmp` against the A3 binary.
- **Result:** `--mem map --prio high` and `--mem private --prio high` are **byte-identical**.
- **Large pages:** `--mem large` exits 4 with "SeLockMemoryPrivilege not held". The privilege
  is not granted on this account.

## Arms, order, instrument

| Arm | Options | Note |
| --- | --- | --- |
| B0 | `--mem map --prio normal` | the A3 path |
| B1 | `--mem private --prio high` | |
| B2 | `--mem large --prio high` | only if the runner's probe shows the privilege; otherwise recorded as not run |

- **Instrument:** `--bench` on P0 (`44de7315…`), context 2048, 128 steps.
  - The document is the first of the A4 FIT document panel with ≥ 2,176 tokens. That is a
    timing input only; no quality is read.
  - One process gives one median step time, so one median tok/s.
- **Order:** 10 rounds. Within a round, B0, B1 (then B2) in fresh processes. A drift of the box
  therefore hits every arm alike.
- **Occupancy:**
  - Pre-flight of 45 s, admissible iff the mean foreign occupancy is below 4.39%: the E52 bar,
    under the A1 addendum 2 rule.
  - Each process is split into its own and foreign occupancy (E44). A process at or above the
    bar is VOID and replaced; more than 5 replacements in one arm make the run VOID.
- **Scripts:** the runner `c50_a1b_run.py` prints numbers only; the verdict comes from
  `c50_a1b_verdict.py`.
- **Cost:** ~70 s per process (the copy of 9.9 GB, a 2,048-token fill, 128 steps), ~25-35 min
  in total. **It needs an empty machine** (owner action), after the A4 → A6 chain.

## Verdict rule (fixed now)

- **Per arm:** PASS iff all 10 processes are admissible and max/min of their 10 medians ≤ 1.06.
- **A1b PASSES iff some arm passes.** The adopted arm is the first passing one in the order
  B0, B1, B2, so the simplest configuration wins. A7 then uses the adopted arm's options.
- **On FAIL:**
  - no lever is added after reading;
  - a new protocol is needed;
  - the roadmap's diagnostic budget applies (3 cells / 3 days);
  - and A7 cannot gate on a lower CI.
- **Reported, not gated:**
  - the mean and min..max of the medians per arm;
  - the within-process max/min step;
  - the foreign occupancy;
  - replacements.
  - The medians are the first engine rates on a clean box at context 2048. They are recorded
    as such, but **they are not the A7 rate**: A7 runs on the final artifact.

## Registered prediction (scored, not a gate)

- **B0:** max/min 1.04-1.15. The spread is smaller than Qwen's 1.14-1.63, because the pinning
  and spin waits are already in place.
- **B1:** ≤ 1.06 (PASS).
- **Median rate of B0 and B1 at context 2048:** 24-30 tok/s. They lie within 3% of each other.

## What this cannot say

- **One box, one artifact.** The gate's scope is this 3600X and P0; another artifact needs its
  own check in A7.
- **Not a rate gate:** stability is a precondition of A7.

## Addendum 1 (17 October 2026, before any A1b process)

- **Why:** S0 extends `c50_engine.c`, which this runner hashed next to the pinned binary. The
  recorded source hash would then no longer have matched the binary.
- **The fix:** the A1b source is frozen as `benchmarks/chatbot50/c50_engine_a1b.c`. It is a byte
  copy of the 7ab2048 `c50_engine.c` (sha256 `f874031b…`).
- **The runner now:** records the hash of that copy, and refuses to run unless
  `bin/c50_engine_a1b.exe` is the pinned `6cbafb90…`.
- **Unchanged:** the arms, order, gate and verdict script.
- **Details:** in the [S0 protocol](C50_S0_V0_PROTOCOL_20261017.md).
