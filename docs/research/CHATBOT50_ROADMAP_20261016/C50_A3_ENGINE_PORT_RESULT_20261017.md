# C50-A3 result: the engine runs the P0 artifact within llama.cpp's own disagreement

17 October 2026. Cell A3 of the [CHATBOT50 roadmap](ROADMAP.md) §5, run under the
[A3 protocol](C50_A3_ENGINE_PORT_PROTOCOL_20261016.md) and its addenda 1-2.

**Verdict: PASS** (`benchmarks/chatbot50/results/c50_a3_verdict.json`).

## Run

- **Binary and sources:**
  - Binary: `benchmarks/chatbot50/bin/c50_engine.exe`, sha256
    `d14ea9a5903557342763c5ebb7eb18897333c86a4484c24566b42c8627c9035d`.
  - Source hashes are in `benchmarks/chatbot50/results/c50_a3_run.json`.
- **Self-test, re-run by the runner with the same binary: PASS.**

  | Type | worst \|AVX2 − generic\| | worst \|generic − oracle\| | planted detected |
  | --- | ---: | ---: | ---: |
  | Q4_K | 2.43e-7 | 1.71e-7 | 197/200 |
  | Q5_K | 3.22e-7 | 2.05e-7 | 193/200 |
  | Q6_K | 2.63e-7 | 1.90e-7 | 198/200 |
  | Q8_0 | 1.34e-7 | 1.09e-7 | 199/200 |

  The bound is 1e-5 on every type.
- **Artifact and positions:** the P0 artifact (sha256 `44de7315…`) on the 26,363
  A2b positions (12 FIT documents). The `CONFIG` line confirms 26,363 positions,
  dump on, 6 threads on cores 0,2,4,6,8,10, F16 KV, F32 attention, RoPE base
  600000, weights scale 2.5, and the types that were loaded.
- **Wall time:** 1,148 s for 26,363 positions with 157,184 log-probabilities
  dumped per position, on a box that was not clean. It is not a rate (protocol
  "recorded beside the gate").

## Gate (engine against R3, llama.cpp's single-token path)

| Statistic | Engine vs R3 | Limit | |
| --- | --- | --- | --- |
| top-1 disagreement | **3.99%** (CI 3.07-5.01) | ≤ 8.20% | PASS |
| mean KL(R3 ‖ engine) | **0.0144 nats** (CI 0.0120-0.0169) | ≤ 0.0299 | PASS |
| \|dBPB\| point | **+0.00051** | ≤ 0.00444 | PASS |
| dBPB 95% CI (reported) | (−0.00177, +0.00267) | — | — |

**Disclosure.** Under the dBPB form first registered ("CI inside ±0.002"), this run
would have FAILED, because the upper CI end is +0.00267. The form was changed by
[A3 addendum 2](C50_A3_ENGINE_PORT_PROTOCOL_20261016.md#addendum-2-17-october-2026-after-the-a2b-floor-before-any-engine-output-on-p0)
after the A2b floor was read and before any engine output on P0. The reason was
that both llama.cpp floor pairs fail the old form against themselves (CI ends
−0.0048 and −0.0038). The point estimate, +0.0005, is a quarter of the old 0.002
constant.

## Beside the gate

| Pair | top-1 | mean KL (nats) | p99 / max KL | dBPB (CI) |
| --- | --- | --- | --- | --- |
| R3 vs engine (gate) | 3.99% | 0.0144 | 0.185 / 3.96 | +0.0005 (−0.0018, +0.0027) |
| R0 vs engine | 4.07% | 0.0151 | 0.194 / 3.31 | −0.0009 (−0.0036, +0.0016) |
| R0 vs R3 (A2b) | 3.84% | 0.0139 | 0.174 / 3.68 | −0.0014 (−0.0038, +0.0009) |
| R0 vs R2 (A2b) | 4.10% | 0.0149 | 0.188 / 3.85 | −0.0022 (−0.0048, +0.0003) |

The engine is **no closer to R3 than to R0**, even though R3 has the same
computation shape (one token per forward). Its distance from either llama.cpp
path matches the distance between llama.cpp's own paths. The engine, R0, R2 and
R3 behave as four roughly equidistant points: each change of rounding order
moves ~4% of top-1 choices.

The heavy tail (p99 ~0.18 nats, max ~4 nats) is the same in every pair. That fits
the hypothesis that rounding differences are amplified by discrete routing flips.
Routing was not instrumented, so it remains a hypothesis.

## What this result says and does not say

- **It says** the engine computes the bailingmoe2 forward on the P0 artifact as
  llama.cpp does, to within llama.cpp's own sensitivity. Nothing in the
  architecture semantics (QK-norm, partial RoPE, group routing, scale 2.5,
  shared expert, dense layer 0, head) is wrong by more than that
  sensitivity. The complete P0 forward runs. Programme rule 1 also asks for
  the forward's per-organ cost breakdown, which is not measured yet (it needs a
  clean box; A7 and the bench mode).
- **It does not say** the engine is free of errors smaller than ~2× the floor.
  The gate catches errors well beyond llama.cpp's own build and path
  disagreement; a systematic bias below ~0.004 BPB or a rare path difference
  would pass. **No planted fault** (for example a deliberately mis-scaled router)
  was run to show the gate fails a known-bad engine. The kernel self-test has
  planted controls; the system gate does not yet. Planted-fault runs are
  registered in A3 addendum 3, and their outcome is recorded below the verdict,
  not in it.
- Nothing about quality (A4) or speed (A7). The wall time above includes the
  dump and a busy box.
- FIT positions only.

## Next

- A1, the kernel rate, and the A2b external rate both wait for the owner's
  clean-box session (`docs/COMMUNICATION.md`, item 1).
- A4 (P0 quality against BF16) is next in order: a frozen task panel and
  fresh-document dBPB.

## Planted-fault control (A3 addendum 3; outside the verdict)

Two planted engines went through the unchanged verdict against R3. Each `CONFIG`
line carries its planted value. Record: `benchmarks/chatbot50/results/c50_a3_planted.json`.

| Engine | top-1 | mean KL (nats) | p99 KL | dBPB (CI) | Verdict | Registered expectation |
| --- | --- | --- | --- | --- | --- | --- |
| healthy (gate run) | 3.99% | 0.0144 | 0.185 | +0.0005 (−0.0018, +0.0027) | PASS | — |
| `PLANT_WSCALE` (routed scale × 0.9) | 6.87% | **0.0443** | 0.525 | −0.0084 (−0.0173, −0.0001) | **FAIL** (KL, dBPB) | FAIL: **met** |
| `PLANT_ROPE` (RoPE base 500000) | 5.15% | 0.0238 | 0.292 | −0.0020 (−0.0055, +0.0014) | **PASS** | none; recorded as a blind spot |
| limit | 8.20% | 0.0299 | — | \|point\| ≤ 0.0044 | | |

**What this establishes.**

1. **The gate fails a known-bad engine.** A 10% error on the routed contribution
   fails it on two criteria. So the A3 PASS is not a gate that cannot fail.
2. **The margin is modest.** That fault's KL is 1.5× the limit, so a routed-scale
   error of ~5% would probably pass. The top-1 criterion did not fire on
   either plant: at this floor it is the weakest of the three.
3. **A RoPE-base error of −17% passes.** It is a fault class this gate cannot
   see. Its KL (0.024) is, however, 1.65× the healthy engine's (0.0144), and
   their bootstrap intervals do not overlap (healthy 0.0120-0.0169). A sharper
   check exists: compare a candidate against the healthy engine's own distance
   from R3, not against 2× the llama.cpp floor. It is not a gate here; any later
   engine change that needs one must register it.
4. Nothing is relabelled. The A3 verdict stands as PASS, scoped to "no error
   larger than the routed-scale plant".
