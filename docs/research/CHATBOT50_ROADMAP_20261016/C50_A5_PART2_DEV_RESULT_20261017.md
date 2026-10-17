# C50-A5 part 2 result: the selected head approximation fails its DEV confirmation (0.99874 < 0.999)

17 October 2026. Cell A5 of the [CHATBOT50 roadmap](ROADMAP.md) §5, under the
[part 2 protocol](C50_A5_PART2_DEV_PROTOCOL_20261017.md). DEV was read once, here.

**Verdict: FAIL** (applied by `c50_a5_select.py --grid dev`).
- Configuration: `rot_r512_q3sim`, K = 32,768, 94.45 MB/token.
- DEV top-1 recall: **0.99874**, i.e. 98 misses in 78,079 positions over 36 documents. FIT
  had given 0.99921.

**Registered prediction scored:**
- DEV recall in 0.9985-0.9995: **right**.
- About 60% probability of PASS: the outcome went the other way. One draw cannot score
  a probability; it is recorded as given.

## Run

- **Capture:** `c50_engine_a5.exe --dump-hidden` on `dev12_ids.tsv` (sha256 `ab741ea7…`).
  - The CONFIG line shows the pinned model, `hidden=1` and 78,079 positions.
  - It took 2,208 s.
- **Confirmation:**
  - The frozen files' sha256 matched (`56a0b409…`, `921f9d3f…`).
  - The Q8_K oracle's rank-code recall at 1 was 1.0 on 7,808 positions.
  - It took 58 s.
- **Records:** `results/c50_a5_dev.json`, `results/c50_a5_gate_dev.json`.

## Where the misses are (read after the verdict; it changes nothing)

| DEV category | Positions | Misses | Recall |
| --- | --- | --- | --- |
| code | 27,827 | 85 | 0.99695 |
| technical_general | 26,132 | 13 | 0.99950 |
| prose | 24,120 | 0 | 1.00000 |

- **The misses are concentrated.**
  - They fall in 16 of the 36 documents, and two documents hold 46 of the 98.
  - The lowest per-document recall is 0.98924.
- **They are confident tokens.** The engine's p(top1) at a miss has median 1.00, and
  88.8% of misses have p ≥ 0.9.
  - These are not near-ties. The preview ranks below 32,768 rows a token the exact head
    is nearly certain of. The cause is probably a code token whose row lies largely
    outside the top-512 eigenspace (a hypothesis, not tested).
  - A confidence fallback would not catch them, consistent with the FIT strata in part 1.
- **Larger K is the only direct fix.** On DEV, recall at K = 65,536 is 0.99965, but that
  costs ~149.5 MB/token.
  - This figure comes from DEV: it is not a selection and cannot become one.

## Registered consequences (part 2 protocol)

- **A5 fails.** The rule does not move to the runner-up (full-rank ternary, K = 256,
  101.0 MB, FIT 0.99926) by itself.
- **DEV is spent for A5.** A new selection would need a new FIT protocol and a fresh
  confirmation set.
- **The decision goes to the owner.** COMMUNICATION item 3 sets out the options with the
  ledger arithmetic below.

## Ledger arithmetic for the decision (DESK-MODEL, bytes only; desk = 1 / (MB/32 GB/s + 2 ms))

| Ling-mini policy at context 2048 | MB/token | desk tok/s | × 0.85 |
| --- | --- | --- | --- |
| P4 with the exact Q6_K head (264.1) | 701.3 | 41.8 | 35.5 |
| P4 with the selected head (94.5), had it passed | 531.7 | 53.7 | 45.7 |
| P4 with the full-rank ternary head (101.0), had it been selected and confirmed | 538.2 | 53.1 | 45.2 |
| P4 with the ledger's placeholder head (23.8) | 461.0 | 60.9 | 51.8 |

**Reading (each sentence names its row):**
- **Even with a PASS, 50 needed a runtime above 0.93 of desk.** The A5 rows show that
  even a passing head would have needed ≥ 0.93 of desk for 50 tok/s (50/53.7). The
  measured complete runtimes reached 0.85 and 0.53.
- **The roadmap's 50 at 2048 depended on the ~24 MB placeholder head** (last row), and
  no measured head comes near it: the head is not low-rank (part 1).

## What this cannot say

- **No rate:** every row above is DESK-MODEL.
- **No quality statement:** a 0.13% top-1 disagreement on teacher-forced positions is not
  a measured chat-quality loss. A4 measures quality, and A4 has not run.
- **Scope of the FAIL:** the FAIL is against the roadmap bar of 0.999 and says nothing
  about whether that bar is the right one.
  - For scale, llama.cpp differs from itself by ~4% top-1 across builds (A2b).
  - Changing the bar is the owner's call. Under rule 4 it cannot be done after reading
    DEV.
