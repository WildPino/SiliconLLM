# P4.3 — Engine consolidation: acceptance report

**Date:** 2026-07-03 · **Deliverable:** `benchmarks/phase60/engine.c` — one core, feature-flags, replacing
the five stage engines. Stage engines stay buildable as the parity oracles (archival, per the E4 law:
kernel self-tests never suffice; acceptance is END-TO-END parity per configuration).

## Pre-registered acceptance

For every stage configuration, the consolidated core must produce a **bit-identical raw fp32 logit
stream** (md5 of 2048 tokens × 1024 logits, windows of 512 with state-reset, val offset 0 — the shared
gate protocol) versus the corresponding archived stage engine. Bit-identical logits subsume every
numeric parity gate of the stage ladder (BPB, top-1, bit-identity) on the tested stream. tok/s of the
consolidated core are re-measured only at training end (no timing claims here).

## Parity matrix

| # | stage config | archived oracle | consolidated flags (`bin/engine`) | md5 (both files) | verdict |
|---|---|---|---|---|---|
| 1 | E1 fp32 + exact-exp | `e1_engine` | `--mlp fp32 --skip off --exp exact` (E1M1) | `db2413043d213673a45d025447f33f99` | **IDENTICAL** |
| 2 | E2 LUT full + exact-exp | `e2_engine` (mlp 1) | `--mlp lut --skip off --exp exact` (E1M1) | `d318292495282fcdea3a11524d7eb83a` | **IDENTICAL** |
| 3 | E3 LUT skip + exact-exp | `e3_engine` (mlp 0) | `--mlp lut --skip on --exp exact` (E1M1) | `d318292495282fcdea3a11524d7eb83a` | **IDENTICAL** |
| 4 | E3.5-iso fp32 + fast-exp | `e35_engine` (scan 1, mlp 0) | `--mlp fp32 --skip off --exp fast` (E1M1) | `237e0e49d4ac419f812df7a1d538073b` | **IDENTICAL** |
| 5 | E3.5 full LUT+skip + fast-exp | `e35_engine` (scan 1, mlp 2) | `--mlp lut --skip on --exp fast` (E1M1) | `f395980c803e6e5358e0fd67dc62f432` | **IDENTICAL** |
| 6 | E4-ref fp32 experts + exact | `e4_engine` (scan 0, mlp 0) | `--mlp fp32 --exp exact` (E4M1) | `834b3c09710f2fee0e112e9ace57de01` | **IDENTICAL** |
| 7 | E4 full LUT experts + fast | `e4_engine` (scan 1, mlp 1) | `--mlp lut --exp fast` (E4M1) | `d3a722532eb8180252984065b50f80d0` | **IDENTICAL** |

**7/7 IDENTICAL — acceptance met.** Observed bonus: rows 2 and 3 share one md5 — the E3 exact-skip
bit-identity gate re-confirmed at the logit-stream level on this run.

## Instruments

- `--dumplogits <file>` added to all five archived engines and the consolidated core (diagnostic only;
  no gate semantics touched).
- `--kselftest` on the consolidated core: all LUT paths (full / row-scalar / tile-skip / windowed-rows)
  bit-exact vs scalar-int on synthetic ternary weights + poly-exp vs libm ≤2e-6 on the documented domain.
  Runs with no weights — the CI entry point (`make selftest`).

## Status

**All seven rows green — acceptance met (2026-07-03).** Stage engines moved to
`archive/benchmarks/phase60_stage_engines/` (kept buildable via `make stage-engines`; they remain the
parity oracles for any future re-acceptance). The consolidated core is the maintained engine from here.
tok/s re-measurement deferred to training end, per the work order.
