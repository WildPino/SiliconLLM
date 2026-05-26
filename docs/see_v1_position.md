# SEE V1.0 — Honest Positioning

*Written after Phase 31 (External Compressor Atlas). This document replaces any prior summary that described SEE as a general-purpose compressor candidate.*

---

## What SEE V1.0 is

A **lossless autoregressive compressor with streaming expert mixture**, running on CPU, encoding one byte at a time.

Core properties:
- Autoregressive: each byte is predicted by a mixture of experts conditioned on all prior bytes.
- Streaming: experts update their state after every byte; no look-ahead, no block reprocessing.
- Expert mixture: a MoE credit-assignment loop dynamically weights active experts per byte.
- Self-describing archive: the `.see` header stores expert profile flags and a SHA-256 of the weights file. Decode requires no external configuration.
- Regression-gated: `scripts/regression_test.py` (55 tests) is the mandatory gate before any core change.

Active experts in V1.0:
- **LZ Top-N (LZ6)**: dictionary expert, hash-based, cache-friendly, ~32KB effective window.
- **TOKPFX**: symbolic expert, predicts inside-token bytes from the token's prefix hash.
- **TOK_PREV_ELIG** (prose profile only): gated token-transition expert, activates at eligible byte positions.
- **BI / UNI**: bigram and unigram priors, always present as baseline.

---

## What SEE V1.0 is not

**Not a general-purpose compressor.** Phase 31 measured SEE against zlib-9, bz2-9, lzma, zstd-22, and brotli-11 across 14 corpora. Result: SEE ties classical compressors only on shuffled/random data (+0.003 BPB vs brotli-11). On every structured domain it loses, with gaps ranging from +0.7 BPB (prose) to +2.0 BPB (markdown).

**Not an LLM.** TOKPFX finds signal in a specific training distribution. It does not generalize across prose registers — the -0.045 BPB gain on `natural_text` does not appear on literary prose (Dreiser, Manzoni: +0.003 BPB, no gain). TOKPFX is a compression organ that found a local pattern, not a language model.

**Not a BWT/LZ77 replacement.** LZ6 (hash-based, fixed key, Top-N candidates) is cache-efficient and elegant, but does not capture long-distance repetition. On real C code, SEE is ~2x worse than brotli-11 (1.985 vs 1.004 BPB on `c_code.c`). BWT and LZMA win because they work with a full suffix-sorted dictionary; LZ6 cannot replicate this.

---

## Where SEE's value actually lives

The compression ratios are not the contribution. The contribution is the **architecture as a laboratory**:

- A working autoregressive MoE with byte-level expert competition and credit assignment.
- A streaming inference loop that is deterministic, reproducible, and regression-gated.
- A clean header format (92 bytes) that binds archive to model: no silent corruption.
- An empirical methodology that rejected experts honestly (SPANPFX, BOL, dual-LZ, TOK_PREV ungated) before they could bloat the design.
- A measurement discipline: every architectural claim is backed by a frozen baseline, a tribunal result, and a regression number.

The silicio has not chosen to become an LLM yet. It has chosen to be an honest laboratory for measuring which compressive organs deserve to exist.

---

## Known gaps, characterized

| gap | domain | delta BPB (avg) | character |
|-----|--------|-----------------|-----------|
| markdown | structured text | +1.96 | Frequent switching between prose/code/math/markup micro-regimes. Not a single missing pattern. BOL and SPANPFX already failed here — the problem is regime segmentation, not a better predictor within one regime. |
| JSON / structured data | semi-structured | +1.33 | High structural redundancy at long range. LZ6 window too small. |
| code | source code | +1.22 | Long-distance repetition (function names, patterns across 10KB+). BWT territory. |
| prose (all registers) | natural language | +0.73–1.17 | Distributional gap: TOKPFX trained on synthetic prose, not literary registers. bz2 wins via BWT on long repeated phrases. |
| log (high volume) | structured log | +0.58–1.57 | Log lines are highly repetitive at long range. The +172KB absolute loss on log_real is the largest single-file damage. |

---

## What next architectural work looks like (if it resumes)

The rational target is markdown/mixed-regime, not prose. But the solution is not "add a markdown expert." Phase 28 already showed that local delimiter-based experts (BOL, SPANPFX) do not help.

The framing that fits the data better:

> Markdown is a problem of **frequent switching between micro-languages** — prose, code-ish, math-ish, markup-ish — that the current experts do not segment. The gap is in the **router**, not in the predictors.

A regime-detection layer — not hardcoded classes, but compressive signals identifying the local regime — could gate existing experts more precisely and allow small dedicated experts to compete inside a detected regime. This is a different architectural direction than adding another expert to the current flat MoE.

This is not a Phase 32 plan. It is a description of what the Phase 31 data points toward, for future reference.

---

## Regression baseline (V1.0)

Before any change to the core, run:

```
python scripts/regression_test.py
```

Exit 0 = green. Any core change that breaks this is rejected before it reaches a tribunal.

Baseline sources:
- `data/baselines/phase29a_baseline.json` — primary (9 internal corpora, 4 profiles)
- `data/baselines/phase29c_baseline.json` — stress test (4 external corpora)
- 3 SHA-256 roundtrip checks (encode → decode → verify)
