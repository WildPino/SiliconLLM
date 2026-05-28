# Silicon Entropy Engine (SEE) — V1.0

A CPU-native lossless data compressor using a streaming mixture-of-experts architecture. SEE encodes files using an ensemble of statistical models that compete byte-by-byte; their weights are updated online via an exponentiated-gradient MoE.

**What SEE is:** a lossless compressor. Encode → decode recovers the original file bit-for-bit.

**What SEE is not:** a language model, a generative system, or a general-purpose neural network. The wave substrate (silicon_entropy) is a fixed reservoir — only the linear readout is trained. The architecture is shaped by CPU cache topology, not GPU parallelism.

---

## Quick Start

```sh
# Encode
see encode input.txt output.see --weights weights/entropy_weights_factors_r16.bin

# Decode
see decode output.see recovered.txt --weights weights/entropy_weights_factors_r16.bin

# Audit BPB without writing an archive
see audit input.txt --weights weights/entropy_weights_factors_r16.bin
```

The `.see` archive is self-describing: the decoder reads all parameters from the header. No flags are needed at decode time other than the weights path.

---

## Expert Profiles

Select the expert set with `--expert-profile <name>` at encode or audit time.

| Profile | Expert set | Use when |
|---|---|---|
| `general` | LZ6 + TOKPFX | **Default. All domains.** Robust on prose, code, JSON, logs, binary. |
| `prose` | LZ6 + TOKPFX + TOK_PREV_ELIG | Modern prose with high word repetition (articles, docs, correspondence). |
| `experimental` | manual flags | Research/ablation only. Not for production use. |

`prose` is not a universal text profile. It targets word-transition patterns in modern prose and does not improve on literary prose from 1800–1920, source code, logs, or structured data. See `docs/profiles.md` for the full reference.

**Alias:** `--expert-profile text` is a backward-compatible alias for `prose`.

---

## Building

Single translation-unit build (requires Clang or GCC with AVX2):

```sh
gcc -O3 -march=native -o see.exe see.c \
    src/see_codec.c src/lz_topn.c src/tok_lz.c src/span_lz.c \
    src/moe_engine.c src/silicon_entropy.c src/silicon_v0.c \
    src/range_coder.c src/regime_prior.c \
    -lm
```

Tested on: Windows 11 + Clang 21, Zen 2 (Ryzen 5 3600X), AVX2.

**`-ffast-math` is forbidden.** It changes `expf()` rounding in the MoE weight update and CDF paths, producing a different bitstream. Archives encoded under `-ffast-math` cannot be decoded by a standard build. `-O2` and `-O3` produce identical output on this platform.

---

## Architecture

SEE is a streaming mixture of experts. At each byte position, 5 experts produce a probability distribution over 256 symbols; the MoE blends them and updates weights based on each expert's cross-entropy loss.

| Expert | Key | Strength |
|---|---|---|
| SEE | Wave ESN readout | C code, structured binary |
| BI | Bigram frequency | Markdown, Italian literary prose |
| UNI | Unigram frequency | Shuffled/random (fallback) |
| LZ | 6-byte hash → Top-N slots | JSON, C headers, logs |
| TOKPFX | 5-byte token prefix hash | Natural text, logs |
| TOK_PREV_ELIG *(prose only)* | Token-transition eligibility | Modern prose with repetition |

The wave substrate (silicon_entropy) is a fixed 128-block AVX2 reservoir (~4096 cells, fits in L1/L2 cache). Its dynamics are not trained; only the linear readout is learned offline. The design exploits cache-line locality: sequential access costs ~3 cycles, L3 miss costs ~56 cycles — the reservoir is sized to stay on-chip.

---

## Measured Performance (V1.0 baseline)

All results use `--expert-profile general --blend moe`.

| Corpus | BPB | Dominant expert |
|---|---|---|
| natural_text.txt | 2.474 | TOKPFX |
| markdown_docs.md | 3.887 | BI |
| c_code.c | 1.940 | SEE / LZ |
| json_synth.json | 2.033 | LZ |
| log_synth.log | 2.921 | TOKPFX |
| c_header_synth.h | 2.051 | LZ |
| shuffled.bin | 5.012 | UNI |
| log_real.log (Apache, 2.3 MB) | 0.981 | LZ |
| c_real.c (zlib inflate.c, 51 KB) | 2.970 | BI |
| prose_real.txt (Dreiser 1911, 512 KB) | 3.330 | BI |

TOKPFX value: +0.108 BPB on average versus no-token baseline (9 internal corpora).

Speed: ~60,000 cycles/byte on Zen 2 (`general` profile, full MoE).

---

## Regression Gate

Before any architectural change:

```sh
python scripts/regression_test.py
```

Reads frozen baselines (`data/baselines/phase29a_baseline.json` and `data/baselines/phase29c_baseline.json`), audits all corpora/profiles, and performs SHA-256 roundtrip on 3 corpora. Exit 0 = no regression. Tolerance: 0.005 BPB.

```sh
python scripts/test_headers.py
```

6 header integrity tests: bad magic, wrong weights, truncated header, profile roundtrip (×2), missing weights.

```sh
python scripts/phase35-36/phase35_reproducibility.py
```

Reproducibility audit: encode determinism, decode of committed fixture archives, header rejection (5 corruption cases), and optional compiler variant comparison (`--skip-compiler` to skip recompile). The fixture archives in `data/fixtures/` are the format identity test — if they fail to decode correctly, the archive format has regressed.

---

## Known Limits

- **Markdown gap**: +1.33 BPB above natural_text on structured markdown. MoE convergence lag is uniform across all byte positions — not a local gap. No expert combination closed this in Phases 27–28.
- **`prose` profile scope**: useful only for modern prose with high word-repetition. Literary prose 1800–1920 (rich vocabulary, complex syntax) is dominated by BI — `prose` saves <0.003 BPB on Dreiser.
- **Small files (<64 KB)**: MoE experts do not have enough context to converge. Results are indicative; dominant expert may differ from large-file behavior.
- **Weights binding**: each `.see` archive stores the SHA-256 of the weights file used at encode time. Archives encoded with an older weights file cannot be decoded with a newer one — re-encode to migrate.
- **Single-threaded**: the streaming MoE is inherently sequential. No parallelism within a file.
- **Version-locked archives**: `SeeArchiveHeader.header_size` is checked with strict equality at decode time. A future codec with a different header layout cannot decode V1.0 archives and vice versa.
- **Platform portability**: cross-compiler and cross-architecture bit-identical output is not guaranteed. Float rounding in the MoE/CDF path is compiler-dependent. Archives are stable within the same compiler family and flags (no `-ffast-math`).

---

## Repository Layout

```
SiliconLLM/
├── see.c                       Main CLI (encode / decode / audit)
├── build.bat                   Build script
├── src/
│   ├── see_codec.h/.c          Codec core: encode, decode, audit, MoE loop
│   ├── silicon_entropy.h/.c    Wave ESN (SEE expert)
│   ├── silicon_v0.h/.c         Wave substrate primitives
│   ├── lz_topn.h/.c            LZ Top-N hash expert
│   ├── tok_lz.h/.c             TOKPFX / TOK_PREV experts
│   ├── span_lz.h/.c            SPANPFX expert (experimental, rejected)
│   ├── moe_engine.h/.c         Fixed-share exponentiated-gradient MoE
│   ├── range_coder.h/.c        Arithmetic range coder
│   ├── regime_prior.h/.c       Regime prior router
│   └── archive/                Obsolete/superseded source files
├── weights/
│   ├── v1/                     V1.0 production weights
│   └── research/               Experimental weights (phases 36–42)
├── data/
│   ├── corpora/
│   │   ├── internal/           Synthetic + curated corpora (c_code, natural_text, …)
│   │   └── external/           Scraped/downloaded corpora (eureparl, kaggle, …)
│   ├── external/               Real-world files for Phase 29B/29C stress tests
│   ├── fixtures/               Format identity fixtures (tiny_*.see + manifest.json)
│   ├── baselines/              Frozen regression baselines (phase29a, phase29c, phase33)
│   └── phase_data/             Phase-specific binary datasets (phase32–42)
├── scripts/
│   ├── regression_test.py      Full regression harness (primary gate)
│   ├── test_headers.py         Header integrity tests
│   ├── phase29/                Phase 29 tribunal + baseline scripts
│   ├── phase31-34/             Regime routing research scripts
│   ├── phase35-36/             Reproducibility audit
│   ├── phase37-40/             Multi-domain / MoE research scripts
│   ├── tooling/                Analysis, plotting, and one-off utilities
│   └── archive/                Superseded scripts (phases 21–28)
├── benchmarks/
│   ├── phase01-14/             Early architecture benchmarks (C)
│   ├── phase18/                Coder + readout training benchmarks
│   └── phase38-42/             Phase 38–42 experiment harnesses
├── bin/
│   ├── phase01-14/             Built binaries for early benchmarks
│   ├── phase18-23/             Built binaries for phases 18–23
│   ├── phase38-42/             Built binaries for phases 38–42
│   └── misc/                   Utility binaries (wave_engine, test_rc)
├── results/
│   ├── phase11–14/             Per-phase result files
│   ├── phase20–42/             Per-phase result files
│   └── misc/                   Unphased result files
├── experiments/
│   └── phase41a/               Phase 41 active experiment
├── docs/
│   ├── profiles.md             Expert profile reference
│   ├── architecture_decisions.md  Architecture decision log
│   ├── see_v1_position.md      External compressor comparison (Phase 31)
│   ├── phases/
│   │   ├── phase35_reproducibility.md
│   │   ├── early/              Phase 1–22 walkthrough notes
│   │   └── archive/            Superseded phase docs
│   ├── research/               Background research notes (CPU arch, derivatives)
│   └── archive/                Superseded docs (v0_architecture, project_summary, …)
└── logs/
    └── phase10/                Phase 10 run logs
```

---

## Research Context

SEE was developed through a sequence of measurement-driven phases (24–35). Each phase posed a specific hypothesis, ran a controlled tribunal, and either promoted or rejected a change. The result is a small, stable set of expert components — not because alternatives weren't tried, but because most were rejected by the data.

Phase 31 measured SEE against zlib-9, bz2-9, lzma, zstd-22, and brotli-11. SEE ties classical compressors only on shuffled/random data. On every structured domain it loses, with gaps ranging from +0.7 BPB (prose) to +2.0 BPB (markdown). This is the honest external baseline.

Phases 32–34B explored regime routing (credit-dynamics-based domain detection). Finding: compression credit contains regime signal, but no single router dominates all corpora. The research is archived in `docs/research/regime_routing_research.md`; not merged into core V1.

Phase 35 confirmed physical reproducibility: deterministic output, committed format fixtures, `-ffast-math` documented as forbidden.

For now: the body is stable and its limits are known. See `CHANGELOG.md` for the full phase history.
