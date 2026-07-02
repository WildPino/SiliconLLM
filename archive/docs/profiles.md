# SEE Expert Profiles

Expert profiles are named configurations for the Silicon Entropy Engine's MoE expert set. They are selected with `--expert-profile <name>` at encode/audit time and written into the archive header — the decoder reconstructs the same expert set from the header without needing the flag.

---

## `general` — default, all domains

**Expert set:** LZ6 + TOKPFX  
**Flags set:** `SEE_FLAG_LZ_ENABLED | SEE_FLAG_TOK_PREFIX`

The recommended profile for all uses. Robust across every domain tested in Phase 29A/29C.

**When to use:** always, unless you have a specific reason to use `prose`.

**Measured performance (Phase 29A internal baseline):**

| domain         | BPB    | dominant |
|----------------|--------|----------|
| natural text   | 2.474  | TOKPFX   |
| markdown       | 3.887  | BI       |
| C code         | 1.940  | SEE/LZ   |
| JSON/headers   | 2.033  | LZ       |
| Italian prose  | 3.656  | BI       |
| logs           | 2.921  | TOKPFX   |
| shuffled       | 5.012  | UNI      |

TOKPFX saves +0.108 BPB on average across 9 corpora versus `no-token` baseline.

---

## `prose` — optional, modern text with word repetition

**Expert set:** LZ6 + TOKPFX + TOK_PREV_ELIG  
**Flags set:** `SEE_FLAG_LZ_ENABLED | SEE_FLAG_TOK_PREFIX | SEE_FLAG_TOK_PREV | SEE_FLAG_TOK_PREV_ELIG`

Adds a gated token-transition expert (TOK_PREV_ELIG) that activates only at eligible bytes — the start of a repeated token. Useful when the text has strong word-to-word transition patterns.

**When to use:**
- Modern English prose: articles, documentation, correspondence, chat logs
- Text with high word-level repetition (many common function words, formulaic phrases)

**When NOT to use:**
- Literary prose from 1800–1920 (rich vocabulary → BI dominates, TOK_PREV_ELIG silent)
- Source code, JSON, logs, binary (LZ or BI dominates; `prose` adds no gain)
- Any file where `general` already works well

**Measured degradation outside prose domain (Phase 29A):**

| domain         | Δ vs general |
|----------------|-------------|
| natural text   | −0.045 BPB  (gain)  |
| markdown       | +0.003 BPB          |
| C code         | +0.002 BPB          |
| JSON/headers   | +0.004 BPB          |
| shuffled       | ±0.000 BPB (correct: silent) |

Maximum degradation on external corpora (Phase 29C): +0.0076 BPB on `c_real.c` (zlib). Well within the +0.010 safety threshold.

**Note on naming:** The profile is named `prose` because it targets word-transition patterns common in modern prose. It does NOT generalize to all prose — literary English 1911 (Dreiser, Jennie Gerhardt, 512KB) shows BI as dominant, not TOKPFX. `prose` is shorthand for "modern prose with token repetition", not for "any text".

**Alias:** `text` (backward compatibility — same flags, no functional difference).

---

## `experimental` — manual expert configuration

**Expert set:** user-defined via individual flags  
**Flags set:** as specified by `--tok-*`, `--lz-*`, `--span-pfx` flags

Not a stable profile. Used for research and ablation. The `--span-pfx` flag (SPANPFX expert) was evaluated in Phase 28 and rejected: W_SPAN=0.0145 on markdown, no improvement on any domain. Do not use `span-pfx` in production.

---

## `no-token` (ablation only)

Not a real profile name — constructed in research scripts as `--expert-profile experimental --lz-key 6` without TOKPFX. Used to measure TOKPFX value. Do not use in production.

---

## Archive header flags

Each profile is stored in the archive as a bitmask in `SeeArchiveHeader.archive_flags`:

| Flag                  | Value  | Meaning                                  |
|-----------------------|--------|------------------------------------------|
| `SEE_FLAG_LZ_ENABLED` | `0x01` | LZ Top-N expert is active                |
| `SEE_FLAG_MOE_ACTIVE` | `0x02` | MoE credit assignment active (always on) |
| `SEE_FLAG_LZ_DUAL`    | `0x04` | Dual-LZ mode (LZ4+LZ8 as separate experts) — archived, not used |
| `SEE_FLAG_TOK_PREFIX` | `0x08` | TOKPFX expert (inside-token prefix hash) |
| `SEE_FLAG_TOK_PREV`   | `0x10` | Token-transition expert (TOK_PREV)       |
| `SEE_FLAG_TOK_PREV_ELIG` | `0x20` | TOK_PREV gated (eligible bytes only) |
| `SEE_FLAG_SPAN_PFX`   | `0x40` | SPANPFX delimiter expert — rejected, experimental only |

The decoder reads `archive_flags` from the header and reconstructs the exact expert configuration without any external flags. A `.see` file is self-describing.

---

## Weights binding

Starting from SEE V1.0 (Phase 30), the archive header stores a SHA-256 of the weights file used at encode time (`SeeArchiveHeader.weights_sha256`). The decoder verifies this at startup and rejects archives encoded with a different weights file:

```
see: weights mismatch — archive was encoded with different weights
     archive sha256: 620ddfe0907f70e9...
     current sha256: acfc3c15d38e73a2...
```

This prevents silent corruption when weights files are updated. Re-encode with the new weights to migrate an archive.

---

## Phase 31: SEE vs classical compressors (atlas)

Measured on 14 corpora spanning 7 domains. Classical compressors at max or near-max quality settings (bz2-9, lzma, zstd-22, brotli-11).

**BPB summary — SEE-general vs best classical per domain:**

| domain         | SEE-general (avg) | best classical (avg) | gap    | best compressor |
|----------------|------------------:|---------------------:|-------:|-----------------|
| random/shuffled | 5.015            | 5.012                | +0.003 | brotli-11 (TIE) |
| prose_synth     | 2.478            | 1.744                | +0.734 | bz2-9           |
| prose_it_lit    | 3.187            | 2.201                | +0.986 | bz2-9           |
| prose_en_lit    | 3.331            | 2.164                | +1.168 | bz2-9           |
| prose_it        | 3.661            | 2.628                | +1.033 | bz2-9           |
| log             | 0.981–2.926      | 0.400–1.354          | +0.581–1.572 | brotli-11 / bz2-9 |
| code            | 1.943–2.985      | 1.004–1.485          | +0.940–1.500 | brotli-11       |
| json            | 2.038            | 0.706                | +1.333 | bz2-9           |
| markdown        | 2.653–5.115      | 0.657–2.865          | +1.627–2.245 | brotli-11    |

**Key findings (Phase 31):**

- SEE ties classical only on shuffled/random (+0.003 BPB vs brotli-11). No WIN on any structured domain.
- `prose` profile gains -0.045 BPB on `natural_text` (synthetic internal) but does NOT transfer to unseen literary prose (Dreiser, Manzoni: +0.003 BPB at most). TOKPFX is calibrated to its training distribution.
- LZ6 leaves a +1.5 BPB gap on real C code vs brotli-11. Larger window / BWT needed for long-distance code repetition.
- Markdown is the worst structural gap: avg +1.96 BPB. Cause is frequent regime switching (prose/code/math/markup/lists), not a single missing pattern.
- Largest absolute volume loss: `log_real` (2.3MB → +172KB extra vs brotli), `promessi_sposi` (1.3MB → +164KB extra vs bz2).

**Encode speed comparison:**

SEE encodes at ~20ms/KB. zstd-3 is 100–1000x faster; bz2/lzma are 10–100x faster. Timing is not a goal for SEE V1.0.

---

## Quick reference

```
# General use (default, recommended)
see encode input.txt output.see --weights w.bin --expert-profile general

# Modern prose (articles, docs, correspondence)
see encode article.txt output.see --weights w.bin --expert-profile prose

# Decode (no profile flag needed — read from header)
see decode output.see recovered.txt --weights w.bin

# Audit BPB without encoding
see audit input.txt --weights w.bin --expert-profile general --blend moe
```
