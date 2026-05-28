# Silicon Entropy Engine — Project Overview

> A streaming Mixture-of-Experts entropy coder for CPU-native lossless compression.
> Status: **Phase 29A in progress** · Build: passing · Version: 0.9-dev

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Expert Profiles](#expert-profiles)
- [Benchmark Results](#benchmark-results)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Changelog](#changelog)
- [Contributing](#contributing)

---

## Introduction

The Silicon Entropy Engine (SEE) is a research-grade lossless data compressor
that routes each byte through a dynamic mixture of specialized prediction experts.
Unlike block-based compressors (zstd, brotli, lz4), SEE operates in a single
streaming pass with no look-ahead, making it suitable for real-time encoding
pipelines.

### Core idea

Each byte $x_t$ is predicted by $N$ experts. A softmax-weighted mixture
$P_{mix} = \sum_i w_i P_i$ produces the final probability. The weights $w_i$
are updated online via exponential loss:

$$w_i \leftarrow w_i \cdot \exp(-\eta \cdot \ell_i)$$

where $\ell_i = -\log_2 P_i(x_t)$ is the coding cost of expert $i$ on the
observed byte.

### Why streaming?

- **Zero latency**: the encoder never buffers beyond a single byte.
- **Adaptive**: weights converge to the right expert within a few hundred bytes.
- **Composable**: new experts can be added without retraining existing ones.
- **Transparent**: the weight vector is a live readout of domain structure.

---

## Quick Start

### Prerequisites

```
gcc >= 12.0 or clang >= 15.0
Windows: MSVC 2022 or mingw64
Python >= 3.9  (for benchmark scripts)
make (optional, build.bat on Windows)
```

### Build

```bash
# Linux / macOS
make

# Windows
build.bat

# Manual (single TU)
gcc -O3 -march=native -o see see.c src/see_codec.c -lm
```

### Encode a file

```bash
./see encode input.txt output.see --weights weights/entropy_weights_factors_r16.bin
./see decode output.see restored.txt --weights weights/entropy_weights_factors_r16.bin
```

### Run an audit (BPB + weight diagnostics)

```bash
./see audit input.txt --weights weights/entropy_weights_factors_r16.bin \
    --blend moe --expert-profile general
```

Expected output for a natural-language file:

```
=== SEE Audit: input.txt ===
Bytes evaluated: 218553
Unigram BPB:     3.9827
Model BPB:       2.4656
Quantized BPB:   2.4741
Cycles/byte:     58751.9
LZ key width:    6 bytes + tok-prefix expert
--- Per-expert BPB ---
SEE Only:        5.1674
UNI Only:        3.9949
BI  Only:        2.9462
LZ  Only:        4.2525  (key=6 bytes)
TOKPFX Only:     2.6219
--- MoE weights ---
Avg  [SEE UNI BI  LZ  TOKPFX]: 0.0031 0.0126 0.0282 0.0054 0.9507
Final[SEE UNI BI  LZ  TOKPFX]: 0.0021 0.0041 0.0260 0.0084 0.9594
```

---

## Architecture

### Expert taxonomy

| Expert   | Type        | Key signal                    | Default profile |
|----------|-------------|-------------------------------|-----------------|
| SEE      | Neural/static | Pre-trained byte distribution | all             |
| UNI      | Unigram     | Per-byte frequency            | all             |
| BI       | Bigram      | Byte-pair frequency           | all             |
| LZ       | LZ-hash     | Repeated substrings (key=6B)  | general, text   |
| TOKPFX   | Token-prefix| Inside-token byte prediction  | general, text   |
| TOKPREV  | Token-trans | After-token transition (elig.)| text            |
| SPAN     | Span-prefix | Backtick/dollar spans (exp.)  | experimental    |

### MoE update rule

The fixed-share update prevents expert weight collapse:

```c
// Normalize + fixed-share
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
    w[i] *= expf(-eta * loss[i]);
    sum  += w[i];
}
for (int i = 0; i < n; ++i) {
    w[i] = (1.0f - share) * (w[i] / sum) + share / (float)n;
}
```

### Range coder

SEE uses a standard carry-less range coder (Schindler 1998) with 24-bit top,
16-bit bottom, and a carry-byte buffer. The codec is symmetric: encode and
decode share the same probability tables and update steps.

```
RC state: { low: u64, range: u32, code: u32 }
Encode: low += range * cum_freq / total
        range = range * freq / total
Renormalize when range < RC_BOT (2^16)
```

---

## Expert Profiles

### `--expert-profile general` (default)

Best for: all domains. Recommended for unknown input.

Experts active: SEE + UNI + BI + LZ6 + TOKPFX

```
Corpus            BPB
──────────────────────
natural_text     2.474
markdown_docs    3.887
c_code           1.940
shuffled         5.012
```

### `--expert-profile text`

Best for: natural language prose (Italian, English, French…).
Adds: `TOK_PREV_ELIG` — gated token-transition expert, fires only when
the previous byte ended an alphanumeric/macro token.

```
Corpus            BPB    Δ vs general
──────────────────────────────────────
natural_text     2.429   -0.045  ✓
markdown_docs    3.890   +0.003  (MoE silences TOKPREV on MD)
c_code           1.942   +0.002  (same)
shuffled         5.012   +0.000
```

> **Rule**: use `text` profile only when input is confirmed prose.
> For mixed or unknown content, prefer `general`.

### `--span-pfx` (experimental)

Adds: span-prefix expert, event-driven (fires only inside backtick/dollar
spans). Phase 28B tribunal result: **rejected** for markdown gain
(+0.0002 BPB vs criterion 0.025). Remains available as a flag for
future investigation.

---

## Benchmark Results

Phase 29A robustness matrix (in progress):

| Corpus              | general | text  | Δ text  | no-token | Δ no-tok |
|---------------------|---------|-------|---------|----------|----------|
| natural_text.txt    | 2.474   | 2.429 | -0.045  | TBD      | TBD      |
| markdown_docs.md    | 3.887   | 3.890 | +0.003  | TBD      | TBD      |
| c_code.c            | 1.940   | 1.942 | +0.002  | TBD      | TBD      |
| shuffled.bin        | 5.012   | 5.012 | 0.000   | TBD      | TBD      |
| json_synth.json     | TBD     | TBD   | TBD     | TBD      | TBD      |
| c_header_synth.h    | TBD     | TBD   | TBD     | TBD      | TBD      |
| project_notes_it    | TBD     | TBD   | TBD     | TBD      | TBD      |
| repo_markdown_mixed | TBD     | TBD   | TBD     | TBD      | TBD      |
| log_synth.log       | TBD     | TBD   | TBD     | TBD      | TBD      |

_Table will be filled in by phase29a_tribunal.py output._

---

## API Reference

### `see_encode`

```c
see_error_t see_encode(
    see_codec_ctx_t  *ctx,
    const uint8_t    *src,
    size_t            src_len,
    uint8_t          *dst,
    size_t           *dst_len   /* in: capacity, out: bytes written */
);
```

Encodes `src_len` bytes from `src` into the pre-allocated buffer `dst`.
`dst_len` must be at least `src_len + SEE_ARCHIVE_HDR_SIZE + 256` on entry.
Returns `SEE_OK` on success, `SEE_ERR_OVERFLOW` if output buffer is too small.

### `see_decode`

```c
see_error_t see_decode(
    see_codec_ctx_t  *ctx,
    const uint8_t    *src,
    size_t            src_len,
    uint8_t          *dst,
    size_t            dst_len   /* must match original size from header */
);
```

Decodes a SEE3 archive. The original size is embedded in the archive header;
`dst_len` must match exactly or `SEE_ERR_MISMATCH` is returned.

### `see_audit`

```c
see_error_t see_audit(
    see_codec_ctx_t  *ctx,
    const uint8_t    *data,
    size_t            len,
    see_weights_t    *out_weights  /* optional, may be NULL */
);
```

Performs a dry-run encode (no output file), collecting BPB statistics and
MoE weight trajectories into `out_weights`.

---

## Configuration

### CLI options

```
--blend moe|<lambda>       Blend mode. moe = dynamic, float = fixed mix.
--eta   <float>            MoE learning rate (default 0.03).
--share <float>            Fixed-share coefficient (default 0.001).
--topk  <int>              SEE top-k candidates (default 256).
--speed full|accurate|fast Quantizer speed/accuracy tradeoff.
--expert-profile <name>    Expert set: general | text | experimental.
--lz-key <4|6|8>           LZ context key width in bytes.
--tok-prefix               Enable TOKPFX expert manually.
--tok-prev-elig            Enable gated TOK_PREV expert manually.
--span-pfx                 Enable experimental span-prefix expert.
--no-lz                    Disable LZ (3-expert mode: SEE+UNI+BI).
```

### Environment variables

| Variable              | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `SEE_WEIGHTS_PATH`    | (none)  | Default weights file path          |
| `SEE_MAX_THREADS`     | 1       | Number of worker threads (future)  |
| `SEE_LOG_LEVEL`       | INFO    | Log verbosity (DEBUG/INFO/WARN)    |
| `SEE_TELEMETRY_DIR`   | (none)  | Directory for per-byte CSV output  |

---

## Changelog

### v0.8 — Phase 28 (2026-05-25)

- **Added** `--span-pfx` expert (event-driven, backtick/dollar gated)
- **Added** `--span-pfx-mute` ablation flag
- **Tribunal result**: SPANPFX rejected — markdown gain +0.0002 BPB
  (criterion 0.025). Flag remains available as experimental.
- **Fixed** SHA-256 symmetry bug: `SEE_FLAG_TOK_PREV_ELIG` was missing
  from archive header encoding, causing decoder state mismatch.

### v0.7 — Phase 27F (2026-05-25)

- **Added** `--expert-profile text` with `TOK_PREV_ELIG` gated expert
- **Validated** eligibility semantics: W_when_elig = 0.8030,
  natural_text gain −0.045 BPB, shuffled correctly muted
- **Fixed** SHA-256 header bug (same session as v0.8 hotfix)

### v0.6 — Phase 27C/27E (2026-05-24)

- **Cartography**: ALNUM_START is the natural_text gap (0.696 BPB, 22.9%)
- **Cartography**: markdown gap = MoE convergence lag +1.3 BPB uniform
- **Added** `--tok-prev-elig` manual flag

### v0.5 — Phase 24 (2026-05-20)

- **Sealed** hybrid architecture: LZ6 + TOKPFX as stable base
- **Removed** experimental LSTM and PPM paths
- **Added** `--expert-profile` CLI flag replacing ad-hoc combinations

### v0.4 — Phase 23 (2026-05-18)

- **Added** compact LZ implementation with micro-audit harness
- **Benchmarked**: LZ6 +0.5 BPB over bigram on c_code domain

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b phase-NN-description`.
3. Run the full tribunal before opening a PR:

```bash
python scripts/phase29a_tribunal.py
```

4. Ensure no historical domain regresses beyond **0.005 BPB**.
5. Update `docs/phases/` with a walkthrough of your changes.
6. Open a PR with the tribunal output attached.

### Code style

- C11, no VLAs, no alloca.
- All functions return `see_error_t`; no bare `void` that can fail.
- Every header is self-contained (no implicit includes).
- No `goto` except cleanup paths.
- Comment only non-obvious invariants.

### Commit format

```
Silicon Entropy Engine VX.Y — Phase NN: short description

Body: what changed and why. Reference tribunal results if relevant.
```

---

_Last updated: 2026-05-25 — Phase 29A corpus generation_

---

## Appendix A: Mathematical Background

### Arithmetic coding fundamentals

Given a probability model $P$ over the alphabet $\{0, \ldots, 255\}$, the
arithmetic coder assigns to a sequence $x_1, x_2, \ldots, x_n$ a code of
length $\lceil -\log_2 \prod_t P(x_t | x_{<t}) \rceil + 1$ bits. The
per-symbol average is the cross-entropy:

$$H_P(X) = -\frac{1}{n} \sum_{t=1}^n \log_2 P(x_t | x_{<t}) \quad [\text{bits per byte, BPB}]$$

The true entropy of the source is a lower bound: $H_P(X) \geq H(X)$, with
equality iff $P$ matches the true distribution.

### Mixture of experts bound

If the true distribution is a mixture $P^* = \sum_i \alpha_i P_i$ with
$\alpha_i > 0$, then the online MoE achieves:

$$H_{\text{MoE}}(X) \leq H_{P^*}(X) + \frac{\ln(1/\alpha_{\min})}{n} + O(1/n)$$

The second term is the *price of ignorance* about the mixing coefficients.
With $n = 100{,}000$ bytes and $\alpha_{\min} = 0.001$, this is $0.0001$
BPB — negligible. This is why online MoE is competitive with optimal
offline mixture at corpus scale.

### Fixed-share and the sleeping expert regret

The standard Hedge algorithm (Freund & Schapire 1997) minimizes regret
against the best single expert. Fixed-share (Herbster & Warmuth 1998)
minimizes regret against the best *sequence* of experts with at most $s$
switches. The regret bound is:

$$R_T \leq \sqrt{\frac{T \ln N}{2}} + O(s \ln T)$$

For compression, this means SEE adapts to domain switches within a file
(e.g., mixed code and prose) with bounded overhead.

---

## Appendix B: LZ Hashing Details

The LZ expert uses a rolling hash over the last $k$ bytes (default $k=6$):

```c
static inline uint32_t lz_hash(const uint8_t *key, uint32_t k) {
    uint32_t h = 2166136261u;  // FNV-1a offset basis
    for (uint32_t i = 0; i < k; ++i) {
        h ^= key[i];
        h *= 16777619u;        // FNV prime
    }
    return h;
}
```

The table stores the *predicted next byte distribution*: an array of 256
counts, incremented on each observation. At prediction time, the counts
are normalised to a probability vector. The table has $2^{20}$ slots
(~4 MB for 32-bit counts).

Collision handling: open addressing with linear probing. Eviction uses
a frequency-weighted LRU approximation — entries with low `freq` field
are overwritten by new entries that hash to the same slot.

Key width selection:
- $k=4$: fast hashing, high collision rate, useful only for very small files
- $k=6$: default; good balance of coverage and collision rate
- $k=8$: best for highly repetitive structured data (binary logs, compressed
  inner layers); higher memory footprint and slower hash

---

## Appendix C: Token-Prefix Expert Design

The token-prefix (TOKPFX) expert exploits a regularity in natural text and
source code: within a token (a contiguous run of alphanumeric characters or
underscore), the continuation probability is very different from the
transition probability.

**Example**: given the prefix `uint8`, the next byte is highly likely to be
`_` (completing `uint8_t`) or a digit/letter (completing a longer token) —
not a space or newline. This is a sharp conditional distribution that the
bigram expert struggles to capture because it only sees one byte of context.

The TOKPFX key is the current token prefix (up to 4 bytes), hashed to a
$2^{18}$-slot table. A separate table tracks whether the byte ended a token,
which is used by the downstream TOK_PREV_ELIG expert.

Learned weights on natural text:

```
Avg  [SEE UNI BI  LZ  TOKPFX]: 0.003 0.013 0.028 0.005 0.951
```

TOKPFX carries 95% of the predictive weight. The other experts contribute
a small regularizing prior that prevents pathological behaviour on
out-of-distribution bytes.

---

## Appendix D: Evaluation Methodology

### Why BPB and not compression ratio?

BPB (bits per byte) measures *information-theoretic efficiency* independently
of file size. Compression ratio conflates codec overhead (headers, checksums)
with actual prediction quality. For audit purposes, we always compare
quantized BPB because it includes the range-coder quantization step.

### Tribunal protocol

Each phase introduces a candidate expert. The tribunal runs three
configurations:

1. **BASE** — current best profile (no new expert)
2. **MUTE** — new expert added but weight is frozen at uniform (measures
   the MoE tax of adding a new arm)
3. **ACTIVE** — new expert with full online learning

Promotion criteria (from Phase 27 onwards):
- Primary gain ≥ 0.025 BPB on target domain
- Regression ≤ 0.005 BPB on all other domains
- Shuffled (random) domain: weight stays at floor (MoE ignores it)
- SHA-256 encode/decode symmetry: PASS on all test files

### Corpus selection

The five historical corpora cover the main structural categories:

| File              | Domain            | Key structure           |
|-------------------|-------------------|-------------------------|
| natural_text.txt  | English prose     | Word-level patterns     |
| markdown_docs.md  | Markdown tech doc | Mixed prose + code      |
| c_code.c          | C source          | Symbol-heavy, low entropy|
| shuffled.bin      | Random permutation| Zero structure (oracle) |
| promessi_sposi.txt| Italian literary  | Long-range coherence    |

Phase 29A adds five new corpora to test generalization beyond the
training distribution.


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.


<!-- filler -->


---

## Frequently Asked Questions


### Q1. What is BPB?

BPB stands for Bits Per Byte. It measures how many bits the coder uses on average to represent each byte of the input. Lower is better. The theoretical minimum is the Shannon entropy of the source.


### Q2. Why not use zstd or brotli instead?

zstd and brotli are excellent general-purpose compressors optimized for throughput and compression ratio on known workloads. SEE is a research platform for studying online mixture-of-experts prediction. The goal is understanding *why* certain byte patterns are predictable, not shipping the smallest archive.


### Q3. What does 'streaming' mean in this context?

Streaming means the encoder processes one byte at a time without buffering future input. At each step, the current probability model depends only on bytes seen so far. This constrains the design: no block-level statistics, no two-pass preprocessing, no dictionary pre-building from the full input.


### Q4. Is SEE lossless?

Yes. The range coder is exact: decode(encode(x)) = x for all inputs. SHA-256 symmetry is verified after every tribunal as a non-negotiable correctness criterion.


### Q5. Can SEE compress random data?

No. Random data has entropy equal to 8 BPB (one bit per bit). SEE achieves 5.01 BPB on shuffled.bin — above 5, worse than uncompressed in practice due to header overhead. This is expected and correct: the MoE assigns near-uniform weights to all experts, and the quantization overhead adds ~0.01 BPB.


### Q6. What is the 'shuffled' corpus?

shuffled.bin is a byte-level permutation of c_code.c. The marginal distribution is identical to c_code (same byte frequencies), but all sequential correlations are destroyed. It serves as an oracle: any gain a sequential model claims on shuffled.bin is a false positive.


### Q7. Why does TOKPFX help so much on natural text?

English and Italian prose consists largely of known words from a limited vocabulary. Within a word (a token), the next character is highly predictable from the current prefix: given 'uniq', 'u' is almost certainly next ('unique', 'uniqueness'). The TOKPFX table stores a separate 256-slot probability vector for each observed within-token prefix, updated online. On natural_text.txt, it captures ~95% of the total predictive weight.


### Q8. Why did SPANPFX fail on markdown?

The content inside backtick and dollar spans in technical markdown is extremely varied: _mm256i, uint8_t, \\mathbb{R}^n, sin(x), my_func(), etc. A prefix-hash LZ dictionary needs to see the same prefix multiple times to build a useful distribution. On a 126KB markdown file, each distinct span prefix appears fewer than 3 times on average. The table stays sparse and the MoE correctly assigns it near-zero weight (W_SPAN = 0.0145 when eligible).


### Q9. What is the fixed-share coefficient?

The fixed-share coefficient (default 0.001) redistributes a small fraction of the total weight uniformly among all experts after each update. Without it, a losing expert's weight decays exponentially and can reach numerical zero, preventing recovery. With share=0.001, even a fully silenced expert retains 0.1% / N weight at minimum.


### Q10. How is 'dominant expert' defined?

The dominant expert is the one with the highest average weight over the entire audit run. Average weight is more informative than final weight because final weight reflects only the last few bytes. On natural text, TOKPFX dominates with avg~0.95. On c_code, TOKPFX still dominates but with a lower avg~0.87 due to more non-token bytes.


### Q11. Does SEE support parallel encoding?

Not currently. The streaming MoE state (weight vector, LZ table, token context) is sequential by design. Parallel encoding would require either blocking (fixed-size chunks with independent state) or a checkpoint protocol to synchronize weight vectors across threads.


### Q12. What is the range coder's symbol precision?

SEE uses 12-bit probability quantization: probabilities are rounded to the nearest multiple of 1/4096. The quantization loss is approximately 0.001-0.003 BPB depending on the sharpness of the distribution (sharper = more loss from rounding peaks). This is the gap between 'Model BPB' and 'Quantized BPB' in audit output.


### Q13. What happened to the PPM and LSTM experiments?

Both were tested in Phases 18-21 and discarded. PPM order-5 performed well on c_code but consumed O(|alphabet|^5) memory and was impractical. LSTM required offline training and could not adapt online without catastrophic forgetting. The MoE approach gives similar gains with no training and O(1) memory per expert.


### Q14. Why is cycles/byte so high compared to modern compressors?

SEE prioritizes compression ratio over speed. The MoE update involves floating-point exponentials for each expert on each byte. A typical run costs ~60,000 cycles/byte vs ~50 cycles/byte for LZ4. This is a research tool, not a production codec. A hardware-accelerated implementation could reduce this by 100-1000x.


### Q15. What does 'MoE tax' mean?

Adding a new expert to the mixture has a cost even if the expert carries no useful signal: the softmax normalization slightly dilutes the weights of existing experts, and the fixed-share mechanism permanently steals a tiny fraction of weight. Phase 28B measured this: adding a fully muted SPAN expert costs +0.0000 BPB on most domains, confirming the tax is negligible at 5-expert scale.



---

## Phase-by-Phase History


### Phase 18: Wave Engine

Replaced boolean bitwise CA with AVX2 saturating arithmetic. Established ESN (Echo State Network) paradigm: fixed reservoir, trained readout. Key result: temporal damping (`>> 1`) is necessary for echo state property.


### Phase 19: Reservoir Readout

Added M4 circular buffer as direct readout input. LMS training converged to 100% accuracy on Echo-5 with M4 channels. Confirmed that the wave is a nonlinear mixer; M4 provides the linear component.


### Phase 20: Multi-Channel Readout

Extracted K=32 channels via regional summation. Demonstrated that spatial gradients are preserved through summation. First working end-to-end encode/decode pipeline.


### Phase 21: Online Expert Mixing

Introduced 3-expert MoE (SEE, UNI, BI) with fixed-share Hedge update. First system to beat any fixed lambda on all domains simultaneously. SHA-256 roundtrip verified.


### Phase 22: MoE Validation

Full cross-domain audit. MoE outperforms every fixed lambda on every dataset. Natural language compression: 2.47 BPB. Code: 1.94 BPB. Shuffled: 5.01 BPB (correct behavior confirmed).


### Phase 23: LZ Expert

Added LZ6 expert (FNV-1a hash, 1M-slot table). BPB improvements: c_code -0.5, markdown -0.3, natural_text -0.1. Introduced micro-audit harness for per-expert attribution.


### Phase 24: Consolidation

Sealed hybrid architecture: LZ6 + base MoE. Removed LSTM and PPM experiments. Established --expert-profile CLI. Defined 'default operational' vs 'experimental' tier.


### Phase 25: Benchmark

Cross-platform benchmark: Linux / Windows / MSVC / GCC. Confirmed cycles/byte consistency within 5% across compilers.


### Phase 26: LZ Tribunal

LZ6 promoted to default. LZ8 and LZ-dual archived. First use of tribunal protocol: BASE / MUTE / ACTIVE three-config pattern.


### Phase 27: Token-LZ Expert

TOKPFX expert introduced. natural_text gain -0.47 BPB. MoE assigns 95% weight to TOKPFX on prose. Shuffled correctly muted.


### Phase 27C: Cartography

ALNUM_START is the natural_text gap (0.696 BPB, 22.9% bytes). Markdown gap confirmed as uniform MoE convergence lag +1.3 BPB, not byte-class specific.


### Phase 27F: Eligibility Audit

TOK_PREV_ELIG promoted to `text` profile. SHA-256 bug found and fixed (missing flag in archive header). natural_text gain: -0.045 BPB. W_when_elig = 0.8030.


### Phase 28: SPAN Tribunal

BOL rejected (volume). SPANPFX rejected (dictionary too sparse). Markdown gap accepted as structural limit of current streaming approach on 126KB corpus. Key learning: the gap is uniform across the file, not concentrated in delimiters.


### Phase 29A: Robustness Matrix

Current phase. 5 new synthetic corpora. 4-profile matrix. TOKPFX value measurement via no-token baseline. Objective: confirm general/text profiles generalize beyond historical 4 files.
