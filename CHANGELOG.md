# CHANGELOG — Silicon Entropy Engine

## V1.0 (2026-05-25) — Release Seal

**Phase 30: Regression Harness & Release Hardening**
- `SeeArchiveHeader` extended: 60 → 92 bytes; `weights_sha256[32]` field added.
  Encoder stores SHA-256 of the weights file. Decoder verifies at startup and
  rejects archives encoded with different weights (silent corruption eliminated).
- `scripts/regression_test.py`: permanent regression gate. Reads both frozen
  baselines (29A internal, 29C external), runs BPB audit on all corpora/profiles,
  performs SHA-256 roundtrip on 3 corpora. Exit 0 = green.
- `scripts/test_headers.py`: 6 header integrity tests (bad magic, wrong weights,
  truncated header, profile roundtrip ×2, missing weights). All pass.
- `docs/profiles.md`: complete profile reference — when to use `general`,
  when to use `prose`, when neither applies, archive flags bitmask.

**Phase 29C: Catalog Hygiene & Baseline Freeze**
- External corpus catalog corrected: `json_real.json` excluded (was a README
  markdown, not JSON — catalog error from Phase 29B).
- `prose_real.txt` added: Jennie Gerhardt by Theodore Dreiser (1911), 512 KB
  from Project Gutenberg pg19. Key finding: BI dominates on literary prose
  (rich vocabulary, low word-repetition → `prose` profile saves only +0.0027 BPB).
- `data/baselines/phase29c_baseline.json` frozen: 4 external corpora × 4 profiles.

---

## V0.9.3 (2026-05-25) — External Stress Test

**Phase 29B: External Real Corpus Tribunal**
- 4 real files downloaded and tested: log_real.log (Apache, 2.3 MB), c_real.c
  (zlib inflate.c, 51 KB), json_real.json (catalog error — markdown), readme_real.md
  (LZ4 README, 4 KB). Robustness check: PASS on all.
- Key finding: `c_real.c` → BI dominant (not SEE as on synthetic C). Real zlib
  code has stronger bigram structure than synthetic corpus.
- TOKPFX value confirmed externally: +0.1054 BPB mean (non-small corpora).

---

## V0.9.2 (2026-05-25) — Catalog .gitignore

Minor: `.gitignore` updated (`_transport/` added).

---

## V0.9.1 (2026-05-25) — Profile Rename + Regression Baseline

**Phase 29A follow-up**
- Expert profile `text` renamed to `prose` (canonical). `text` retained as
  backward-compat alias. Help text updated.
- `data/baselines/phase29a_baseline.json` created: 9 corpora × 4 profiles,
  tolerance 0.005 BPB. Authoritative internal baseline.
- `scripts/gen_baseline_29a.py`: regenerates the baseline from scratch.
- `scripts/phase29b_external.py`: scaffold for external corpus tribunal.

---

## V0.9 (2026-05-25) — Profile Matrix & Corpus Robustness

**Phase 29A: Profile Matrix**
- 4 profiles × 9 corpora matrix (4 historical + 5 synthetic).
- `general`: ROBUST — beats unigram on all 9 corpora.
- `text`/`prose`: VALID for modern prose (-0.045 BPB on natural_text); max
  degradation elsewhere +0.010 BPB. Not useful on Italian literary prose (BI
  dominates Promessi Sposi-style text).
- `span-pfx`: REJECTED (confirmed, stays experimental).
- TOKPFX average value: +0.108 BPB across 9 corpora.
- Domain map established: LZ→JSON/headers, BI→markdown/Italian literary,
  SEE→C code, TOKPFX→prose/logs, UNI→shuffled/random.

---

## V0.8 (2026-05-25) — SPANPFX Tribunal

**Phase 28: Line/Block Expert Tribunal**
- BOL (beginning-of-line) expert: rejected. W_BOL vanishes when competing with
  LZ; volume too low to justify.
- SPANPFX (backtick/dollar span delimiter expert): rejected. W_SPAN = 0.0145 on
  markdown — dictionary is too sparse. No improvement on any domain.
- Gap markdown +1.33 BPB accepted as current architecture limit.
- Final profile definitions locked:
  - `general` = LZ6 + TOKPFX
  - `text`/`prose` = LZ6 + TOKPFX + TOK_PREV_ELIG

---

## V0.7 (2026-05-25) — Gated MoE + Expert Profiles

**Phase 27F: Gated MoE Validation**
- SHA-256 bug fixed: `SEE_FLAG_TOK_PREV_ELIG` was missing from archive_flags
  on encode, causing decoder mismatch. Fixed.
- W_when_elig = 0.8030: TOK_PREV expert weight when eligible, confirming gated
  architecture (expert is allowed to stay silent on ineligible bytes).
- natural_text gain: -0.0445 BPB with TOK_PREV_ELIG active.
- Gated architecture validated: "il diritto al silenzio è architetturale."

---

## V0.6 (2026-05-25) — Token-LZ Expert

**Phase 27: Token-LZ Expert (TOKPFX)**
- TOKPFX expert promoted: uses token-prefix hash as LZ key (5-byte prefix of
  the current token's UTF-8 encoding).
- natural_text: -0.47 BPB (95% MoE weight on prose).
- markdown: -0.13 BPB improvement.
- shuffled: correctly silent (0.000 BPB gain — expert learns to abstain).

**Phase 27C: Cartography**
- ALNUM_START identified as the natural_text gap: 0.696 BPB over UNI,
  22.9% of bytes. Markdown has MoE convergence lag of +1.3 BPB across all
  positions (uniform — not a local gap, a global warm-up cost).
- LaTeX floor confirmed: ~4.5 BPB regardless of expert.

---

## V0.5 and earlier — LZ Top-N + MoE Foundation

**Phase 26: LZ Tribunal**
- LZ6 (6-byte key) promoted as default. LZ8 and dual-LZ archived.
- Gap markdown confirmed as non-LZ (LZ does not help markdown structure).

**Phase 24 and earlier — Architecture Foundation**
- Hybrid stateful architecture: silicon_entropy (wave ESN) + range coder +
  bigram + unigram + LZ Top-N expert.
- MoE credit assignment (fixed-share exponentiated gradient).
- LZ Top-N: hash table with N candidate slots per key; LZ builds per-symbol
  probability from slot frequency counts.
- SEE3 archive format with self-describing header.
