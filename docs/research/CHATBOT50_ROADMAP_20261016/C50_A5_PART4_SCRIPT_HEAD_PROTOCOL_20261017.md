# C50-A5 part 4 (exploratory, FIT only): a head restricted by a fixed script rule

17 October 2026. Pushed before any statistic of it is computed. It follows the
[A5 FAIL on DEV](C50_A5_PART2_DEV_RESULT_20261017.md).

## Status

**This is not a selection, and it decides nothing.**
- A5 has failed. The way forward is the owner's decision (COMMUNICATION item 3).
- This part only measures, on FIT, whether a different kind of reduction is worth
  proposing: option A of that item.
- It reads no DEV or RESERVED data, and it does not define a confirmation set. That
  set is part of the owner's decision.

## Idea

Ling-mini's vocabulary is bilingual Chinese/English (157,184 rows). The intended use is
Italian/English chat and code. Rows whose token decodes to text in scripts that use
never needs can be left out of the decode-time argmax.

The scan over the kept rows is **exact**: no approximation, the same Q6_K rows, the
same Q8_K activation. The only error is a position whose engine top-1 is a dropped
token.

Likelihood scoring (BPB, PIQA) keeps the full head, as the roadmap requires.

## Kept set: rule fixed now, from the tokenizer alone (no corpus, no model output)

The tokens are byte-level BPE (`tokenizer.json`). Each vocabulary string is mapped back
to its bytes through the GPT-2 byte table.

- **Kept:** the 262 added tokens (roles, EOS and other specials).
- **Kept:** a token whose bytes are **not** valid UTF-8, i.e. a fragment of a
  multi-byte character. This is conservative: fragments can belong to any script.
- **Kept:** a valid UTF-8 token whose every code point lies in:
  - U+0000–U+052F (Latin, Latin-1, Latin Extended A/B, IPA, Greek, Cyrillic);
  - U+1E00–U+1FFF (Latin Extended Additional, Greek Extended);
  - U+2000–U+2BFF (punctuation, super/subscripts, currency, letterlike, arrows, math, technical, box drawing, shapes, symbols, dingbats);
  - U+FB00–U+FB06 (Latin ligatures) and U+FE00–U+FE0F (variation selectors);
  - U+1F000–U+1FAFF (emoji and pictographs).
- **Dropped:** every other valid UTF-8 token (CJK, kana, Hangul, Arabic, Hebrew,
  Devanagari, Thai, …), and the 31 padding rows that have no token (157,153 tokens,
  157,184 rows).

## Measured (`c50_a5_script_head.py`, capture `FIT12`, 79,709 positions; numbers only)

- **Set size:** N_keep, dropped rows by class, and the exact restricted scan:
  N_keep × 1,680 B per token.
- **FIT top-1 recall:** the fraction of positions whose engine top-1 is kept, overall
  and per category. Every miss is listed with its token string.
- **Combination:** the part 1b rotated r512 3.5-bit preview restricted to the kept rows,
  with K ∈ {1,024 … 32,768}, then exact rerank. Recall uses the same ground truth, and
  bytes are priced as in part 1b over N_keep rows.
- **Control:** the Q8_K oracle rank code on every 10th position (recall at 1 ≥ 0.999,
  else VOID). This is the same as parts 1b and 2.

## Registered prediction (scored, not a gate)

- **Set size:** N_keep between 60,000 and 100,000 (drops 36-62% of the rows).
- **Recall:** FIT recall ≥ 0.9999. The FIT documents are code, English technical text and
  English prose.
- **Bytes:** the exact restricted scan costs 100-170 MB/token. That is cheaper than the
  exact head, but not cheaper than the failed 94.5 MB preview.
- **The combination is the interesting arm:** it prices below 70 MB at 0.999 recall.

## What this part cannot say

- **No Italian on FIT.** FIT has no Italian text, so Italian coverage is argued from the
  rule (Latin script is kept), not measured. A4's Italian chat panel is the measurement,
  if the owner takes this route.
- **No confirmation and no rate.** It decides nothing; see Status above.
