# C50-A5 part 5 protocol: a head restricted to the vocabulary actually used, confirmed on an unread set

17 October 2026. This is the owner's option A after the A5 FAIL (COMMUNICATION, decided today):
keep Ling-mini, and try new head routes on FIT under a registered protocol, with a confirmation on
a set not yet read. Pushed before any statistic of this part is computed.

## Idea

- Part 4 showed that dropping rows by script makes the scan exact wherever the dropped rows
  are never wanted. But a script rule keeps 99k rows (166.6 MB).
- Most of the 157k rows are rare. Here the head keeps only the rows that real text uses most.
  - The "use" is measured on a corpus that no evaluation reads.
  - The scan over the kept rows is exact, so no preview is involved.
  - The approximation fails only where the exact top-1 is a row outside the set.
- **Gate (unchanged from A5):** teacher-forced top-1 agreement with the exact head ≥ 0.999.
  BPB and PIQA keep the exact head (A5 rule).

## Data (all new, from the pinned holdout machinery)

`c50_holdout_ext.py` reruns the `c50_holdout_v1` selection with larger counts: same seed, salt,
exclusions, span rule and NFC skip. The selection key is a salted hash in rank order, so the
first items are unchanged.
- **Proof check, before anything is used:** the tool asserts that the regenerated FIT ranks
  1-48 and DEV ranks 1-32 have exactly the `item_sha256` of the committed `fit.jsonl` and
  `dev.jsonl`.
- **SEL, the selection corpus:** FIT-role items ranked 49-248 per category, 600 documents. Only
  their source-tokenizer IDs are used, as token counts. No model runs on them.
- **DEV-B, the confirmation set:** DEV-role items ranked 33-44 per category, 36 documents.
  - DEV items 1-12 were read by A5 part 2.
  - Items 13-32 are reserved for the A6 DEV confirmation.
  - DEV-B is read once.

## Kept set and grid

- **Frequency:** for each category, count the target IDs (every token of each SEL document)
  and divide by that category's total. Sum the three normalized counts, so each category
  weighs one third.
- **Kept set K_N:** the N most frequent IDs, ties to the lower ID. It always includes the 256
  byte-level single-byte tokens and the 262 added tokens.
- **Grid:** N ∈ {8,192; 16,384; 24,576; 32,768; 40,960; 49,152; 65,536; 81,920; 98,304}.
- **Bytes:** |K_N| × 1,680 B (Q6_K rows of the exact head), scanned exactly.

## Statistic, selection and confirmation (fixed now)

- **FIT statistic:** on the 79,709 FIT12 positions already scored by the engine
  (`a5/FIT12.pos.bin`, exact top-1), the fraction whose top-1 is in K_N. SEL and FIT12 share
  no document.
- **Selection:** the smallest N whose FIT12 recall is **≥ 0.9995**.
  - The extra 0.0005 above the gate is fixed now. Part 1b's selection cleared 0.999 by 0.0002
    on FIT and then failed DEV.
  - If no N qualifies, part 5 FAILS on FIT and DEV-B is not read.
- **Confirmation:** the engine scores DEV-B once (`c50_engine.exe --score`, P0 GGUF, ~37 min).
  The route PASSES iff the DEV-B top-1 recall of the selected K_N is ≥ 0.999.
  - The scoring runs only after the selection is frozen: K_N IDs and sha256 committed.
  - The verdict script `c50_a5_used_vocab.py --confirm` reads the frozen set and DEV-B and
    prints PASS/FAIL. No other N is evaluated on DEV-B before the verdict line.
- **Not gated, reported after the verdict:**
  - the DEV-B recall of every grid N;
  - recall per category;
  - the most frequent missed tokens.

## Budget arithmetic (desk, not a rate)

P4 on the fastest A6 arm (E4B, 773.6 MB at context 2048 with the exact head) becomes
773.6 − 264.1 + |K_N| × 1.68e-3 MB.

## Registered prediction (scored, not a gate)

- **Selected N:** 32,768-49,152, about 56-83 MB.
- **DEV-B recall of the selected set:** 0.9990-0.9997.
- **Misses:** mostly code identifiers and CJK tokens.

## What this cannot say

- **No generation-quality claim.** A token outside the set can never be emitted, and the
  sequence then continues from a different token. The recall bounds how often that happens
  under teacher forcing only.
- **Fixed text mix.** The kept set reflects the SEL mix (one third code, technical text,
  prose). A different product mix, such as more Italian chat, needs its own SEL.
- **No rate:** the exact restricted scan is priced by bytes.
