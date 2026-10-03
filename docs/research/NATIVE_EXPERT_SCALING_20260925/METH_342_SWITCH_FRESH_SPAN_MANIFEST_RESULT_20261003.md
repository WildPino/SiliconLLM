# METH-342: new source-only span-reconstruction cohort frozen

Freezef6ac7d5; rawd2762b9, authoritative exec34641 exit0.
Raw SHA `bed91d901b26caf9b23992f1b4198d692e0ca770eeea60571caae9e2c884f89f`.
MAIN46.329s after imports/endRSS1,504,366,592B. All three source-only gates PASS.
No model loaded/scored, original/native outputs not used for source selection.

1243 original cached PG19train14 book rows, unchanged308,888,290B parquet.
All tracked research/donor JSON hashes recorded;36 book rows excluded using
corpus source IDs and conservative source_row references.24 hash-ranked new
book rows:855,845,776,333,316,101,800,1238,102,364,384,335,774,1194,475,689,
373,445,697,210,234,727,139,8.24 distinct whole original source hashes.
No row overlap with recorded old corpus sources. This is not an unknown
original-pretraining exclusion or certification of undocumented source use.

Each new book: four distinct63-token windows from fixed4096-character excerpt,
one8-token mask per window, original local T5/Switch tokenizer. Encoder57
tokens, known target11 (sentinel0+original8+sentinel1+EOS), forced input11.
ALL96 source/labels/IDs/full raw excerpt/hash recorded before scoring.
Tokenizer emits a >512 warning for the4096-char source-only excerpt (e.g.1153
tokens); excerpt is never passed to model. Actual encoder inputs all57 tokens
within fixed context bounds; no truncation/long-model-inference claim.

Next separately frozen343 original PRIMARY donor/native341 paired prediction
and teacher-forced known-answer reconstruction, book-level uncertainty, then
free generation/task/rate if prediction qualified. No instruction/chat, useful
larger-n/LUT/physical DRAM/quality or accepted-rate conclusion from manifest.
Commands/criteria in [342 protocol](METH_342_SWITCH_FRESH_SPAN_MANIFEST_PROTOCOL_20261003.md).
