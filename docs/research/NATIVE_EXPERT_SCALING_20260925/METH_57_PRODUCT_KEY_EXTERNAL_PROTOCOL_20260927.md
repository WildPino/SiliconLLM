# METH-57: frozen external audit of the E128 product-key checkpoint

**Uncertainty and decision.** METH-56 retains token-level donor
agreement through update 512, but METH-47's similar automatic gates
missed unsupported claims. Test whether the fixed METH-56 update-512
checkpoint conserves source-grounded behavior and task quality on
unseen documents. A failed external or semantic gate stops promotion
of this checkpoint; a pass only permits native export work at E128,
not a large-E quality claim.

Bind METH-56 result SHA-256
`06aa8934d849fc3bbe2ae2fb85c5ab4d302c9f3f1c5922138604bf1347273a39`
and its update-512 checkpoint SHA-256
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`.
Use Qwen2.5-0.5B-Instruct at its pinned revision and tokenizer.
Select 8 code, 8 prose and 8 technical/general source items from
METH-41's source-verified selector with seed `meth57-external-57057`,
excluding every METH-41 and METH-45 source ID and text overlap.
Freeze the 24 document IDs, 384-character excerpt, chat-format prompt,
source provenance and hashes before inference. The prompt asks for a
two-sentence summary and one specific detail, treating the excerpt
as data. Inspect the frozen excerpts for answerability **before**
running either model; if one is unusable, replace the manifest under
a new protocol and seed, not after viewing continuations.

Automatic paired audit: document BPB by category, prompt-position
top-1, greedy 128-token chat continuations, EOS/loop diagnostics,
full 1,838-item PIQA and route/expert-pair utility. Pass bounds are
pooled document ΔBPB≤+0.02 and each category≤+0.04; pooled prompt
top-1≥95% and each category≥90%; student EOS count at least donor−2,
and no category more than donor+1 in repeated 8-gram or short
non-EOS outputs; PIQA accuracy delta≥−2 percentage points with
paired bootstrap 5th percentile≥−5 points. Pair-utility diagnostic
permutes trained expert A/B factors against fixed product keys using
a fixed seed; require BPB penalty≥+0.002. Require ≥64 changed B
slots/layer. A route-oracle check must recover identical top-four
pair IDs from 16 candidates and an exhaustive E128 pair matrix on
every external prompt position.

Semantic audit: create an arm-blinded review file after generation,
with donor/student order chosen by a fixed hash of source ID. Before
revealing labels, read each 384-character prompt excerpt and both
responses. Count (a) unsupported factual claims about the excerpt,
(b) severe unsupported named-entity, numerical, comparative or
causal claims, and (c) missing requested specific detail. Quote a
minimal supporting or contradicting span for each finding. Treat
claims requiring outside information as unsupported because the
prompt explicitly constrains the response to the excerpt. Record
ambiguous cases separately and do not count them as unsupported.
Freeze the blind verdicts and their hash before unblinding. Pass only
if the student has no more unsupported, severe unsupported or missing
detail findings than the donor; report raw counts for both arms.
The blind reviewer is one agent and its judgments are inspectable,
not an independent human panel or a population error estimate.

Run on the local RTX 3060 only, ≤10.5 GiB peak allocated GPU,
≤20 GiB RSS and ≤30 minutes for automated scoring. Stop on a budget
overrun and preserve partial records. No T4. Automatic and semantic
gates must both pass before this E128 artifact is considered
quality-valid for native parity and rate work.

## Pre-inference source-pool amendment

The first manifest attempt stopped before writing a manifest: only
three eligible prose IDs remain in METH-41's two 32-row held-out JSONL
files after the source exclusions (selector reported 64 source rows,
60 prior IDs and one overlap). Keep the frozen 8/8/8 category counts.
For prose, take those three plus five distinct additional PG19 rows
from the local publisher `test` parquet file
`data/external/pg19/data/test-00000-of-00001-29a571947c0b5ccc.parquet`,
SHA-256
`9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc`.
The file has 100 physical rows. Choose the five by ascending SHA-256
of `meth57-pg19-extra-57057|<source ID>` among unused IDs, taking a
deterministic ≤4096-byte UTF-8 span positioned by the same hash.
Require source-ID exclusion and first/middle/final 256-byte fragment
non-overlap against earlier METH-17/19/25/41/45 selections, the
Qwen calibration/held-out text and already chosen METH-57 items.
Record parquet file, physical row ID, full-document SHA-256, span
offset, text/hash and tokenized IDs. If fewer than five survive,
stop and register another source change before inference.
