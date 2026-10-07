# First token-ID mismatch: repair2 observation retention, before execution

The FIRST actual tokenizer job under metadata repair1 stops on first golden
`default_system_generation`: rendered text equals frozen expected literal,
but canonical/encode/source-Rust/independent-BPE IDs are unequal. Seven other
fixtures and stopping probes were not entered. No stage admission or source
semantics change follows. Retain original worker/log/launcher fault bytes.

Worker20168/create1791378490.7597106 exit1, OS peak253243392B sampled AFTER exit
through actual OS handle. Launcher22052/create1791378486.3305311, elapsed10s,
peak snapshot30261248B. Entire first family within prospective512MiB/120s;
source correctness FAIL remains. First frozen sources7fb1a38, metadata repair
freeze102d3c3. PyTorch absent as intended; that library warning is not the fault.

The assertion occurred before copying the four observed ID arrays into the
failure record. Those first arrays were lost. Cannot identify which of the
four paths disagrees from the old raw. Repair2 ONLY records all four arrays
and HF/source pretokenizer descriptors before the unchanged ALL-case equality
assertion. Keep source loader, BPE mathematics, fixture text/IDs derivation,
criteria and budget unchanged. The FAILED first tokenization must be repeated
once to recover the missing diagnostic vectors; disclose this replay. No
completed fixture/native/LLM response experiment is repeated or retropromoted.

Freeze changed worker/protocol/new binding before this sole diagnostic repair.
Reuse sealed runtime/input trees. Stop unchanged mismatch with the new vectors
retained; next choose the exact correction from source evidence rather than
guessing or relaxing equality. If the local HF loader differs from source JSON,
preserve that incompatibility and require a source-preserving loader variant.
If independent BPE is wrong, correct only that apparatus and reuse retained
first-case HF/Rust observations. Goal remains full compact CHATBOT-to-engine.
