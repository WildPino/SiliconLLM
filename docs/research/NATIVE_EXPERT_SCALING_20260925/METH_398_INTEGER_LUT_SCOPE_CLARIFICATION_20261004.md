# METH-398 output-comparison scope clarification

Post-result clarification, scientific controller/protocol/raw/result retained
unchangedf33015c. The result report's sentence “all full output hash comparisons
require exact bytes” must be read as **exact fingerprint equality**, not an
exhaustive archived ninety-output byte-file comparison.

318 and398 `output_hash()` compute the same64-bit mixed fingerprint over each
matrix result, control result, chosen parent and head score. Raw398 compares
ALL90 such fingerprints exactly319. The fingerprints are not cryptographic
SHA256 hashes of archived raw output files. Hash collisions are possible.
Independent scalar tests compare16960 actual sampled matrix/scaled-output
rows and source controls; algebraic integer bounds support the exact arithmetic
transformation. These tests do not themselves compare every full output byte.

The cost rejection is unchanged: all repetition/fixed-input medians exceed
14ms, pooled stability fails. No changed learned model/native whole quality or
accepted model rate is promoted. Any future claim needing exhaustive whole
output byte identity must archive and cryptographically bind those outputs or
compare them directly under its own prospective protocol. Qualified Switch
whole-model byte/hash evidence393/389 and rates376/391 are separate evidence.
