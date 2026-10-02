# METH-264: original donor cached-reference stop

The frozen cached generation screen stopped on the original BF16 donor,
before any complete continuation or candidate generation. Same-prefix
full/cached choices differ on 2 of 48 checks: code step15 tokens16/12 and
tech step15 tokens5785/5666; prose has no choice mismatch. Maximum logit
absolute differences are .34375/.28125/.359375 for code/prose/tech.
Prefill choices match. This records the source backend's BF16 arithmetic
limit, not a failed transfer-quality measurement.

Raw [result](meth264_complete_core_generation_result.json), SHA256
`c016596e419476e7033fc95e774b06bc184e54f8f6ceaf08989dcc54e3865029`.
Decision `cached_reference_choices_fail_hold_generation_and_tasks`,
only gate `bf16_donor_cache_choices=false`, empty generation arms.
Session99155 exited0,17.797s after imports, RSS4,229,001,216 bytes,
peak GPU3,081,579,520 bytes. No job remains active.

Keep the zero-mismatch guard and preserve this failed cached recipe.
Do not change prompts or relax it. METH-265 explicitly changes evaluation
to recompute the full prefix on every step, with the same frozen model,
prompts, head, EOS/cap and health criteria. It is a slower reference
semantic evaluation; it supplies no production-cache or CPU-rate proof.
Native decoding will need its own arithmetic/quality and accepted-rate
checks on this same archive.
