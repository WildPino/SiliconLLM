# METH-62: stored mixed core passes automatic quality but fails blind semantics

**Decision:** reject promotion of this BF16-attention/R8-head+FFN Instruct
core. The precommitted direct document, generation and PIQA gates pass
on new sources, but the frozen blind semantic gate fails. Do not treat
the passing automatic metrics as a quality-valid native bundle. The
older METH-60 ≥95% prompt top-1 proxy also remains failed.

The [protocol](METH_62_MIXED_CORE_DIRECT_QUALITY_PROTOCOL_20260927.md)
fixed the precision map, held-out source selection, thresholds and blind
review process before inference. The [exporter](../../../benchmarks/donor_adaptation/s1/meth62_export_mixed_core.py)
read the METH-59 R8 head/FFN codes and original BF16 attention tensors
into one packed-only safetensors artifact. Its SHA-256 is
`1cebf7bae9f7664ec7ec30a80917ee4f46adc95c458a4c969cd8d41c3684aecb`;
the [export report](meth62_mixed_core_export.json) records exact tensor
readback, 539,955,024 physical bytes and 548,320,256 ideal addressed
core+router+top-four factor bytes. These are neither measured CPU DRAM
traffic nor full-model accepted-token rate.

The [frozen source manifest](meth62_mixed_external_manifest.json) chose
eight code, eight PG19 prose and eight technical/general sources that
exclude prior audit and training material. All 24 excerpts passed the
pre-inference [answerability screen](meth62_answerability_screen.json).
The [evaluator](../../../benchmarks/donor_adaptation/s1/meth62_mixed_external_audit.py)
was committed before the run. Its [raw result](meth62_mixed_external_audit_result.json),
SHA-256 `968e872d814f79eebe9c05d450f9b77ed990ac73689d8b2251b872a97dabef9f`,
contains per-source losses, positions, generated responses and all
1,838 PIQA choices. Recomputing from raw document and task rows gives
the recorded totals.

| Precommitted automatic endpoint | Donor | Mixed student | Result |
|---|---:|---:|---|
| Pooled document BPB, 98,280 bytes | 1.172077 | 1.167678 | Δ−0.004400; pass |
| Code / prose / technical ΔBPB | — | −0.002162 / −0.006130 / −0.004907 | each passes |
| Greedy EOS, 24 prompts | 21 | 23 | pass |
| Repeated 8-gram 3× / short non-EOS | 0 / 0 | 0 / 0 | pass |
| Full PIQA correct, 1,838 items | 1,291 | 1,285 | −0.326 percentage point; paired bootstrap fifth −0.871 point; pass |
| Prompt top-1 agreement, 4,258 positions | — | 3,977 = 93.401% | diagnostic, not a new gate |

After generation, the [blinder](../../../benchmarks/donor_adaptation/s1/meth62_blind_semantic_review.py)
created [A/B responses](meth62_blind_semantic_review.json), SHA-256
`e9f0ebf528363406cdabaa6ce9aba0389b53bd18ce36bc66afce6f419641f3b2`.
The [verdict builder](../../../benchmarks/donor_adaptation/s1/meth62_blind_verdict_build.py)
and [24-row verdict](meth62_blind_semantic_verdict.json), SHA-256
`fa8209408fb6bfb1c0d14ca5275ed98ff4a49c2d22f5efd47163261dce768a76`,
were committed at `baa10c8` before unblinding. Each counted finding
contains a claim and excerpt evidence; ambiguous readings are listed
separately. The subsequent [unblinded score](meth62_semantic_score.json),
SHA-256 `e33756eeab165edb849c4a3088a3b4052183146de1696741cb8b69c9efe75cd6`,
is:

| Blind semantic finding | Donor | Mixed student | Gate |
|---|---:|---:|---|
| Unsupported claims | 37 | 41 | fail |
| Severe unsupported claims | 22 | 22 | pass |
| Missing requested specific detail | 0 | 1 | fail |
| Ambiguous readings, excluded from gates | 6 | 3 | — |

The unsupported counts differ by +2 code, +1 prose and +1 technical;
the missing detail occurs in technical/general. Both arms make many
excerpt-grounding errors. This is one reviewer's judgment over only 24
short, clipped excerpts, so the counts are sensitive to rubric decisions;
the preregistered comparison still fails. No verdict was edited after
unblinding. The automatic run took 632.687 s on local RTX 3060, peaked
at 2.431 GB allocated GPU memory and ended at 3.539 GB RSS.

This experiment neither establishes ≥50 accepted tok/s in
`benchmarks/phase60/engine.c` nor tests quality at larger counts of
distinct learned experts. A new precision/adaptation rule must be
fixed on development material and checked on untouched sources before
this core can be reconsidered. METH-62 responses are now viewed and
cannot be a new promotion set.
