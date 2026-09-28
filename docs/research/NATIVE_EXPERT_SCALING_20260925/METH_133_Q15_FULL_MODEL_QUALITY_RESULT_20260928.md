# METH-133: Q15 passes automatic full-model checks but fails blind grounding

**Decision: reject Q15 for quality promotion.** The same-footprint
METH-132 Q15 child-B bank passed every frozen automatic METH-133 gate,
but failed the independently frozen arm-blind excerpt-grounding gate.
Its full-model status is therefore **not quality-valid**. Do not
integrate this Q15 bank into a target artifact or cite METH-132's
component speed as quality-preserving full-model performance.

The [protocol](METH_133_Q15_FULL_MODEL_QUALITY_PROTOCOL_20260928.md)
was committed at `df9f55e` before source selection. The deterministic
[manifest](meth133_q15_external_manifest.json), 24/24
[answerability screen](meth133_answerability_screen.json) and
[evaluator](../../../benchmarks/donor_adaptation/s1/meth133_q15_full_model_audit.py)
were committed at `0d3257d` before inference. The 24 sources are eight
code files, eight distinct PG19 train-shard-5 books and eight technical
files; their source IDs and excerpt fragments exclude all earlier audit
manifests through METH-131. They remain within previously used broad
Git and PG19 domains. The Q15 bank SHA-256 is
`9e03941ace0ecc441bd9dbd71108a0ac05d42642dea5c8fa506621002536eb18`.
The evaluator verified its stored router/shared-A bytes and all exact
BF16 B rows against the loaded centered checkpoint, rejected reserved
nibble 15, and dequantized Q15 B for PyTorch BF16 inference. This is a
full-model quality proxy; native pair-LUT full-model numeric parity was
not run.

The [automatic result](meth133_q15_full_model_audit_result.json), SHA-256
`18ac0e551f721f64e4d488f8b763926097bc06dadda8f9e52423c0803617a053`,
was committed at `1e5778c` before anonymous pairing. It completed in
881.485 seconds on the local RTX 3060, with 4.262 GB peak allocated
GPU memory and 2.558 GB final process RSS; all resource stops were
respected. No T4 was used.

| Frozen automatic measure | Exact BF16 E1280 | Q15 E1280 | Gate |
|---|---:|---:|---:|
| Pooled document BPB, 98,280 bytes | 1.234653 | 1.234823 | delta +0.000170 <=+0.005 |
| Largest category BPB delta | reference | technical +0.000378 | <=+0.02 |
| Donor prompt top-1, 4,268 positions | 94.916% | 95.173% | delta +0.258 point >=-1 point |
| Greedy EOS, 24 prompts | 21 | 20 | loss <=2 |
| Threefold repeated 8-gram / early non-EOS | 0 / 0 | 0 / 0 | pass |
| PIQA correct, 1,838 repeated items | 1,291 | 1,283 | delta -0.435 point >=-2 points |
| Paired bootstrap lower fifth percentile | reference | -0.816 point | >=-5 points |

The [anonymous pairs](meth133_q15_blind_pairs.json), SHA-256
`fee824711e58993cc6bbb0930ccc2f363a16d0ff5346044d09de54deb34cf791`,
were committed at `0681397` with a separate hidden mapping. Four of
the 24 A/B responses were identical. The excerpt-only
[verdict](meth133_q15_blind_verdict.json), SHA-256
`e8b0f58e71794e16a9ebdb6e89fa0ad71378e2a8bfba44cb19d2382bc1e00c84`,
was committed at `d52d4d8` **before** inspecting the mapping. It
counted A/B unsupported claims 42/47 and severe claims 17/23. The
[revealed mapping](meth133_hidden_map_revealed.json), SHA-256
`fa59a8ba43ae499d1dc3ea91593916f5deaf6e69ccc9f9fa22016f4198ee7100`,
and [unblinded score](meth133_q15_unblinded_score.json), SHA-256
`e03a9106d9d6fe777c10d287b8134070f67ff5f53c984705b2c38440283c5ae9`,
give:

| Blind grounding count | Exact BF16 E1280 | Q15 E1280 | Frozen rule |
|---|---:|---:|---|
| Unsupported claims | 39 | 50 | Q15 <= exact: **fail** |
| Severe unsupported claims | 16 | 24 | Q15 <= exact: **fail** |
| Missing requested supported detail | 0 | 0 | Q15 <= exact: pass |

By category, exact/Q15 unsupported counts were code 11/14, prose
14/20 and technical 14/16; severe counts were code 4/3, prose 6/13
and technical 6/8. One prose answer hallucinated eight associations
between adjacent index entries and locations; this contributes much
of the gap. That concentration is a limit on generalization from one
24-pair review, not a reason to change the preregistered gate. The
review used one agent and only the 384-character excerpts. It does
not estimate population quality precisely, and the BF16 reference
itself makes unsupported claims. Both arms retain these failure rows
as visible evidence.

METH-130 Q7 and METH-132 Q15 now both have useful measured storage and
CPU component behavior but **fail separate fresh blind full-model
grounding checks**. A smaller fixed-state factor L2 error did not
guarantee fewer autoregressive unsupported claims. The quality-valid
factor reference remains the exact METH-126 BF16 shared-A bank. A
future precision or mixed-format candidate needs a new preregistered
mechanism and fresh source-disjoint grounding, not threshold tuning on
these viewed responses. The larger objective still requires a compact
quality-valid donor core, native same-artifact >=50 accepted-token/s
and a learned expert-count ladder that measures routing cost and
quality as RAM permits more distinct experts.
