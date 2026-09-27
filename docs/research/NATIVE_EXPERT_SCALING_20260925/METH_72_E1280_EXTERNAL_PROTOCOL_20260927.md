# METH-72: source-disjoint external audit of the trained E1280 bank

**Decision.** METH-71 passes its development quality and route gates at
E128 and E1280. Test whether the E1280 checkpoint preserves useful
donor behavior on material outside its training and development sources.
The exact bound model is METH-71 E1280 checkpoint SHA-256
`ead764d1f258819f2ee2d7e6af7995d329a078e360f06f085325c96ac6437fbb`,
not a continued or reselected checkpoint. The E128 result remains a
development control; external E128 comparison can follow the E1280 gate.

**Fresh content.** Before inference, freeze 8 code, 8 PG19 prose and 8
technical/general 4 KB spans with seed `meth72-external-72072`. Use the
METH-62 source commit `882bb43f9118df1e4f79f85a6072111897bede9e`
and PG19 physical test parquet. Exclude source IDs and first/middle/final
256-byte fragment overlaps with METH-17/19/25/41/45/57/62 selections,
the calibration and held-out texts, and this audit's previously selected
spans. Require at least 4,000 UTF-8 bytes and 256 tokens per document.
Prompt for a two-sentence summary and one excerpt-grounded specific
detail using the same Qwen chat template as METH-62. Screen all 24
excerpts for a visible specific detail before inference.

**Automatic gates.** Compare the original BF16 Instruct donor and the
METH-71 E1280 student on the frozen sources. Require pooled document
ΔBPB ≤ +0.02, each category ≤ +0.04; prompt-position top-1 ≥95%
pooled and ≥90% per category; greedy 128-token student EOS count ≥
donor−2, and each category repeated 8-gram 3× and short non-EOS count
≤ donor+1. On all 1,838 PIQA items require accuracy delta ≥−2 points
and paired bootstrap fifth percentile ≥−5 points. Re-evaluate exact
product-key top-four versus exhaustive pair scores on every external
prompt position. Permuting expert A/B factor pairs without changing
the router must worsen held-out raw BPB by at least +0.002. Require
at least 640 changed B slots per layer. Report route coverage and
max/mean load on external prompts; the METH-71 development load gate
does not silently become an external gate.

**Blind semantic gate.** After generation, form a deterministic A/B
file hiding donor/student identity. Review each pair against the
384-character excerpt and count unsupported factual claims, severe
unsupported named-entity/numerical/comparative/causal claims, and
missing requested specific details, with brief excerpt citations.
Keep ambiguous cases separate. Freeze the reviewer verdict before
unblinding. Require student counts no greater than donor in each of
the three failure categories. A single reviewer cannot establish a
population error rate.

Local RTX 3060 only; peak allocated GPU ≤10.5 GiB, RSS ≤20 GiB and
≤35 minutes for automatic scoring. Preserve partial evidence on a
failure and do not use T4. Even a joint pass would not prove packed
CPU LUT throughput, full native `engine.c` parity, 100B-scale RAM
feasibility or ≥50 accepted tokens/s on the same artifact.
