# METH-133: new-source full-model quality audit of the Q15 E1280 bank

## Decision and uncertainty

METH-131 rejected Q7 at its blinded semantic gate despite passing
automatic full-model checks. METH-132 reduced fixed-state factor error
from Q7's 0.0465 to 0.01894 pooled relative L2 in the same 276.6 MB
bank. Autoregressive grounding may still degrade. This experiment
decides whether the exact stored METH-132 Q15 bank is eligible for
native full-model integration; it does not test a compact donor core,
accepted-token rate, or the usefulness of more than 1,280 learned
children.

Bind the BF16 Qwen2.5-0.5B-Instruct source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-56 parent checkpoint
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`,
METH-107 centered child checkpoint
`15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`,
METH-126 exact BF16 bank
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`,
and METH-132 Q15 bank
`9e03941ace0ecc441bd9dbd71108a0ac05d42642dea5c8fa506621002536eb18`.
Verify every router/shared-A byte and exact centered BF16 B row against
the loaded checkpoint. Decode Q15 nibbles as signed levels -7..7;
reject the reserved value 15 and nonfinite/negative scales. PyTorch
BF16 inference after dequantization is a quality proxy. Native pair-LUT
full-model numeric parity is a separate gate.

## Frozen data and gates

- Deterministically select 24 fresh 4,095-byte excerpts with seed
  `meth133-q15-full-model-133000`: eight code and eight technical
  files from pinned Git ref `882bb43f9118df1e4f79f85a6072111897bede9e`,
  and eight PG19 books from local train shard
  `train-00005-of-00023-87dece6a0ab708e8.parquet` SHA-256
  `047e8d8d9609a57bff14be4091d9226cebe5fab59f504eb3e5a917c8f69df5ac`.
  Exclude source IDs and first/middle/final 256-byte fragment overlap
  with every previous audit through METH-131. A shared Git repository
  domain is visible, but files and fragments must differ. Freeze
  source, excerpt, document-token and prompt-token hashes, and screen
  answerability from the 384-character visible excerpts before any
  model inference. Stop if eight per category are unavailable.
- Against exact E1280 on identical documents, candidate-minus-exact
  pooled BPB must be <=+0.005 and each category <=+0.02. Candidate
  donor prompt top-1 agreement may be at most one percentage point
  below exact pooled and two points below in any category. Save every
  source row; stop on failure.
- If the document/prompt gate passes, generate up to 128 greedy tokens
  per prompt. Q15 may have at most two fewer EOS terminations pooled,
  one additional threefold repeated 8-gram response, and one additional
  early non-EOS stop under 16 tokens, pooled and per category. Stop on
  failure.
- If generation passes, score the same 1,838 local PIQA items. Q15
  minus exact accuracy must be >=-0.02, and the paired bootstrap lower
  fifth percentile >=-0.05. PIQA is a repeated regression task.
- Automatic success only admits a separate anonymous A/B review of
  the 24 excerpt-grounded responses. Freeze the anonymous pairs and a
  verdict before revealing the mapping. Q15 must have no more
  unsupported claims, severe unsupported claims, or answers missing a
  requested supported detail than exact E1280. Record ambiguities and
  the limitation of one 24-pair review.

## Cost, stop and interpretation

Use the local RTX 3060 only; no T4. Stop at 35 minutes, 10.5 GiB peak
allocated GPU memory or 20 GiB RSS. Save partial gate rows, hashes,
runtime and failures. Do not tune thresholds, choose another Q15 scale
or select documents after seeing outcomes. A full pass licenses native
pair-LUT full-model parity and same-artifact throughput with a
quality-valid compact core. It does not show >=50 accepted tokens/s or
RAM-proportional *useful distinct* experts at 10B/100B scale.
