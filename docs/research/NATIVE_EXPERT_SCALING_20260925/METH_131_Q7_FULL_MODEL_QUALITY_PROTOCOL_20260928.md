# METH-131: fresh BF16 full-model quality test of the Q7 E1280 bank

## Uncertainty and decision

METH-130's Q7 child-B bank passes a native fixed-state residual and CPU
component screen, with 276,578,344 stored bytes and zero route changes.
Autoregressive generation can amplify its 0.0465 pooled residual
relative L2 error. This test decides whether that exact bank can
advance toward native full-model integration. A pass does not by itself
establish >=50 accepted tokens/s, a compact donor core, or quality at a
larger learned-child count.

The reference is the BF16 Qwen2.5-0.5B-Instruct donor plus the
mean-preserving METH-107 centered E1280 adapter that passed METH-123.
The candidate differs only in child B reconstructed from the stored
METH-130 Q7 code/FP32 scale bank SHA-256
`20329a07f7dfcd3bfee08d4ee64b7e243d0004f4e18ec3543c78516c203b0265`.
Check the source donor SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-56 parent checkpoint
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`,
METH-107 child checkpoint
`15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`,
and METH-126 BF16 bank
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`.
The evaluator must compare the Q7 bank's dequantized B against the
centered BF16 B tensor for all rows before inference, and must verify
router/shared-A identity. PyTorch's BF16 matmul of dequantized Q7
weights is a quality proxy for the native pair-LUT arithmetic; numeric
parity between these implementations remains a later integration gate.

## Frozen data and staged gates

- Deterministically select 24 new 4,095-byte source excerpts: eight
  code and eight technical files from pinned Git ref
  `882bb43f9118df1e4f79f85a6072111897bede9e`, and eight distinct
  PG19 books from local train shard 00004. Use seed
  `meth131-q7-full-model-131000`, source IDs and first/middle/final
  256-byte overlap exclusion against every earlier audit manifest
  through METH-121, including its 24 sources. Freeze file, excerpt,
  token and prompt hashes before inference. All prompts use the same
  384-character excerpt template as METH-121; screen answerability
  before generation. If the pool cannot supply eight per category,
  stop before inference and amend the protocol visibly.
- On identical documents, require candidate minus exact E1280 pooled
  BPB <=+0.005 and each category <=+0.02. Against the untouched BF16
  donor, require candidate prompt top-1 agreement no more than one
  percentage point below exact E1280 pooled and no more than two points
  below it in any category. Capture all per-source losses and counts.
  Stop on a failed document/prompt gate.
- If those pass, generate up to 128 greedy tokens on all 24 prompts.
  Candidate may have at most two fewer EOS terminations pooled than the
  exact E1280 reference, at most one additional response with a
  three-times repeated 8-gram, and at most one additional early
  non-EOS stop under 16 tokens, pooled and per category. Stop if this
  gate fails.
- If generation passes, score all 1,838 local PIQA items with the
  existing paired scorer. Candidate minus exact E1280 accuracy must be
  >=-0.02 and the paired bootstrap lower fifth percentile >=-0.05.
  PIQA has already been used in previous decisions; it is a repeated
  regression task, not independent task generalization.
- Automatic success only admits a separate arm-blind excerpt-grounding
  review. Freeze anonymous A/B generations and one verdict per pair
  before unblinding. Count unsupported claims, severe unsupported
  claims and responses missing a requested supported detail. Candidate
  must have no larger count than exact E1280 for all three. Keep the
  excerpt limit and reviewer uncertainty visible; a single 24-pair
  review cannot establish population superiority.

## Resources and interpretation

Run on the local RTX 3060, at most 35 minutes, 10.5 GiB allocated GPU
memory and 20 GiB RSS. Do not use T4. Save partial rows at each gate,
hashes, runtime and failures. No threshold or format tuning after
examining new-source outcomes. If all quality gates pass, the next
step is native Q7 integration with a quality-valid compact donor core,
numerical parity on the same artifact, then accepted-token throughput.
Neither a pass nor a fail on E1280 answers whether 10B/100B donors can
create proportionally more *distinct useful learned* child experts.
