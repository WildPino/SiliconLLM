# METH-137: fresh quality test of learned E12800 against matched E1280

## Decision and binding

Run this test only if METH-136 completes both 256-update arms and passes
its finite, resource, artifact-readback, BF16-distinctness, coverage and
route-load gates. The question is whether the extra trained slots improve
held-out prediction while preserving donor-relative behavior and excerpt
grounding. Distinct B bytes and training loss alone do not count as useful
expert capacity. This is one 0.5B-family rung, not a 10B/100B result.

Before any METH-137 model inference, freeze a manifest that binds the
SHA-256 of the METH-136 result and both exported banks, the unchanged
METH-126 BF16 bank, the METH-107 centered checkpoint, the donor and
parent checkpoints, the tokenizer, all source/excerpt/prompt/token hashes,
and the answerability screen. Verify all bound hashes at evaluation.
Reject a mismatch. Evaluate four unchanged functions on identical IDs:
BF16 donor, original centered E1280, METH-136 continued E1280 control,
and METH-136 trained E12800 candidate. Reconstruct both trained arms
from their exported BF16 banks and fixed router/A bytes; verify the
METH-136 readback and eight pretraining parity prompts before inference.
The E12800 third-tier router stays exactly as frozen in METH-136.

## New sources and fixed measurements

Use deterministic seed `meth137-e12800-external-137000` to select eight
code files, eight technical files and eight distinct PG19 books from
`data/external/pg19/data/train-00006-of-00023-c50fea56d0518a87.parquet`
(SHA-256 `d4d8e090b8dd808dad686d93ea1309221edfbff3e412bb5da76a1f28108a7f5a`)
at pinned Git ref `882bb43f9118df1e4f79f85a6072111897bede9e` for Git
files. Follow the METH-133 4,095-byte excerpt, 384-character grounded
prompt and source/fragment-overlap selection method. Exclude source IDs
and first/middle/final 256-byte fragment overlaps with all earlier
manifests through METH-133 and with identifiable METH-136 training
sources. The H0 raw training stream is tokenized from its pinned
`calib.txt` corpus; record its provenance and check selected text
against that corpus if available. If disjointness or eight sources per
category cannot be established, stop. Freeze and hash all 24 selected
rows and a model-free answerability screen before inference.

Use the same 24 document BPB and donor prompt-top1 calculations as
METH-133. Candidate minus continued-control pooled BPB must be at most
`-0.00005` and the paired 5th-percentile bootstrap bound on
control-minus-candidate BPB must be above zero (10,000 source-level
resamples, seed 137137). Candidate minus control BPB must be no more
than `+0.02` in each category. Report the same comparisons against
original E1280 and donor; no improvement claim may use only the
original E1280 baseline. Candidate donor-top1 agreement may be no more
than one percentage point below the control pooled and two points below
in each category. Stop before generation if any document or prompt gate
fails.

If those gates pass, generate at most 128 greedy tokens per prompt.
Candidate may have no more than two fewer EOS terminations than control
pooled, and no more than one additional response with a threefold
repeated 8-gram or an early non-EOS stop below 16 tokens, pooled and
per category. Report original E1280 and donor generations too. Stop on
failure.

If generation passes, score all 1,838 local PIQA items with the same
answer scoring as METH-133. Candidate minus control accuracy must be
at least `-0.02`; the paired bootstrap lower 5th percentile must be at
least `-0.05`. Report original E1280 and donor scores. This is a
repeated regression task, not independent external task evidence.

An automatic pass only admits an anonymous, excerpt-only comparison
of candidate versus continued control on all 24 responses. Freeze
anonymous pairs and the reviewer verdict before revealing the mapping.
Candidate must have no more unsupported claims, severe unsupported
claims or answers missing a requested supported detail than control.
Record ambiguities and identical responses. A blind pass with the
BPB gain establishes evidence of useful extra capacity for this rung;
it does not establish a population guarantee or the project-wide
quality/speed/RAM scaling objective.

## Resources and stop rule

Use the local RTX 3060 only. Stop at 70 minutes, 10.5 GiB allocated GPU
memory, 40 GiB process RSS or 10 GB of new disk; save partial rows and
the exact failed gate. Do not tune thresholds, choose another bank or
select sources after viewing any inference output. No T4.
