# METH-213/214: fresh quality of the saved Q8/exact-head core

Candidate is fixed before selection: METH-211 core SHA256
`4bb34a2ac14bfa3f4fe42490a6755513cf6b08e82e981c86926e374b12353c1a`,
loaded exclusively by the passing METH-212 loader, with original centered
E1280 factors. No training or candidate selection from this cohort.
Bind the METH-212 result, donor, tokenizer and parent/child checkpoints.

METH-213 chooses eight code and eight technical documents at METH-121's
pinned Git commit and eight PG19 books from
`data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet`.
Seed `meth213-stored-core-quality-213213`. Reuse METH-176 source and
first/middle/final 256-byte overlap exclusion, including its full prior
manifest/teacher list and the METH-176 quality manifest itself. Novelty
means selected source/fragment disjointness, not a claim that the whole
parquet file has never been scanned. Use fixed 4095-byte spans, >=256
tokens, 384-character excerpt prompts and the same lexical screen.
Freeze file/span/token/prompt hashes and one excerpt-visible supported
detail per item before any donor or candidate inference. Stop on failed
bindings, overlap or category quotas. CPU selection <=50 minutes,
12 GiB RSS and <1 GB output.

METH-214 evaluates three fixed arms in the same 24-item order: BF16
pretrained donor, original BF16 centered E1280, and stored compact core
with the same centered E1280. Use full exact-head probabilities for NLL;
report the K64 proposal separately. Require candidate-minus-donor AND
candidate-minus-BF16-E1280 BPB <=0.01 pooled, <=0.02 each category.
Donor-top1 agreement may lose <=1 percentage point pooled and <=2
points per category versus BF16 E1280. No BPB improvement is required
for a compression claim. Report paired source bootstrap intervals
(10000 resamples, seed 214214) without using them to pick a candidate.
Stop before generation/task scoring on any prediction gate failure.

On prediction pass, verify cached versus full recomputation greedy
choices for 16 steps on the first prompt of each category, for all
three arms; any mismatch stops promotion. Generate <=128 greedy tokens
on every prompt. At every candidate generated state, require the full
BF16 argmax to occur in the fixed R8/FP16-scale top64 shortlist and
require exact selected BF16-row reranking to reproduce the full choice,
with lowest-token-ID ties. Any omission/mismatch rejects the K64 path.
This remains a finite-state screen, not a universal guarantee.

Relative to EACH control, candidate may have <=2 fewer EOS terminations
pooled, <=1 additional threefold repeated 8-gram or early non-EOS stop
below 16 tokens pooled and per category. Then score all 1838 local PIQA
items using the prior mean-suffix-NLL scoring. Candidate accuracy minus
each control must be >=-0.02 and paired bootstrap fifth percentile
>=-0.05. PIQA is previously consumed regression evidence. Preserve
partial metrics at every stage; stop on an automatic failure.

Only after automatic passes, freeze anonymous three-arm excerpt-only
responses and blind grounding verdicts before reading the mapping.
Candidate may have no more unsupported claims, severe unsupported
claims, or missing requested supported details than either control.
All 24 prompts must have an answerable frozen anchor; record whether
each response supplies a supported specific detail. A compression
quality pass requires at least 20/24 candidate responses with such a
detail, in addition to the relative blind gates. Review is by the same
research agent and its limited independence must be stated.

Local RTX 3060 only; six host threads, <=70 minutes inference/task,
20 GiB RSS, 10.5 GiB allocated GPU and <1 GB new outputs. No T4.
Passing licenses native composition and real traffic/precision checks;
it does not prove useful additional experts, 10B/100B transfer or
>=50 accepted batch-1 tokens/s. The larger-n local-key experiment
retains its separate route/function gates.
