# PQT-007 source admission: independent-of-fitting news evaluation

Prospective, 5 October 2026. This stage downloads and binds public source
bytes/metadata only; no text/token selection, donor/reference computation,
fitting, quantization or quality evaluation. Native timing remains pending
owner coordination and is not a prerequisite for this separate preparation.

The next whole-model screen will retain the pinned Qwen/WikiText training
sources from PQT-001, but evaluate news texts from a separate fixed corpus.
[Provider dataset card](https://huggingface.co/datasets/fancyzhx/ag_news/blob/main/README.md)
describes English news text with a 7,600-example test split and categorical
labels. Use only its `text` field for next-token fidelity and continuations;
classification labels are not the language-model task. Metadata lists the
license as unknown; keep source bytes in the private research workspace,
without public redistribution or a claim of licensing terms not established
by the provider. No model-pretraining novelty or contamination claim.

Read the official public repository metadata once, bind its full 40-character
commit before downloading exactly one `data/test-*.parquet` file using that
immutable revision. No credentials/auth header, train-split download or public
dataset creation. Record full public API metadata, canonical revision URL,
file length/SHA-256, UTC time, controller/source commit and first failures.
File cap 20 MiB, each network request timeout 60s, total stage <=180s.
Exclusive owned `results/progressive_ternary/PQT-007/source_001/`; no overwrites.
On failure preserve partial bytes and a failed admission, then decide on a
numbered repair rather than retry silently.

Check source references at frozen checkpoint `d55f188` with `git grep` for
`ag.?news|fancyzhx` in docs/benchmarks/scripts. Prior manual `rg` returned no
references; retain the exact committed search result now. Absence of source
references is not proof that every parent experiment or the donor pretraining
never saw these texts. Declare new use in this research chain, independently
of WikiText fitting; do not use observed validation outcomes to choose samples
or thresholds. Verify selected token sequences are distinct from calibration
when the later frozen tokenizer/selection procedure is implemented.

This metadata/download stage must finish with an immutable
`PQT_007_NEWS_SOURCE.json` before scientific source/bundle freeze. The complete
PQT-007 protocol will fix all matrix scope, methods, calibration, news selection,
metrics, gates, seeds and runtime stops before any full-model conversion or
evaluation. The current whole-model objective remains unproven; this stage is
input admission, not an empirical quality result or goal completion.
