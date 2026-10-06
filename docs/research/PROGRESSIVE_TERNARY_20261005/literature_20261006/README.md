# Literature review evidence package

6 October 2026. Literature phase complete; experimental goal active.

Start with [report.md](report.md), the authoritative English review; [report.html](report.html)
and [report.pdf](report.pdf) are convenience copies. The parent
[decision/resumption record](../LITERATURE_DECISION_20261006.md) and
[index](../TERNARY_INDEX.md) connect the review to pending experiments.

## Contents and strength of evidence

47 unique primary paper/project clusters; 49 lane registry entries are merged
without counting repeated INQ versions as independent confirmation. The three
method/MoE lanes and compensation lane preserve original records. No paper was
numerically reproduced in this phase. Some published author code was inspected;
pinned companion hashes and unpinned status are recorded where available.

- `sources.jsonl`: canonical paper identities, aliases, versions, authors,
  dates and bibliography numbers. URL identity hashes are not content hashes.
- `evidence.jsonl`: 53 bounded reading/extraction records with precise
  section/table/code locations; no full paper copies or invented quotations.
  The append-only PTQTP correction supersedes an earlier entropy-label phrase.
- `claims.jsonl`: 217 atomic reading observations and numeric extracts linked
  to evidence. These are not 217 independent replications.
- `report_traceability.jsonl`: citations and evidence for47 cited report
  paragraphs/tables, with scope review. Uncited recommendations are prospective;
  local facts use linked project records; arithmetic is separately checked.
- `local_sources.jsonl`:12 project documentation identities read for this
  reconciliation. These hashes identify the review snapshot, not all underlying
  raw experiment data; the original run records remain authoritative.
- `search_log.jsonl` and `run_manifest.json`: queries, cutoff, assumptions,
  providers, consolidation counts and phase status.
- `checks/`: strict URL/citation outcome, derived arithmetic, scope/format
  verification, PDF hashes and all15 rendered pages/four contact sheets.

## Verification and reproduction

[Verification summary](checks/review_verification.json) records all checks and
the repaired encoding/display/pagination failures. Structure and citation
completeness pass; all47 primary URLs were accessible. The auxiliary support
checker finds217supported reading claims, with zero unsupported. It is a
lexical consistency check and does not establish semantic truth. Scope/number
review and primary locators remain necessary. The support adapter explicitly
labels its text fields as paraphrases/numeric extracts, not verbatim quotes.

`report.draft.md` is the assembly source with symbolic paper keys;
`assemble_review.py` resolves those keys, merges retained lanes and writes the
Markdown/bibliography/ledgers. Reassembly resets verification status and must
be followed by review; it is not an automatic approval. `package_review.py`
creates HTML/PDF using the verified Markdown, the repository research template
and the bundled document runtime. `render_review.py` renders every PDF page.
`verify_review.py` checks registry resolution, report/section/citation coverage,
local identities and derived costs, and records the completed scope review.

The skill's report, citation, claim-support and HTML validators were used.
Windows terminal output requires Python UTF8 mode. PDF inspection found and
repaired an orphan final word; the final15pages are readable with complete
bibliography, no clipping/overlap and no missing glyphs. PDF author metadata is
empty. An explicit artifact-creation marker was successfully run once before
the first PDF authoring operation.

No original model fitting, GPU benchmark, private upload or native timing was
performed. PQT012's frozen candidate screen remains unexecuted. A valid negative
outcome requires the finite scoped investigation described in the review; this
package completes the literature phase alone.
