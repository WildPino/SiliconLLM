# Behavior-aware post-training quantization evidence

Cutoff: 2026-10-06. Fifteen primary paper clusters were inspected in full-text views, including two 2026 additions and the historical incremental-conversion precedent. Search snippets were used only for discovery. No scientific runs, fitting, GPU work, private operations, or native timings were performed. `sources.jsonl` records stable SHA-256 identifiers of canonical URLs, versions, dates, and inspection locations; these identifiers identify URLs, not downloaded content. `evidence.jsonl` contains bounded paraphrases and no quotations. Dynamic repository files are explicitly identified as unpinned.

## Decision-relevant interpretation

The user's proposal belongs to an established class: change discrete weights progressively while measuring or approximating the resulting behavioral error and compensating in remaining degrees of freedom. Random proposal-and-acceptance search is one possible algorithm, but the inspected evidence favors exploiting activation covariance, learned rounding, block reconstruction, or structured codebooks. The literature does not make random search necessary, and an exponentially large search space does not become inexpensive merely by using small batches.

INQ supplies an especially close historical precedent: successive freezing and recovery from pretrained weights. Its learning/data requirements differ from a small calibration-only reconstruction budget; that distinction belongs in any comparison with this research.

Alphabet, fitting budget, and fidelity target must be distinguished. A scalar three-level grid, a sum of multiple ternary planes, a transformed vector codebook, and an additive collection of codewords are different representations. A nominal bitrate cannot establish compatibility with the existing ternary kernel. Code indices, scales, codebooks, rotations, residuals, retained floating-point parameters, packing, and runtime temporaries must be counted separately.

A small improvement in local reconstruction, a favorable perplexity result, preserved benchmark means, high distributional agreement, and identical greedy generations answer different questions. Neither a theory about a quadratic proxy nor a published positive low-bit result guarantees the original Switch FFN 1% held-out RMS gate or the complete research goal. Conversely, the audited negative arms do not establish a universal impossibility theorem.

## Mapping to existing audited evidence

| Existing evidence | Literature implication, with explicit inference status | Consequence for future work |
| --- | --- | --- |
| PQT-005: excellent calibration reconstruction failed to transfer to development expert contexts. | BRECQ's generalization argument and CoreQ's treatment of unreachable propagated mismatch identify finite-calibration overfitting and input mismatch as plausible mechanisms; this is an interpretation, not a retrospective causal diagnosis. | Preserve actual original-context qualification and held-out gates. A material new hypothesis must change the representation, objective, or input mismatch treatment. |
| PQT-007: local matrix improvements accompanied worse whole-model divergence. | Block and end-to-end reconstruction approaches explicitly account for interactions that independent matrix proxies omit. | Local improvements remain a screening result. Admission to whole-model evaluation requires an exported artifact and a frozen quality protocol. |
| PQT-008/009: longer fitting and broader calibration improved some metrics but missed absolute goals. | Published calibrated methods vary substantially in data, reconstruction scope, and tuning. Their positive results cannot justify endless repetitions of the same fixed alphabet and objective. | Do not relaunch an unchanged longer/broader arm. Quantify the unresolved mechanism before another bounded test. |
| PQT-010: restoring a sensitive shared tensor did not rescue quality. | Sensitive-layer exceptions can matter, but one exception does not establish that the complete network is recoverable. | No recommendation to repeat the same embedding/head restoration. Any new exception must have independent prospective evidence and counted storage. |
| PQT-011: learned scales yielded limited or harmful transfer. | Learned clipping/ranges in other papers are not evidence that this specific scale-only ternary arm will succeed. | Do not equate an existing learned-scale failure with a complete OmniQuant or AutoRound reproduction, and do not repeat it without a materially changed question. |
| PQT-012: low-rank output residual on four naturally routed final-encoder experts, preregistered independently. | A changed residual representation offers degrees of freedom absent from the failed pure scalar arms; reviewed PTQ work supports accounting for those added degrees of freedom, not its success. | Treat as mixed representation, count complete storage, fit only calibration, and retain the 1% and comparison gates. No claim of whole-model or native usefulness from an expert screen. |
| Native packed correctness qualified; owner reservation for native timings is pending. | Compression and specialized decoding can produce speedups in published environments, but none transfers a speed claim to the local CPU path. | Preserve the native timing reservation and run no competing timing workload. |

## Materially different hypotheses, not commitments to new runs

1. Original-versus-partially-quantized input mismatch, with a limited correction rather than unrestricted calibration interpolation. This differs from independent matrix fitting and requires a fresh protocol before use.
2. Changed representation capacity: a small counted output residual, a richer asymmetric three-level grid, or a multiple-plane representation. The latter two change the kernel/representation contract and cannot be promoted as the already tested symmetric alphabet.
3. Selective precision based on prospectively estimated behavioral sensitivity and an exact capacity budget. Restoring the already failed shared embedding/head is not sufficient justification.
4. If a pure ternary branch is closed negatively, richer 2/3/4-bit vector or additive formats are alternate research objectives, not evidence that the ternary goal has been achieved.

## Negative closure and remaining gaps

A reproducible, scoped negative result is a valid research outcome. Closure should identify the exact model revisions, alphabets, granularity, original-context coverage, fitting budgets, precision exceptions, storage accounting, held-out quality, and native measurements actually tested. A defensible statement is that no tested candidate met the joint requirements within the preregistered methods and resources. A statement that pretrained models cannot be converted to ternary is unsupported.

Potentially important open gaps are natural-context coverage of rarely activated experts, representation granularity different from group 64, strict common-scale symmetry versus independent positive/negative levels, accumulated input mismatch, broader capability evaluations, and native integration costs. These are not all required experiments. The lead research record should explain which were investigated, rejected as outside scope, unavailable under the resource budget, or still justify a finite discriminating test.

New-source triage: arXiv 2609.01962, 2509.16989, and 2510.03267 were discovered and passed to the ternary/MoE lane for full inspection. They are not counted among this lane's fourteen inspected clusters. A search result calling 2602.05902 SNRQ refers to an earlier title; the latest inspected version is CoreQ v2. OmniQuant and AQLM latest versions were inspected after earlier HTML versions surfaced, and older-version locations were not silently used as the authoritative version.
