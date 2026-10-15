# First pilot result: exploratory negative

15 October 2026. Kernel version 1 completed. The frozen exploratory gate failed
on validation and the reserved test. Experiments 2 and 3 were not executed.
No new cloud run is launched by this record.

The bootstrap reports 167.799 seconds total cloud process time, including setup;
the numerical pilot reports 93.827 seconds. All three arms completed their 1,200
optimizer updates. This is approximately 0.0932 T4-hours if conservatively
counting both allocated devices throughout bootstrap runtime; platform quota
charges and startup overhead are separate observations. GPU 0 performed the work.

## Observations

The source span contains 59,649,536 parameters. The fixed surrogate has 147,922
parameters; conditional and additive-context controls each have 148,576.
These local counts do not imply whole-model compression or useful quality.

| Measure | Validation | Reserved test |
| --- | ---: | ---: |
| Conditional reconstruction gain relative to fixed | -3.3361% | -3.0905% |
| Conditional reconstruction gain relative to additive context | 0.0967% | 0.0568% |
| Donor next-token NLL | 2.903802 | 2.943332 |
| Fixed-surrogate next-token NLL | 3.251781 | 3.339570 |
| Conditional-surrogate next-token NLL, horizon 16 | 3.184427 | 3.255215 |
| Additive-context next-token NLL, horizon 32 | 3.188206 | 3.260396 |
| Conditional donor-relative KL, horizon 16 | 0.353947 | 0.358164 |
| Additive-context donor-relative KL, horizon 32 | 0.338080 | 0.347977 |

Mapping aggregates use 24 documents per held-out partition; whole-model metrics
use the first 12 selected documents per partition. Whole-model NLL is evaluated
on fixed dataset histories. Four short own-history continuations per arm are a
separate smoke diagnostic and do not constitute useful-chat admission.

The contextual program worsens pooled hidden-state reconstruction while improving
token NLL over the matched fixed arm. That discrepancy motivates investigating
task-sensitive approximation geometry. It does not prove the hypothesis: the
additive-context control performs similarly in NLL and better in donor-relative
KL. Dynamic modulation has not established a distinctive advantage.

The donor retains a substantial advantage. Horizon changes from 16 to 64 have
small observed NLL effects in this restricted corpus, which could reflect weak
program influence rather than robust long-lived computation. Neither a new gate
nor a positive interpretation replaces the frozen negative result.

## Custody and operational notes

Full output download:
`E:/SiliconLLM_Artifacts/campaigns/context_program_20261015/pilot-a-20261015/output-1792093304`.
Small JSON/log readback:
`E:/SiliconLLM_Artifacts/campaigns/context_program_20261015/pilot-a-20261015/summary-readback`.

Twelve downloaded JSON records were checked against the cloud artifact manifest;
all twelve SHA256 comparisons matched. This is a scoped receipt check, not an
independent numerical replay or verification of every tensor file. No tensors
were loaded or model calculations executed locally.

The first output CLI returned exit 1 after reporting file downloads because its
Windows console could not encode Unicode log/progress text. The cloud run itself
completed. A UTF-8 child CLI retrieved the summary/log successfully. The local
transport now forces UTF-8 for child CLI processes; the already executed kernel
and its source/config pins are unchanged. Preserve both operational receipts.

The owner subsequently expanded the available investigation envelope to 80 hours
without specifying whether these are session-hours or GPU-hours. Do not silently
reinterpret that as 160 GPU-hours. No additional allocation is made here; future
work should use explicit conservative T4-hour budgets and new held-out evidence.

## Hypothesis suggested by the discrepancy

For a source tail with logits z(h) and p=softmax(z), a small state perturbation e
has local donor-relative KL approximately

`0.5 * e^T J^T (diag(p) - p p^T) J e`,

where J is the logit Jacobian with respect to h. Euclidean reconstruction error
does not weight directions by their downstream effect. This identity motivates a
different prospective experiment, not a retroactive change of the current gate.
Preserving the next token alone is insufficient: representations must also
preserve distinctions needed by later continuations and actual own-history use.
