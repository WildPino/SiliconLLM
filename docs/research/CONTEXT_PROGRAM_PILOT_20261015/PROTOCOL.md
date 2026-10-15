# Context-program pilot

Status: authorized for preparation and one bounded Kaggle launch on 15 October
2026. No result is implied by this protocol. Runtime receipts are authoritative.

## Purpose and relationship to the project

The final project objective remains a reproducible transformation from pretrained
capacity into useful conditional execution, with at least 50 accepted batch-one
CPU tokens/s on the same quality-tested artifact. Dense and native MoE sources
are both allowed. This pilot tests a possible representation; it does not qualify
a complete runtime, large-model transfer, useful chatbot quality, or CPU speed.

The hypothesis is that a small program derived from a causal prefix can change a
shared cheap transformation in a way that remains useful across subsequent
tokens. Later work would replace repeated program construction with selective
retrieval from a large archive. The analogy to biological catalysts motivates a
question; it is not evidence.

Previous evidence distinguishes useful target-aware sparse subsets from cheap
predictable selection. It also shows that reducing expert payload can leave an
unaffordable shared core. This pilot therefore approximates a contiguous
multi-block transformation rather than selecting the source's original neurons.
See [the prior research brief](../PIPELINE_REASSESSMENT_20261015/CARVING_RESEARCH_BRIEF_20261015.md).
The previous recipient-plus-memory direction remains unproved; this is a new
bounded mechanism study, not a completed replacement pipeline.

## Authorization and resource envelope

The owner authorized English documentation, implementation, private Kaggle
uploads and execution using the existing three accounts. Local work is limited
to source preparation, static checks and lightweight operational requests: no
local donor acquisition, tensor training or model inference.

Use one account for the first launch, with T4 acceleration and a maximum kernel
runtime of 14,400 seconds. Count the reserved two-GPU shape conservatively as
up to eight T4-hours, even if a phase uses only one device. Verify the actual
device count/name in the cloud report; do not claim both GPUs are utilized when
they are not. Keep the other accounts available; do not launch redundant GPU
runs just because credentials exist.

The owner accepted a conditional second session up to four hours. It must have a
positive first-stage finding and a specific reason; it is not an automatic retry
or permission to search many hyperparameters. Aggregate ceiling: 16 T4-hours.
Queue time and quota accounting are platform observations, not assumed GPU-hours.

The official Kaggle notebook documentation currently describes a 12-hour CPU/GPU
session limit. Our four-hour cap is below it, so no GitHub cron is needed for
this campaign. Do not revive the old LM training workflow. A longer successor
would require bounded checkpoint continuation, a persistent total-budget ledger,
and a terminal stopping rule before scheduling relays.

Sources: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
[efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage).

## Fixed inputs and independence

Source model: `Qwen/Qwen2.5-0.5B-Instruct`, immutable revision
`7ae557604adf67be50417f59c2c2f167def9a775`.

Natural text: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, immutable
revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

Use the dataset's train/validation/test partitions, group lines into articles,
and reject article hash overlap across selected partitions. Keep every window
from an article in its partition. Record exact selected text hashes, token IDs,
selection rules and model/tokenizer revisions. Do not silently substitute
synthetic text, pick test examples after observing errors, or retune on test.
This encyclopedia-text diagnostic does not establish conversation diversity.

Initial configuration: 128 training articles, 24 validation articles and 24 test
articles; up to 256 tokens per sequence, a 32-token causal prefix, horizons
16/32/64, and the mapping from entry to layer 8 to entry to layer 12, using
zero-based indexing. The frozen JSON config and runtime's effective config
resolve exact implementation choices.

## Experiment 1: local thought dynamics

Capture authentic source inputs and outputs for the selected span once on Kaggle.
The cheap surrogate has a shared nonlinear residual path and a small program
generator. At a program boundary, derive the program exclusively from preceding
prefix states, never from the target horizon or later tokens. Hold it fixed
across the measured horizon. Start with three separately fitted controls:

1. A fixed shared transformation with no contextual program.
2. A contextual transformation with a prefix-derived program.
3. A contextual static-vector correction to distinguish dynamics modulation from
   carrying a useful additive vector.

Report parameter counts, effective active computation and program bytes for each
arm. A larger conditional model's advantage alone is not proof of temporal reuse.
At evaluation, also exchange programs between documents. Never re-fit an arm on
held-out observations to strengthen a favorable comparison.

First evaluate held-out span reconstruction and its dependence on horizon.
Source input states make this an oracle-input diagnostic. Then physically bypass
the source span in whole-model own-history continuations; inserting a hook after
computing the omitted blocks would not constitute a replacement. No teacher
resets are allowed in the candidate continuation. Full-prefix recomputation is
acceptable for this diagnostic, but is not an efficient deployed decoder.

Report the unmodified donor and all matched candidate controls, finite checks,
per-document errors, held-out token loss and generated token IDs/text. Keep a
clear distinction between held-out loss on fixed histories and own-history
generation; different generated histories have no direct token-level target
identity. Do not report surrogate reconstruction as chatbot preservation.

Prospective exploratory continuation gate: at horizon 16, contextual validation
pooled reconstruction SSE is at least 10% lower than the fixed arm (same tokens
and targets, so the common normalization cancels in this paired ratio);
shuffling programs destroys at least half this improvement; contextual held-out
token loss is no worse than the fixed arm; reconstruction improves at least 5%
over the prefix-conditioned additive-vector control; and the paired-document
bootstrap 95% interval for gain over the fixed arm has a positive lower bound.
All three prescribed training schedules must complete their actual optimizer
updates; a timeout cannot earn a positive decision. The frozen checkpoint rule
selects the best validation reconstruction checkpoint during each fixed schedule.
Report uncertainty across documents
and all horizon values, not just the best result. The reserved test is read once
after the validation decision, with no resulting parameter selection. Generation
remains a separate stability check, not a substitute for a blinded quality panel.

Incomplete training or an exhausted wall budget is inconclusive. A failed gate
rejects this finite recipe for escalation, not every contextual representation.

## Experiment 2: predictable exceptions

Only after the first-stage gate, reuse cached traces to compare keeping an old
program versus constructing a new causal program. Benefit labels may be produced
with the donor offline. Predictor features must be available to the cheap path
before the decision; do not feed it target errors, omitted-layer states or future
tokens. Separate predictor fitting and evaluation by article.

Compare learned, periodic and random refresh decisions at the same refresh count.
An oracle decision rule using observed benefits is a labeled upper reference,
never a deployable selector. If labels are local reconstruction benefits, report
them as such; do not silently call them a closed-loop generation improvement.
Count predictor cost and program construction/access work. A worthwhile signal
is reproducible benefit over the matched refresh controls. No such evidence is
established merely by a classifier's accuracy.

## Experiment 3: reusable capacity

Only after a positive first-stage gate, form two training-only program archives,
initially 64 and 256 entries if enough distinct training prefixes exist. Select
one program from causal features and report search cost. Keep the shared path,
program size and number active fixed. Compare retrieval with a freshly
constructed privileged program, no program, random retrieval and duplicates.
Validation/test programs must never be added to the training archive.

A signal is useful transfer to new articles, with increased distinct archive
capacity helping beyond duplicate expansion. Scanning this small archive is a
mechanism diagnostic, not proof of scalable bounded addressing. Learned retrieval
and real large-family donor transfer would remain later work.

The first implementation stores training-only prefix prototypes as archive keys
and values and constructs the modulation after retrieval. It therefore tests
prototype reuse rather than a compiled executable-program archive. Its zero-input
control retains generator biases, so it is labeled zero-prefix-input rather than
absence of every program contribution. Both follow-ups use direct batched calls
that repeat the generator per position; their runtime is charged as executed,
and no amortized-speed claim is made from those calls. The main experiment uses
one generator evaluation per causal anchor within each prefix pass.

## Execution, faults and scientific custody

The code bundle is private on Kaggle and includes exact source/config SHA256
pins. Acquire source weights and data on Kaggle; upload no credentials or local
corpus artifacts. Record software versions, devices, elapsed phases, peak memory,
actual steps and optimizer updates, curves and checkpoints. Fail explicitly on
NaN/Inf, wrong hardware, missing inputs, causal violations or numerical-control
failures. Never mask a fault as a weak scientific result.

The bootstrap enforces a total subprocess deadline below the four-hour platform
cap, including dependency setup, and leaves shutdown/output reserve. The pilot
also enforces internal training/evaluation deadlines. A killed process or timeout
is a terminal operational result, not PASS. A fresh launch consumes budget even
if it fails early; do not silently rerun under the same receipt.

Save launch, status and output receipts outside migrating artifacts under
`E:/SiliconLLM_Artifacts/campaigns/context_program_20261015`. The repository
contains only source, protocol, configuration and a small dated run record.
Do not modify historical sealed reports, the handoff's pinned bytes, or files
inside in-flight storage migration units. No scientific success is inferred from
successful upload or a RUNNING kernel status.

The final report must distinguish observed results, diagnostic controls,
inconclusive runs and future hypotheses. A useful outcome is a sufficiently
specific observation to justify or reject a larger next investment.
