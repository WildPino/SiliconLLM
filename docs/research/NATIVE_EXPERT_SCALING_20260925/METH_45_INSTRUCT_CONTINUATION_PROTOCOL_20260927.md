# METH-45: Instruct E128 continuation and fresh external quality gate

**Uncertainty and prior.** METH-44 starts from exact Instruct donor logits,
trains all 24 E128/top-4/rank-8 FFNs, and after 16 updates passes a
same-corpus 24-prompt chat top-1 development gate (95.981%) with raw BPB
−0.001843. It has not shown useful routing, sustained quality, generative
retention, external document/task quality, or compatibility with a compact
native export. Continue its exact optimizer/RNG checkpoint under a fixed
budget and decide whether this Instruct adaptation is a viable parent for
export and larger-E learning. A failure closes this recipe, not all
conditional transfer methods.

Bind the METH-44 checkpoint SHA-256
`4fade15804f5e2297ff60cafacf870e5ae11926ba886786a524110dea94f16da`,
Instruct weights SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-43 teacher data SHA-256
`f42f678280a9086a0fc2ec6e57a712e33fdea8f5cb62bdac7f53c9f28a5f1dd7`,
METH-43 training manifest SHA-256
`17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055`,
and METH-44 development manifest SHA-256
`703abca3f83a983d9380ba84c1a79744d15881d638174b064cf38a47453f47ba`.
Use the pinned Qwen h0 train IDs; exclude the METH-44 development rows
from raw sampling. All 256 teacher responses stay in the chat training pool.

**Fresh evaluation fixed before continuation.** The
[external manifest](meth45_fresh_external_manifest.json), SHA-256
`15a68db9a43a8066e70b1b261f819cab3d6779d8b52633bcce7c725d8d54acfb`,
contains 12 source-disjoint spans of up to 4,095 bytes: four each of code, PG19 prose,
and technical documents. Its selector first reconstructs the original
METH-41 manifest exactly, then excludes those 24 source IDs and span
fragments, using seed `meth45-external-45045`. It also stores 12 Instruct
chat-summary prompts from the first 384 characters of those spans. This
is independent of all METH-43/44 chat training rows. The full PIQA
validation task source and labels remain pinned at METH-21 hashes
`93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d`
and `b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb`.
PIQA was previously viewed for a *different* base donor/adapter; it is
an external task check for this Instruct candidate but not a never-viewed
benchmark. No candidate hyperparameter selection may use these external
results.

**Training.** Resume the METH-44 16-update checkpoint, including optimizer,
NumPy and Torch RNG states. Keep its exact 2 raw plus 2 full-chat
microbatches, loss masks, KL coefficients, LRs (`3e-4` factors, `3e-5`
router), BF16 frozen donor and gradient clipping. End at update 1024.
Save recoverable checkpoints at 256, 512, 768 and 1024. At those updates,
evaluate only the existing raw heldout slice and METH-44 same-corpus
development prompts. Stop if raw student-minus-donor BPB exceeds +0.05,
development top-1 falls below 95%, any gradient/loss is nonfinite, or
budget is exceeded. A stopped run is a failed continuation; do not select
an earlier checkpoint after viewing fresh external results. Local RTX 3060
only, ≤35 minutes, ≤10.5 GiB peak allocated GPU and ≤20 GiB RSS; no T4.

**Terminal gate, only if update 1024 completes.** Load the 1024 checkpoint
and compare donor/student on the frozen external manifest. Require pooled
student-minus-donor BPB ≤+0.02 and each category ≤+0.04, ≥95% pooled
prompt-position top-1 agreement and ≥90% in every category. On 128-token
greedy chat continuations require student threefold repeated-8-gram count
≤donor+1 pooled and per category, early non-EOS truncations ≤donor+1,
and EOS completions ≥donor−2 pooled; save every continuation for manual
factual inspection. On full 1,838-item PIQA require accuracy delta
≥−2 percentage points and paired bootstrap 5th percentile ≥−5 points,
using METH-21's exact scoring convention and 20,000-draw seed 212121.
The trained route must matter: independently permute router expert rows
within each layer with NumPy seed 451616 and require permuted-minus-trained
BPB ≥+0.002 on the fixed raw heldout slice. At least 64 distinct output
slots per layer must have changed. External evaluation budget ≤30 minutes
on the same GPU/memory bounds; stop on any mismatch or budget breach.

A joint pass would license a packed-export and CPU-route experiment on
this parent, plus a separately predeclared distinct-expert-count ladder.
It would not establish 10B/100B quality, bounded large-E routing, or
≥50 accepted batch-1 tok/s in `benchmarks/phase60/engine.c`. A failure
must be recorded with its exact failing gate and prevents native promotion.
