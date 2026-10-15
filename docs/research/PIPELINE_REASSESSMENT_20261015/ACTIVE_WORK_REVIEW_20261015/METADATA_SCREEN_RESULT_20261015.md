# Conditional-memory source structure: bounded metadata result

15 October 2026. Metadata retrieval and an analytical source-work ledger are
complete. No tensor values, tensor headers, model execution, training, GPU job,
CPU benchmark or source-quality evaluation was performed. The pipeline goal
remains incomplete. Architectural interpretation continues in the
[carving brief](../CARVING_RESEARCH_BRIEF_20261015.md).

## Reproducible inputs and result

The [retrieval receipt](metadata/retrieval.json) records14 requests,1,109,168
retained bytes and7.8 seconds. Individual metadata bodies are bounded to2 MiB,
the intake to16 MiB/20 calls/300 seconds. Mutable API discovery is distinguished
from subsequent immutable config/code/index URLs. No downloaded Python was
executed. [Acquisition code](acquire_metadata.py) was committed at379dcee.

The LongCat-Flash-Lite checkpoint revision is
`b62b68827ead0b7fef3ba98b57f18484acaaec06`; the reviewed Transformers v4.57.6
implementation resolves to `753d61104116eefc8ffc977327b441ee0c8d599f`.
The source index has11,160 names and declares138,179,712,000 total tensor bytes,
including auxiliary prediction tensors; index bytes are not acquired weights.
All acquired bodies have SHA256 seals in the receipt.

[Ledger code](derive_active_work.py), committed at3d8de20, rechecked every
retrieval body size/hash, expected implementation snippets and indexed matrix
names before deriving the [ledger](active_work_ledger.json). The isolated
stdlib-only command exited0. Matrix shapes are inferred from pinned config and
source, **not verified tensor headers**. Exact arithmetic gives:

| Source role | Coefficients | Hypothetical representation | Ideal payload bytes/token |
| --- | ---: | --- | ---: |
| 28 MLA sublayers | 915,668,992 | 4bit | 457,834,496 |
| 28 dense gated FFNs | 1,585,446,912 | 4bit | 792,723,456 |
| 14 routers | 16,515,072 | F32 | 66,060,288 |
| Full vocabulary head | 402,653,184 | 4bit | 201,326,592 |
| 12 memory post-projections | 9,437,184 | 4bit | 4,718,592 |
| Accounted subset | 2,929,721,344 | Mixed, above | 1,522,663,424 |

At50 tokens/s that subset alone corresponds to76.1331712 GB/s of hypothetical
payload delivery. It excludes routed experts, KV/state/attention math, norms,
biases, base memory rows, physical quantization padding/scales and frontend.
It assumes a straightforward source-shaped streaming implementation; caching,
fusion, amortization or changed functions need a separate physical ledger.
These numbers are neither a measured DRAM rate nor a universal latency bound.

## What the conditional memory actually changes

The12 n-gram tables have31,406,985,216 inferred distinct coefficients. At each
token the source selects one256-element row per table from current/recent
token IDs, requiring6,144 BF16 row bytes before embedding/projection overhead.
This is a real large-capacity, small-lookup pattern. But14 dual decoder blocks
still execute28 MLA and28 dense FFNs. The input memory's exact extraction
does not make that decoder optional or transfer its behavior into a smaller
recipient automatically.

The Sparse successor config retains the shared dimensions; its sparse history
attention does not establish sparse execution of these weight matrices. Other
acquired Switch configs support family comparison only. In particular, larger
Switch checkpoints do not necessarily retain the base model's small core.

## Decision

Deprioritize an unchanged LongCat port for this target; do not acquire full
weights on the premise that lookup capacity makes the shared core cheap.
Retain conditional-memory extraction as a distinct possible transformation
lead. Its learned fusion, context dependence, donor-specific usefulness and
conversion data cost need a separate hypothesis and bounded test.

This is one dated source-structure screen. It neither proves that no suitable
native family exists nor qualifies a compact replacement, a chatbot, useful
capacity scaling or50 accepted CPU tokens/s. Primary literature review notes
remain in [sources](sources.jsonl) and [evidence](evidence.jsonl); the broader
strategy decision is pending.
