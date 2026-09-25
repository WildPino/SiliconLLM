# METH-01: active payload of the bound GigaChat Q4_K_M base

**Question.** Does the producer's actual mixed GGUF leave a plausible
20 ms/token path on the Ryzen 5 3600X without changing its active organs?
This is a header-derived byte calculation, not a C or llama.cpp throughput
measurement. It refines [METH-00's](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md)
ideal, uniform W4 scenario.

## Artifact, method and controls

The input is the producer **base-only**
`GigaChat3.1-10B-A1.8B-q4_K_M.gguf`, compared for quality to its
[source-bound BF16 control](../donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md).
Its size is 6,474,702,976 B; it was
previously hashed SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`
in the [fresh-quality manifest](../../../benchmarks/donor_adaptation/density/results/strat01_gigachat_fresh_quality_v2/heldout_q4_scores.jsonl.manifest.json).
This run checked the file size and GGUF header, **not** the full payload hash;
the prior scorer manifest supplies the input identity. Reader: `gguf-py`
from the local pinned llama.cpp revision
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.

Tool: [`gguf_active_ledger.py`](../../../benchmarks/native_expert_scaling/gguf_active_ledger.py).
It rejects any tensor-name discrepancy from the 26-layer GigaChat base
inventory, reads the declared 64 experts/top-4, counts all 414 tensors, and
reconciles the six large-matrix active element counts with the independent
1,628,078,080-element organ audit. For a stacked routed tensor, it charges
`n_bytes / 64 × 4`; for embeddings, one row; other tensors, their whole stored
payload. This is a **lower-bound addressed-weight model**: it excludes
runtime cache misses, repeated reads, activation/KV/state traffic, compute,
dequantization and format alignment beyond each tensor's `n_bytes`. The
physical DRAM traffic can be lower if a tensor is resident or higher if
accesses repeat; neither behavior was measured here.

From the repository root, with the pinned llama.cpp `gguf-py` path supplied:

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\gguf_active_ledger.py --model benchmarks\donor_adaptation\density\results\strat01_gigachat_q4_97045b2\GigaChat3.1-10B-A1.8B-q4_K_M.gguf --gguf-py C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\gguf-py --expect-file-bytes 6474702976 --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth01_gigachat_q4_active_ledger.json
```

The linked [raw JSON](meth01_gigachat_q4_active_ledger.json) is write-once.
The command loads no model weights and ran in about 9 seconds of wall time,
with no GPU/T4 execution, scoring, or C benchmark. It ran while NES-01
trained; no throughput inference is made from this timing.
The 6,468,600,064 B of summed tensor payload leave exactly 6,102,912 B
between payload and file size, matching the independently recorded GGUF
payload start in the donor metadata audit.

## Result and decision

| Organ | Stored tensor payload MB | Addressed payload MB/token | Actual types |
|---|---:|---:|---|
| MLA projections | 372.470 | 372.470 | Q4_K, Q5_0 |
| Routed experts | 5,697.700 | 356.106 | Q4_K, Q6_K |
| Shared experts | 89.027 | 89.027 | Q4_K, Q6_K |
| Dense layer-0 FFN | 26.772 | 26.772 | Q4_K, Q6_K |
| Router | 9.830 | 9.830 | F32 |
| Untied output head | 161.603 | 161.603 | Q6_K |
| Norms/biases and one embedding row | 111.199 stored including the full embedding table | 0.386 | F32, Q4_K |
| **Total** | **6,468.600** | **1,016.194** | mixed |

The producer Q4_K_M requires **50.810 GB/s of addressed compressed payload
at 50 tok/s** if each active tensor slice is read once. At the measured
40 GB/s aggregate DRAM yardstick in [PHASE64_BUDGET.md](../../PHASE64_BUDGET.md),
this payload alone prices to **25.405 ms/token**, before all other work. The
14 ms streaming design allotment would require at most 560 MB/token, a
reduction of **456.194 MB/token (44.89%)** from this file's active payload.
These are arithmetic requirements **under the stated yardstick**, not a
measured rate bound; cache, hardware, kernels and context can differ.

Even replacing the entire MLA and routed-expert payload with hypothetical
exactly 2-bit weights leaves **597.587 MB/token**, above the 560 MB allotment.
That calculation ignores 2-bit scales/padding and any quality loss. An
additional head/shared treatment or a structural change would be needed
under this design budget. The previous [STRAT-02E W2 failure](../donor_adaptation/probes/STRAT_02E_W2_BF16_EXPERT_SCOUT_RESULT.md)
is on another donor/format, so it neither qualifies nor rejects GigaChat W2.

**Decision:** do not promote the direct Q4 port or an MLA+routed-only ideal
W2 proposal as a credible ≥50 tok/s architecture on this CPU. Keep Q4 as the
quality/reference baseline. Before a conversion run, select a target geometry
with a quantitative path below the measured hardware's payload allotment and
freeze a paired donor-relative quality test. The next step is a candidate
selection across the already screened donors and transformation families,
using this actual-file ledger rather than nominal active parameter count.
No T4 or donor port was started by this cell.
