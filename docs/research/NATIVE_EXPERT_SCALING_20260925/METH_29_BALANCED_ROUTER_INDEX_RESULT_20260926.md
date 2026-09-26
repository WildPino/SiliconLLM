# METH-29: post-hoc balanced router index fails route-recall gate

The [prospective protocol](METH_29_BALANCED_ROUTER_INDEX_PROTOCOL_20260926.md)
fixes one weight-only hierarchical index on the trained Qwen2.5-0.5B
E128 router and two candidate budgets. Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth29_balanced_router_index.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth29_balanced_router_index_result.json`.
The run binds the stored R8 core and factor-0.50 E128 adapter hashes
from METH-24/19, reconstructs the 24 METH-27 prompts (manifest SHA-256
`94abd7544d4f887e476286774b2ce51bbc0c5c68f079907b88b3984589d62ec6`),
and observes every layer input on the **exact original route**.
The 16 balanced leaves of eight rows per layer are made by recursive
leading-singular-direction splits of frozen router weights. They use
no prompt activations or labels for fitting. The candidate index
scores all 16 leaf means, then exact-scores 32 or 64 selected rows.

| Frozen candidate budget | Row-dot equivalents vs 128 exhaustive | Exact top-4 IDs included | Full top-4 sets matched | Mean missed gate mass |
|---|---:|---:|---:|---:|
| 4 leaves, 32 candidates | 48 (37.5%) | **62.484%** | **13.703%** | 0.3433 |
| 8 leaves, 64 candidates | 80 (62.5%) | **85.848%** | **53.956%** | 0.1245 |

The audit covers **6,144 prompt positions × 24 layers = 147,456
input-layer cases** and 589,824 exact selected IDs. With 64
candidates, ID inclusion is 85.586% code, 86.782% prose and 85.176%
technical; per-layer inclusion ranges 78.837–93.424%. Both arms
miss the frozen ≥99.9% pooled ID, ≥99% full-set and ≥99% per-category
gates by large margins. Mean missed gate mass is not an end-to-end
loss or task metric; it quantifies how much exact selected weight the
index omits on these inputs. Group assignments, per-layer/category
counts and exact/approximate route-stream SHA-256 values are retained
in the [machine result](meth29_balanced_router_index_result.json),
SHA-256 `249836830ed41e77b9e2c6036292e71838359221e430eac29295e6bb7944d502`.

The first invocation ended **before any prompt or route measurement**
because the runner looked for category counts in the audit module
instead of the selection module. Only that module reference was
repaired; grouping, candidate budgets, prompts and gates stayed
fixed. The successful RTX 3060 diagnostic took 57.5 s, peaked at
2.275 GB allocated GPU memory and ended at 3.330 GB RSS. It measures
GPU route fidelity and counts row-dot work. It does **not** measure
CPU lookup, selected-expert LUT time, memory traffic at large E,
language-quality effects of route replacement, or trained capacity
at E>128.

**Decision:** reject this *post-hoc mean-centroid grouping* before C
implementation. Its failure at trained E128 makes a 10B/100B speed
extrapolation invalid. A hierarchy trained to predict fine-route
membership, or another bounded index, needs a new prospective
recall/quality audit. It must still coexist with a generative repair
and with distinct learned experts as E grows.
