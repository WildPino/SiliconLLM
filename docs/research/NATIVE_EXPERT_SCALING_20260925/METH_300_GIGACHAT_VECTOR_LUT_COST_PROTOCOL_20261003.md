# METH-300: complete cost of the full-feature two-coefficient LUT proposal

## Question, reused evidence and prospective decision

Does the [proposed representation](FULL_FEATURE_VECTOR_LUT_PROPOSAL_20261002.md)
leave a plausible complete GigaChat path under the existing 560MB/token
addressed-weight design allotment (14ms at the 40GB/s hardware yardstick)?
Freeze this protocol and implementation before running the cost ledger.
The 20ms complete-token target still includes compute, routing, KV and state.
The yardstick is not a measured native latency or physical DRAM bound.

Reuse the actual 414-tensor source/Q4 GGUFs and 01/04 ledgers, not a generic
donor-port restart. 06/08/09 low-bit quality failures, 298 rank192 failure
and 299 omission failure remain closed. Old 34/130/132 child-factor LUT
controls and 198 synthetic timing cannot qualify source coefficient palettes.
No weights or scientific input captures will be read; no donor inference,
download, GPU/T4, optimizer or native benchmark is started by this screen.

## Exact representation and complete inventory

Every consecutive two original coefficients receive one U8 index into a
225-entry non-Cartesian FP32 two-dimensional palette, with one positive
FP32 scale per original output row. Preserve all original rows/channels.
Price MLA, all selected routed projections, shared FFNs, dense first FFN,
and complete untied head. Keep embeddings, all F32 routers, norms and
biases at their actual Q4 baseline types. This is not an exported format:
normalization, palette optimization and error are unmeasured.

Use a bounded standard-library GGUF-v3 metadata/descriptor reader. Check
file sizes, topology, exact source/Q4 shape/name agreement, 414 names,
type/block alignment, contiguous payload spans and header+payload size.
Freshly hash only the metadata/descriptors/alignment; report previous full
hash provenance explicitly. Full payload hashes are NOT rechecked here.
Reconcile every source/Q4 organ with the immutable 01/04 records and the
1,628,078,080 active large-matrix element audit. No matrix data is read.

Count these source-derived queries separately:

- Routed gate/up inputs shared among four selected macro experts for each
  projection palette; four different routed down inputs per layer.
- Shared gate/up/down inputs priced in full; dense first FFN priced in full.
- MLA K-B [128,512,32] has 32 distinct head query inputs; MLA V-B
  [512,192,32] has 32 distinct head attention-result inputs. One palette
  per projection tensor can span its banks, but inputs cannot be equated.
- Q, KV-A and attention output projections, plus the complete output head.

Price codes, row scales and all palettes conservatively addressed once;
separately report FP32 query table writes, centroid/input operands, one
lookup/add and one FP32 table gather per code, and row scaling. These are
logical operations/accesses, never claimed physical DRAM transactions.
No cache residence, implicit zero-cost head, AVX gather rate or native speed
is assumed. Exclude alignment, activation accumulators, KV and repeated
accesses; these omissions can only license a later explicit implementation.

## Frozen scenarios, controls and gates

1. Routed-only palettes; all other Q4 components remain unchanged.
2. Complete source projection palettes, shared within each tensor's banks.
3. Complete optimistic common palettes: merge Q/KV-A palettes within a
   layer; merge routed/shared gate/up palettes and their identical normalized
   FFN input; merge routed/shared down palettes but keep all five inputs;
   merge dense gate/up palettes. This stronger sharing is an unvalidated
   quality constraint and saves construction/palettes, not code/scale bytes.

Each passes cost eligibility ONLY if total addressed weight <=560,000,000B.
If none passes, reject training this exact two-coefficient U8 proposal as a
complete candidate. Do not relax the gate after results or borrow cached
weights, head selectors, earlier quality passes or component rates.

Two diagnostics cannot promote a candidate: an impossible zero-head,
zero-palette lower accounting scenario, and the necessary maximum code
bits/coefficient at full source features and the fixed scales/control bytes.
They identify the transformation still needed, not another measured format.

Price n=64/640/6400 with fixed top4 and shape only: codes/scales in RAM and
the unchanged flat F32 router/bias grow with n, while selected routed bytes
and table construction do not. Only the existing 64 source experts are real.
No additional parameters, capacity or learned routing are instantiated.

## Resources, failures and reproduction

Local CPU metadata arithmetic: expected seconds, hard 120s, 1GiB RSS,
16MiB maximum header, 2MB result. Save any apparatus failure; repair only
its cause without changing scenarios/gates. Results are write-once.
No CPU rate measurement overlaps this work. Source helper and prior result
hashes are pinned in the new script. Preserve every older helper unchanged.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth300_gigachat_vector_lut_cost.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth300_gigachat_vector_lut_cost_result.json
```

Promotion beyond this descriptor screen would need a genuinely changed
complete representation, frozen calibration/control/error/resource rules,
composed independent quality and actual phase60 same-artifact accepted rate.
The full goal and useful RAM-scale expert-count requirements remain open.
