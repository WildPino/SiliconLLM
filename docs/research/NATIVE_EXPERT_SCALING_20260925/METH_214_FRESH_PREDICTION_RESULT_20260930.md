# METH-214: saved Q8/exact-head core fails fresh pooled ranking

The fixed saved-core candidate is tested once on the frozen 24 new
source/fragment-disjoint METH-213 documents and prompts. Its loader
again supplies all 290 parameters from all 364 artifact tensors.

| Metric | BF16 donor | BF16 centered E1280 | Saved compact + E1280 |
| --- | ---: | ---: | ---: |
| Pooled document BPB | 1.194418 | 1.189030 | 1.188890 |
| Pooled donor-top1 agreement | 100% | 95.506% | 94.215% |
| Code agreement | 100% | 96.949% | 96.000% |
| Prose agreement | 100% | 95.950% | 95.093% |
| Technical agreement | 100% | 93.797% | 91.835% |

Pooled agreement loses 1.291 percentage points against the frozen
<=1-point limit. Every category remains within its <=2-point limit,
although technical loss is 1.962 points. Pooled and category BPB
gates against both controls pass. The fixed K64 shortlist omits no
full-head choices and exact-row reranking mismatches none on these
prompt states. These passes do not override the failed ranking gate.

Decision: stop generation, PIQA and blind review. The prepared
METH-215 generation runner remains unexecuted and enforces the passing
prediction prerequisite. No threshold, candidate or source replacement
is made after this outcome. This cohort is now consumed diagnostic
data and cannot adjudicate a follow-up compact candidate as fresh.

Local RTX 3060: 86.640 seconds, 3.764 GB end RSS, 4.210 GB peak
allocated GPU. [Raw result](meth214_stored_core_fresh_prediction.json)
SHA256 `593ada3d37e63d243b6e1ddd133bb8ed0c34a38512e1aadc066236dbc0dc5a90`.
The core remains a physical, loadable development candidate and fails
this independent prediction gate. Native quality/rate promotion is
not licensed. Continue the separate changed local-key expert-count
geometry on the original BF16 E1280 source; compact-core recovery
requires a changed, train-validated representation or adaptation.
