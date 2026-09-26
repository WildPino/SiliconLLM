# METH-38: candidate omissions and rescore numerics both matter

The [prospective diagnostic](METH_38_CANDIDATE_CAUSE_PROTOCOL_20260926.md)
reused the METH-37 prompts to separate missing candidates from
the exact-rescore implementation. The stored R8 core, adapter
and rank-64 int8 sketch were unchanged. Exact and 64-candidate
top-1 streams reproduced all 24 METH-37 saved streams exactly
before the 96/128-candidate arms were interpreted.

| Exact candidates | Exact top-4 IDs included on own trajectory | Full top-4 sets matched | Next-token top-1 agreement vs exhaustive model |
|---:|---:|---:|---:|
| 64 | 589,451/589,824 = 99.937% | 99.747% | 6,043/6,144 = 98.356% |
| 96 | 589,815/589,824 = 99.998% | 99.994% | 6,104/6,144 = **99.349%** |
| 128 | 589,824/589,824 = **100%** | **100%** | 6,123/6,144 = **99.658%** |

The 96-candidate arm passes its diagnostic ≥99% top-1 and
≥99.9% route-ID inclusion thresholds on these reused prompts.
However, the 128-candidate apparatus arm **fails** its fixed
≥99.99% top-1 threshold despite an exact top-4 set in every
input-layer case. It changes 21 next-token top-1 outputs even
when no expert is omitted. The shortlist path computes fine
candidate scores by per-element multiplication and summation,
whereas the exhaustive reference uses `F.linear`. Their numeric
gate differences and downstream trajectory changes are therefore
an active confound; 96 candidates cannot be promoted from this
result.

The [machine result](meth38_candidate_cause_result.json),
SHA-256
`a14e5a04475ffac972a45ad405e7382bcbe883ae61ed07f7c07faedc9132bbe4`,
retains per-layer counts. Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth38_candidate_cause.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth38_candidate_cause_result.json`.
The RTX 3060 audit took 88.81 s, peaked at 2.281 GB allocated
GPU memory and ended at 3.163 GB RSS. No T4 was used.

**Decision:** first isolate exact-rescore arithmetic using an
all-candidate parity control, then test a deployable rescore
implementation. The 96-row budget is a promising diagnostic,
not yet a quality-valid router or a CPU-speed result. This
result does not change the parent model's generation failure.
