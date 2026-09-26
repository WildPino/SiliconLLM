# METH-21 result: PIQA task retention passes on the exported adapter

**Decision:** the half-amplitude Qwen2.5-0.5B E128 adapter passes the
prospective **PIQA task-retention** gate. Its primary accuracy is
0.653 percentage points below donor, inside the fixed 2-point
point limit and 5-point lower-confidence limit. This establishes
relative retention on this one labelled task; METH-20's high
absolute repetition and the native cost problem remain open. The
[protocol](METH_21_HALF_ADAPTER_PIQA_PROTOCOL_20260926.md),
[token manifest](meth21_half_adapter_piqa_manifest.json),
[all 1,838 paired outcomes](meth21_half_adapter_piqa_result.json),
and [runner](../../../benchmarks/donor_adaptation/s1/meth21_half_adapter_piqa.py)
bind the evidence.

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth21_half_adapter_piqa.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth21_half_adapter_piqa_manifest.json --manifest-sha256 58acfd5882ae5d7e155bdd0ae826728aba510dc6ef4f599b7b807161a498531d --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth21_half_adapter_piqa_result.json
```

The run checked exact donor/tokenizer/adapter identities, scored
**all 1,838** pinned PIQA validation rows with the existing
EOS + separately tokenized question/answer convention, and recorded
each answer's total and mean suffix NLL. The largest sequence was
245 tokens. Donor and student used the same RTX 3060 BF16/SDPA
model instance, switching only the loaded experts. Runtime was
**388.469 s**, peak allocated GPU memory **1.477 GB**, end RSS
**2.766 GB**. After the run, an explicit equality guard between
model and tokenizer EOS IDs was added to the runner; local config
inspection confirms **151643 = 151643**. It changes no scoring path
or stored result.

| Lower-NLL choice | Donor correct / accuracy | Adapter correct / accuracy | Difference |
|---|---:|---:|---:|
| **Mean suffix NLL (primary)** | **1,298 / 70.620%** | **1,286 / 69.967%** | **−0.653 points** |
| Total suffix NLL (sensitivity) | 1,293 / 70.348% | 1,288 / 70.076% | −0.272 points |

There are **39** donor-correct/student-wrong and **27**
donor-wrong/student-correct primary-choice items. A 20,000-draw
item-paired bootstrap, seed 212121, gives a one-sided 95% lower
bound of **−1.360 percentage points**, above the frozen −5-point
limit. The −0.653-point point delta is above the −2-point limit.
The mean gold-option token NLL changes from **3.422084** to
**3.423011**. These are paired measurements, not a claim that the
adapter improves PIQA.

PIQA is one English physical-commonsense multiple-choice task. The
source files were used in a prior GigaChat research line but not to
train or choose this Qwen residual adapter. The result cannot
establish code usefulness, broad task transfer, fluent generation,
low-bit compatibility, engine C fidelity, ≥50 accepted tok/s, or
large-E scaling. It keeps this exact exported candidate eligible for
the next native/core-cost investigation while those gates remain
open.
