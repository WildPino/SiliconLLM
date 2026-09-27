# METH-49: row-scaled int8 also fails ranking retention

The [frozen protocol](METH_49_INT8_ROW_FACTOR_BANK_PROTOCOL_20260927.md)
changed only the int8 scale granularity from METH-48. Each rank-8 input
factor row of `a` and each eight-value output factor row of `b` now has
its own fp32 scale. The [runner](../../../benchmarks/donor_adaptation/s1/meth49_int8_row_factor_bank.py)
verified all bound source hashes and saved tensor readback, then matched
all 12 original METH-47 greedy streams before evaluating the quantized
factors with the donor and router unchanged.

| Fixed measure | Original → row-int8 result | Gate |
|---|---:|---|
| Pooled document BPB | 1.355188 → 1.355178; Δ −0.0000104 | ≤+0.002, pass |
| Worst category BPB Δ | technical +0.0001757 | ≤+0.002, pass |
| Prompt top-1 agreement | **2,002/2,091 = 95.744%** | ≥99%, **fail** |
| Greedy EOS / repeated 8-gram | 12/12 → 12/12 EOS; 0 → 0 loops | pass |
| Exact greedy continuations | 4/12 | reported, not gated |

The finer scales reduce relative squared factor reconstruction error
from METH-48's 7.61e-5 to 5.13e-5 for `a`, and 8.56e-5 to 1.11e-5 for
`b`. Yet prompt top-1 and exact continuations are worse on this set.
The ranking failure cannot be repaired by claiming a near-zero average
BPB penalty. It also does not establish that every row scale is harmful:
the benchmark is a single fixed checkpoint and 12 previously viewed
documents, and small numerical differences can change later routing and
greedy trajectories. No scale choice was selected after these outcomes.

The local [packed artifact](../../../results/native_expert_scaling/meth49_e128_factor_int8_row.safetensors)
is 55,157,296 bytes, SHA-256
`f3039decaec0600ccae9f93b6c11cb063298a53b469b39e2cbcdba92aceef196`.
Tensor payload is 430,848 bytes per expert across 24 layers:
344,064 code bytes and 86,784 scale bytes. An unchanged-geometry E27,355
bank projects to 11,785,847,040 payload bytes; E273,547 projects to
117,857,177,856 bytes. These are **hypothetical** storage projections,
excluding donor, router/index and runtime workspace. At top-4,
selected factor payload is 1,723,392 bytes/token independent of E,
before cache and LUT construction. Neither 10B nor 100B bank was trained
or CPU benchmarked here.

The [machine result](meth49_int8_row_factor_bank_result.json), SHA-256
`822d07450f3b9cb2241fea5f19994494417d648e4ce39a7d7f9139efb50b2e4c`,
holds paired document scores and continuation token streams. Command:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth49_int8_row_factor_bank.py --artifact results/native_expert_scaling/meth49_e128_factor_int8_row.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth49_int8_row_factor_bank_result.json
```

The measured run took 104.390 s on the local RTX 3060, peaked at
2.437 GB GPU allocation and 4.171 GB RSS; no T4 was used.

**Decision:** reject this row-scale int8 export under the fixed top-1
gate. Stop post-hoc scale-granularity tuning on this viewed set. Preserve
an exact or high-precision packed-factor anchor, and next test where the
small factor perturbations alter route and token rankings before choosing
joint quantization-aware adaptation. Bounded CPU routing at growing,
distinct E and source-grounded semantic quality remain separate gates.
