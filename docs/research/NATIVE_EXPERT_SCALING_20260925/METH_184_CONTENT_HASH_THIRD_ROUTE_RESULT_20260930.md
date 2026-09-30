# METH-184: fixed content-score/hash mixing fails training load

**Decision: reject this third-tier route rule under the frozen screen.**
None of the five preregistered Gumbel-noise intensities passes both
training raw and chat. The runner therefore stopped before evaluating the
source-separated METH-150 documents. It did not train B factors or
measure quality or native CPU cost.

The [protocol](METH_184_CONTENT_HASH_THIRD_ROUTE_PROTOCOL_20260930.md)
was committed at `fc31ee9` before the
[runner](../../../benchmarks/native_expert_scaling/meth184_content_hash_third_route.py)
at `97ee29a` and before execution. The exact METH-126 E1280 source bank,
METH-135 third projection/keys and METH-107 child checkpoint were
verified by SHA-256 through the METH-136 loader. Eight BF16 prompts
gave zero logit difference between the source teacher and CPU-bank
control. A vectorized/single-scalar hash self-test passed. The rule
used the same frozen E1280 parent IDs/top-four gates, normalized all
ten seeded content scores per selected child and added deterministic
hash Gumbel noise. There was no special structural slot.

| Training cell and beta | Worst candidate/control load ratio (≤1.25) | Worst hot-parent share (≤25%) | Minimum coverage (≥4,000) | Lowest score advantage (≥0.05) | Lowest argmax agreement (≥15%) |
|---|---:|---:|---:|---:|---:|
| raw, 0.5 | 4.651× | 81.78% | 7,287 | 1.352 | 56.43% |
| raw, 1 | 3.011× | 47.47% | 7,644 | 0.893 | 33.71% |
| raw, 2 | 1.923× | 30.12% | 7,812 | 0.484 | 20.45% |
| raw, 4 | 1.439× | 20.91% | 7,837 | 0.246 | 14.59% |
| raw, 8 | 1.314× | 18.85% | 7,861 | 0.122 | 12.07% |
| chat, 0.5 | 6.879× | 100% | 7,413 | 1.331 | 55.20% |
| chat, 1 | 4.409× | 100% | 7,934 | 0.875 | 32.36% |
| chat, 2 | 3.455× | 100% | 8,207 | 0.462 | 19.38% |
| chat, 4 | 3.369× | 100% | 8,255 | 0.223 | 13.56% |
| chat, 8 | 3.418× | 100% | 8,285 | 0.107 | 11.01% |

These extrema represent different layers and are diagnostics, not a
single combined layer. At beta 8, raw passes the share, coverage and
score-advantage gates in every layer but misses the load-ratio gate in
two layers and argmax-agreement in every layer. Chat fails both load
gates in all 24 layers at beta 8; 17 layers have a worst hot-parent
share above 99%. Several shares are exactly 100%, while others are
`256/257` or `256/258`, consistent with repeatedly presented chat
tuples. This is a mechanistic inference from the counts: METH-149/150
already measured recurrent token/previous/position tuples in these
draws, and the stateless hash maps an identical tuple, child and layer
to the same noise vector. It does not prove that every hotspot comes
from one tuple. Large beta also erases much of the score preference.

The [raw per-layer result](meth184_content_hash_third_route_result.json)
has SHA-256 `6e2ada10e4ec679aab071277768db3c8a90114f511ccf2a0503bf2bedb6d464`.
Run from the repository root with:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth184_content_hash_third_route.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth184_content_hash_third_route_result.json
```

The local RTX 3060 run took 117.015 seconds with six host threads,
2.945 GB peak allocated GPU memory, 4.150 GB process RSS and less than
1 GB new disk. No T4 was used. The source-separated manifest was
hash-checked at binding, but no model inference used its documents;
those cells are absent by the frozen stop rule. This result cannot
claim external load generalization.

The failure sharpens the next decision. Stateless token/previous/
window-position noise does not solve repeated-context concentration
while preserving the specified content signal. A shared recurrent
path can account for those repeated choices, as METH-150 tried, but
the later METH-175 shared/hash training did not produce route-specific
function or held-out quality improvement. Repeating a noise-strength
sweep or the same shared-B recipe is not justified. A future large-E
candidate needs a new specialist-learning signal and must count any
shared structural slot honestly; meanwhile the pretrained-to-native
core and same-artifact 50 tok/s gap remain independent critical work.
