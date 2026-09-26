# METH-41: C96 route passes the independent E128 quality gate

The [prospective protocol](METH_41_C96_INDEPENDENT_ROUTE_PROTOCOL_20260927.md)
froze the METH-36 rank-64 int8 sketch, **96** exact candidate
rows and elementwise candidate-only fine rescoring before reading
new model outcomes. The [manifest](meth41_fresh_c96_manifest.json),
SHA-256 `8fff79f2502612d1f49083d1c48dd36d2718acec6335d676ccb5b8ded726365f`,
contains 8 code, 8 prose and 8 technical documents (98,280 bytes).
The selector excluded METH-17/19/25 source IDs and Qwen/prior
fragment overlap. Code and technical spans are from pinned Git
revision `9fb7791`; prose comes from two pinned external held-out
corpora. The available external code and technical snippets
failed the overlap preflight and were not used. Thus category
results have different source distributions and cannot be read
as one uniform corpus. The source documents were not used to
choose rank64 or C96.

The runner verified source, stored R8 core, E128 adapter and
stored sketch hashes/shapes, reconstructed all 169 core matrices
and 121 controls, preserved the tied head, and reproduced all
24 saved METH-37 exact-route top-1 streams before scoring the
new prompts. On the local RTX 3060, the model phase took
**364.78 s**, peaked at **2.843 GB** allocated GPU memory and
ended at **3.232 GB** RSS. No T4 was used.

| Category | Docs | BF16 donor BPB | Exact R8+E128 BPB | C96 BPB | C96 − exact |
|---|---:|---:|---:|---:|---:|
| Code | 8 | 0.670263 | 0.679006 | 0.679043 | +0.000036 |
| Prose | 8 | 1.093650 | 1.090747 | 1.090772 | +0.000025 |
| Technical | 8 | 1.637634 | 1.648124 | 1.648239 | +0.000115 |
| Pooled | 24 | 1.133849 | 1.139292 | 1.139351 | **+0.000059** |

The category-stratified 20,000-draw bootstrap one-sided 95%
upper bound for C96-minus-exact BPB is **+0.000198**, below
the frozen +0.002 limit. C96-minus-original-donor pooled BPB
is **+0.005503**; code is +0.008780, prose −0.002877 and
technical +0.010605, all within their separate limits. On
24×256 fixed prompt positions, C96 next-token top-1 agrees
with exact on **6,099/6,144 = 99.268%**, above ≥99%.

| Repeated 8-gram ≥3× on 128-token greedy continuations | Exact | C96 |
|---|---:|---:|
| Code | 5/8 | 5/8 |
| Prose | 7/8 | 6/8 |
| Technical | 4/8 | 4/8 |
| Pooled | **16/24** | **15/24** |

Neither arm produced an early non-EOS continuation under 16
tokens; both reached EOS on one prompt. Relative generation
passes the frozen +1 allowance in every category and pooled.
The smaller loop count is a sample observation, not a general
improvement claim. **15/24 looping C96 continuations still fail
the parent's absolute usefulness screen** (METH-27), so the
complete R8+E128 artifact remains unpromoted.

The [raw paired result](meth41_c96_independent_route_result.json),
SHA-256 `c345295593fac6568102a521af674e64b4ad4762695ffad052d82f45ae1c5cea`,
retains per-document nats, top-1 streams and every generated ID
and decoded continuation. An independent readback recomputed
all BPB values, 6,099 matches and repetition counts from those
rows. Reproduction command:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth41_c96_independent_route.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth41_c96_independent_route_result.json
```

**Decision:** C96 passes its independent *E128 route-replacement*
loss, ranking and relative generation gates, as well as the
donor-relative document screen. The frozen C64 rule remains
rejected by METH-37. C96 is a quality-valid E128 route component
for further native investigation, but not a useful full model,
large-E quality proof, or CPU speed result. METH-40's synthetic
E273,547 scan already takes 18.453 ms/token with six threads,
leaving 1.547 ms for candidate rescoring, selected-expert LUT,
core and overhead inside the ≥50 tok/s target. The next decisive
work is a generation repair and a large-E router design that
avoids that exhaustive traffic, followed by a same-artifact
`engine.c` quality/rate test.
