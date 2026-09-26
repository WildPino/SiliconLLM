# METH-18 result: residual amplitude drives the fresh code loss

**Decision:** carry a fixed half-amplitude candidate into a *new*
document audit. METH-18 reused the METH-17 documents for diagnosis, so
its apparent quality gains cannot promote a checkpoint. The
[prospective diagnostic](METH_18_RESIDUAL_SCALE_ROUTE_DIAGNOSTIC_PROTOCOL_20260926.md),
[machine record](meth18_residual_scale_route_diagnostic.json) and
[runner](../../../benchmarks/donor_adaptation/s1/meth18_residual_scale_route_diagnostic.py)
bind the arms and inputs.

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth18_residual_scale_route_diagnostic.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth17_fresh_document_manifest.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth18_residual_scale_route_diagnostic.json
```

The script reconstructed the METH-17 selection, checked the manifest,
Qwen2.5-0.5B donor and tokenizer, and loaded the same update-1024
E128/top-4 checkpoint. With all expert output factors at 1.00, it
reproduced the earlier pooled donor/student BPB to much better than
1e-5 and repeated 24/24 code-document loss signs. Only the output
factors' scalar or the router-row permutation changed. The RTX 3060
run took **139.046 s**, peaked at **2.549 GB allocated GPU memory**
and ended at **3.202 GB process RSS**.
After the run, the runner's pre-existing truthy METH-17-result hash
check was tightened to equality against its committed SHA-256
`c9e51c9736904c1674c755ef528b26ea656d1d33dfaf35af10d51c5c5e2e52a2`;
the scoring path and stored measurements did not change.

| Arm | Pooled delta to donor | Code | Prose | Technical |
|---|---:|---:|---:|---:|
| Trained, factor 0.25 | **−0.003813** | +0.000203 | −0.003626 | −0.012218 |
| Trained, factor 0.50 | **−0.002339** | +0.005109 | −0.003924 | −0.014067 |
| Trained, factor 1.00 | **+0.013215** | +0.029126 | +0.003550 | +0.000724 |
| Permuted router, factor 1.00 | **+0.035510** | +0.029778 | +0.050856 | +0.016284 |

At factors 0.25/0.50/1.00 respectively, **12/20/24 of 24** code
documents worsen relative to donor. The trained factor-1.00 router
beats its row-permuted null by **+0.022295 pooled BPB** but only
**+0.000652 on code**; the corresponding prose and technical gaps are
+0.047306 and +0.015560. The null is destructive and does not measure
an optimal route. Internal development BPB is donor **0.840579**,
factor-0.25 **0.824680**, factor-0.50 **0.818605**, factor-1.00
**0.824196**. This internal set was used in METH-16 and supplies no
independent validation.

The code loss follows residual amplitude while the trained router
has little apparent code-specific advantage over a random row
permutation. That supports a bounded residual contribution as the
next inexpensive *candidate*, without claiming a learned
domain-sensitive gate or attributing the entire failure to one
component. This diagnostic does not alter the underlying dense BF16
donor, the exhaustive E128 route, or native speed. It also does not
establish that a factor picked after seeing METH-17 generalizes.
