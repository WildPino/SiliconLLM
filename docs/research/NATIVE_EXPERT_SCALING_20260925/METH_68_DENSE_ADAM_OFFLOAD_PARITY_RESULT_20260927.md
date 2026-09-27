# METH-68: CPU-offloaded factors reproduce dense E128 AdamW for three steps

**Decision:** pass the three-step numerical apparatus gate. With
identical METH-55 E128 initialization and input tensors, CPU-resident
factors plus dense-semantics AdamW reproduce the dense GPU reference's
routes, BF16 outputs and A/B/router gradients exactly in all three
steps. The largest post-update parameter difference is
4.657×10⁻¹⁰. This removes an inherent numerical incompatibility of
factor offload as an explanation for METH-66's E128 gap; METH-66 used
a different initialization, sparse row-age optimizer and draw schedule.

The [protocol](METH_68_DENSE_ADAM_OFFLOAD_PARITY_PROTOCOL_20260927.md)
was committed before the [runner](../../../benchmarks/donor_adaptation/s1/meth68_dense_adam_offload_parity.py)
executed. The [raw result](meth68_dense_adam_offload_parity_result.json),
SHA-256 `f82a9ab92edb062ade51fd42b0f232e4e902bf0e35ae1b54e7a71b3d52432fa0`,
records each step's errors and resource use.

| Step | Unique selected expert rows | Max forward error | Max A/B/router gradient error | Max post-update A/B/router error |
|---:|---:|---:|---:|---:|
| 1 | 77 | 0 | 0 | 1.164×10⁻¹⁰ |
| 2 | 83 | 0 | 0 | 2.328×10⁻¹⁰ |
| 3 | 74 | 0 | 0 | 4.657×10⁻¹⁰ |

Both arms use packed product-key router parameters, top-four selection,
BF16 factor arithmetic and the METH-55 seeded A/B/key tensors. The CPU
optimizer decays moments and updates every row, including unselected
ones with previous momentum, matching dense AdamW rather than the
row-age semantics of METH-66. It consumes only selected-row gradients
returned from GPU. Peak allocated GPU was 70,846,464 bytes and end
RSS 2,293,325,824 bytes for this one-layer synthetic test.

The next necessary check is a complete donor-backed 16-update replay
with the *same* METH-55 sample draws and initialization, comparing its
terminal factor/router values and output metrics to the frozen METH-55
checkpoint. Only after that should a newly frozen E1280 recipe be
evaluated on disjoint development and external content. This synthetic
test alone says nothing about useful large-E quality or native rate.
