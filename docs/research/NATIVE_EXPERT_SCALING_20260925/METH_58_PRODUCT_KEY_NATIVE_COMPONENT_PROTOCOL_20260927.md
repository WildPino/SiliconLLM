# METH-58: native parity bridge for the quality-screened E128 product-key adapter

**Uncertainty.** METH-57 passed donor-relative external and blind semantic
gates for a PyTorch E128 checkpoint. METH-54 only timed a synthetic CPU route;
METH-52 timed a different replicated expert bank. We do not yet know whether
the *learned* projection, product keys and distinct selected factors can be
serialized and evaluated faithfully in C.

**Decision.** Build a versioned binary containing the METH-56 update-512
rank-64 projection, 8/16 keys and BF16-effective A/B factors for all 24
layers. Reject the export if checkpoint SHA, shape, dtype, finiteness or
readback fails. Implement a C component reader and exact 16-pair top-four
search, gate softmax and selected rank-8 factor path on the exported values.
Use deterministic BF16 activation vectors and at least one actual
hidden-state input per layer from a frozen METH-57 prompt. The initial
component gate is exact unordered top-four ID agreement, per-position gate
probability maximum absolute error ≤1e-5 and selected residual maximum
absolute error ≤0.03 versus the BF16-effective PyTorch wrapper. If it fails,
diagnose numeric step and do not claim C model parity. This component pass
would only permit integration into `benchmarks/phase60/engine.c`; accepted
tokens/s and end-to-end model quality on that native artifact are separate
gates. Preserve all inputs and raw per-layer outcomes.

**Bound inputs.** METH-56 result SHA-256
`06aa8934d849fc3bbe2ae2fb85c5ab4d302c9f3f1c5922138604bf1347273a39`,
checkpoint SHA-256
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`,
METH-57 manifest SHA-256
`9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75`.
Take its first code prompt's first and last non-padding hidden states per
layer, plus two fixed-seed synthetic BF16 vectors per layer, before comparing
C outputs. Save source token positions and activation hash. The model and
tokenizer remain at METH-57's pinned Qwen2.5-0.5B-Instruct revision.

**Cost and stop.** Local RTX 3060 only for hidden-state capture, at most
10.5 GiB peak allocated GPU, 20 GiB RSS and 10 minutes. C export/readback
and parity should take <15 minutes on the local six-core CPU; stop and
retain partial records on overrun. No T4. This probe is a native component
fidelity check, not a claim that the old SSM/SWA `engine.c` already executes
the Qwen donor core or that the E128 factors form a quality-valid LUT bank.
