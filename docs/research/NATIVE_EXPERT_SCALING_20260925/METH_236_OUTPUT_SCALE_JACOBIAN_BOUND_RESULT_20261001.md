# METH-236: scale-only repair cannot meet the source-derivative gate

Frozen at `1f65aed`; session90041 completes,exit0. All bindings and source
gradient replays pass; each exact one-dimensional scale least-squares
normal equation passes. Optimal scales are positive,so the unconstrained
minimum is also the minimum under the codec's positive-scale requirement.

**All16** parents' FP64 optimum-scale Jacobian errors remain>.01. Worst
lower bound=.011757464 (1.17575%),against original.011780391. Rounding the
optimal scales to FP32 does not resolve the deficit. Thus scale-only repair
with these actual code rows cannot pass; no calibrated snapshot/function
fit/validation is opened. This does not exclude changed code assignment,
quantization-aware projection or another representation.

[Raw result](meth236_output_scale_jacobian_bound_result.json),SHA256
`d6febbb46d8387935a6c0653ad3df171588a8b0a6ae294b1a457a3b72d478a23`.
Runtime2.390s after imports,RSS1.361GB,GPU peak280.116MB,local3060,no T4.
All16 analytic bounds/scale hashes/normal residuals retained.

Next: [METH-237](METH_237_QUANTIZED_OUTPUT_PROJECTION_PROTOCOL_20261001.md)
changes the projection to incorporate the actual encoded coefficients
between fixed cycles. The frozen four-cycle final outcome,not a best
intermediate, must still pass the original stored1% derivative gate.
