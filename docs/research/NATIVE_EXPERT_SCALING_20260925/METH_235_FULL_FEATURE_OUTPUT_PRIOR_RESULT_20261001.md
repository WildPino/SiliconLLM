# METH-235: continuous output projection passes, row-Q8 derivative fidelity fails

Frozen at `a9fb91c`; session18122 completes,exit0. Full nonlinear basis
derivatives/autograd, FP64 QR rank/conditioning, coefficient growth,
unrounded reconstruction, center values, all16 distinct coefficients and
every snapshot readback pass. Source controls and fixed route labels/counts
replay exactly. No affine branch is added.

Stored Jacobian distortion fails in **all16 priors**:1.0542%–1.1780%,above
the frozen1% bound. The unrounded FP64 projection reconstructs the source
derivative within numerical error; that claim cannot be extended to the
encoded row-Q8 coefficients. Maximum condition11.7816 and relative
coefficient change.0117945 pass,so rank/large continuous changes are not
the failing gates. Full FP32 output autograd checks at0,15 also pass.

Pooled original-source fit-input SSE/energy=.0002437703 (0.024377%),within
1% and below unchanged source-down baseline.0003109190 (0.031092%).
These fit-input controls do not override the failed stored-derivative gate.
No captured output-label fitting or validation/new-source score is run.

**Decision: stop this one-shot output derivative projection and row-max-Q8
encoding before conditional fitting.** Not a universal output-only prior
impossibility. A continuous prior followed by independent weight rounding
has not preserved the requested slope accuracy in this experiment.

[Raw result](meth235_full_feature_output_prior_result.json),SHA256
`132013d3991622143d268090015900abfcbb88e22d5799eb4df1073b4a97b140`.
Local ignored snapshot
`results/native_expert_scaling/meth235_layer12_full_feature_output_priors.safetensors`,
83,596,004bytes,SHA256
`029e459061360a492c74c1b0bc4ab96ea7695fa4f25fdd33197f9c918e2ea5cd`.
Runtime23.000s after imports,RSS1.699GB,GPU peak804.065MB,local3060,no T4.
All16 audits and derivative tests/hash controls retained; no job active.

Next: [METH-236](METH_236_OUTPUT_SCALE_JACOBIAN_BOUND_PROTOCOL_20261001.md)
analytically bound scale-only repair at the fixed codes. Its source-Jacobian
metric/conditional output basis differs from the stopped group64 global
weight-MSE scale recipes, but a bound failure closes scale-only repair for
these codes without fitting or tolerance changes.
