# METH-237: encoded feedback improves slopes slightly, all stored priors still fail

Frozen at `57b4c09`; session96785 completes,exit0. Exactly four cycles use
the actual decoded coefficients between source-Jacobian projections and
row-Q8 encodings. All cycle histories retained; only final cycle judged.
All source/native/route controls and original fit SSE replay exactly,
all9 unchanged snapshot control tensors match, every tensor reads back.
Protocol originally said11 controls; that was a prose counting error,
corrected to9 without changing the dynamic control selection or gates.

Every continuous cycle/source-center/autograd and coefficient-growth gate
passes; all16 final code+scale pairs remain distinct. No repeated encoding
hashes occur within any parent's five initial/intermediate states. This
does not prove convergence or license extra iterations.

**All16** final stored derivatives remain above1%:1.03982%–1.14303%.
Worst falls from METH-235's1.17804% to1.14303%,but fails the unchanged gate.
Pooled fit-input source function SSE/energy=.0002535480 (0.025355%),above
the replayed old.0002437703 (0.024377%),both within the absolute1% gate.
Maximum final coefficient distance/base norm=.0134595,well below.25.
Initialization cuBLAS context warning retained; no implementation failure.

**Decision: stop this fixed four-cycle quantized-feedback recipe.** No
extra-cycle/best-history selection,conditional fit,validation/new-source
score,trained-bank native timing or full-model quality is opened. No
universal quantization-aware compiler impossibility follows.

[Raw result](meth237_quantized_output_projection_result.json),SHA256
`304c15003a01cbd81583fb015298b98602b78bdc7fa21cced54bcc2edb6ce982`.
Local ignored snapshot
`results/native_expert_scaling/meth237_layer12_quantized_output_priors.safetensors`,
83,595,988bytes,SHA256
`36b59d69039e577498acd94d5c7320b4313717d81caa9a7c90096723c0653aec`.
Runtime20.656s after imports,RSS1.589GB,GPU peak776.433MB,local3060,no T4.
All64 cycle audits,coefficient hashes and source-function controls retained.

Next: [outlier-split output precision proposal](OUTLIER_SPLIT_OUTPUT_PROPOSAL_20261001.md),
change encoded precision/row dynamic range rather than retry cycles/scales.
Qualify actual codec and CPU cost before reusing any source-prior mechanism.
