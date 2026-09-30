# METH-195/196: one deterministic Q8 scale refinement

METH-194's max-absolute grouped-Q8 FFN scales miss the frozen
development top-1 gates by 0.113 pooled and 0.229 code percentage
points, despite near-equal BPB. Before further training or external
evaluation, test exactly one same-byte quantizer. Retain all
METH-193 source/core/head/attention/control bindings and E1280 bank.

For each 64-weight FFN group, start at the METH-193 FP16 scale
`max(abs(weight))/127`, or one for an all-zero group. Perform exactly
three alternating rounds: round/clamp signed codes to [-127,127]
at the current stored FP16 scale; then set the scale to the FP16
rounding of `sum(weight*code)/sum(code^2)`, retaining the old scale
when the denominator is zero. Recompute codes once at the final
scale. Evaluate squared weight error after BF16 reconstruction for
this candidate and the original METH-193 max-scale candidate; select
the lower-error codes/scale independently per group, keeping the
original on a tie. This guarantees no group's reconstructed weight
MSE increases. It does **not** guarantee model quality improves.
Store exactly the same int8 code and FP16 scale shapes, with no
extra per-token metadata. Verify all tensor byte readbacks and FFN
BF16 reconstructions. Ideal core+router cost must remain
559,981,568 bytes/token; physical file size is reported separately.

On the same already viewed 24 METH-121 sources, repeat all four arms
and exact BF16 baseline parity checks of METH-194. Apply its unchanged
pooled/category BPB and donor-top-1 gates. If any fails, stop this
Q8 route before fresh-source quality or full native C rate. If all
pass, freeze a source-disjoint quality protocol next, and still
verify same-artifact native correctness, LUT scaling, ≥50 accepted
batch-1 tok/s, and bytes as n expands.

Use the local RTX 3060, 15 min wall, 10.5 GiB allocated GPU,
20 GiB RSS and 1 GB output cap for each run. No T4 is authorized.
