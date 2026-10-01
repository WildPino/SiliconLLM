# METH-250: inherited output directions miss most residual error

Frozen d81e3cc, session61839 exits0. All16 actual parent fit SSEs/labels
replay exactly; eigen, inverse, projection and physical readback controls
pass. This is fit-only evidence; no child fit or validation target was read.

| Share of centered actual-parent error in a stored rank32 space | Fraction |
|---|---:|
| Inherited METH-247 left basis | 2.9418% |
| BF16 basis derived from actual-parent residual covariance | 15.4024% |

Both prospective feasibility gates pass: the stored new space captures
at least10% and exceeds old captured energy by at least10 percentage
points. Arbitrary per-state corrections plus a free full mean intercept
could remove16.0550% of total fit residual; this is an optimistic bound,
not a learned feature mapping or quality prediction.

Actual basis checkpoint4,587,888bytes includes diagnostic FP64 and BF16
tensors, SHA256
`c457466afb56ab19ab5f7872309a8e73ef51db5e726e6cceaff51cafbe4095a0`.
[Raw result](meth250_residual_output_space_result.json) SHA256
`8220ad28bd577730f78bb02827e8b09f66b9d2c010a453da009bbdb69a920bfb`.
Measured after imports22.609s, RSS2,056,511,488bytes, GPU peak2,134,826,496;
local RTX3060/six threads, no T4/download/native timing. Wall time before
imports completed is outside that measured runner interval.

## Licensed next comparison

Freeze a continuous matched E16/E160 pilot in the new basis. Both arms
must have the same extra output directions; comparing expanded E160 to
the old rank32-only parent would confound expert count with active rank.
First fit16 zero-prior residual mappings in the new space over their
whole parent supports, retaining the original parent function. Then retain
each complete newly adapted parent and fit its10 private residuals in
the same space. Hold source/key/rank/variance/tau fixed. Same fit/numerical
prerequisites and consumed count/rotated/prior10% gates, no tuning.
No encoded/native bank, useful large-n, full quality/rate or donor-scale
transfer is established by this fit projection.
