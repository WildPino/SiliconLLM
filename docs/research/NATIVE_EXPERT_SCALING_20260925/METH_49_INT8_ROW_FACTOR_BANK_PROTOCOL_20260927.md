# METH-49: finer int8 scales for the trained factor bank

**Uncertainty.** METH-48's per-expert-scale int8 factor bank has almost
unchanged document BPB but only 95.839% prompt top-1 agreement with the
original METH-47 checkpoint. Test whether its one scale over 7,168 factor
values is the avoidable source of ranking drift. This changes scale
granularity only; codes remain signed int8 and the donor/router, checkpoint,
data, generation rule and evaluation gates stay fixed. The METH-45 set has
already been viewed, so this is a conversion diagnostic only.

Bind METH-47 update-512 checkpoint SHA-256
`0c6f09562efff7e78921036fe6ab030d7377150fe9afbe06b3deb1c029e0aa59`,
Instruct donor weights SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-45 manifest SHA-256
`15a68db9a43a8066e70b1b261f819cab3d6779d8b52633bcce7c725d8d54acfb`,
and METH-47 external result SHA-256
`9640f086c00c03db7ab7ff28ae13436a3b8fdef18996d2313aab772ddc652c29`.

For every expert, quantize each of the eight input-factor rows of
`a[8,896]` separately, and each of the 896 output-factor rows of
`b[896,8]` separately. Each row uses `max(abs(row))/127`, round-to-nearest
even and clamp [-127,127]. Store fp32 scales, read back every stored tensor
exactly, then reconstruct into the unchanged BF16 donor and original fp32
router. Report code and scale bytes independently, including the E27,355
and E273,547 **shape projections**.

**Fixed gates:** quantized-minus-original document BPB ≤+0.002 pooled and
in each category; ≥99% pooled next-token top-1 agreement on all 12 prompt
positions; no category may lose an EOS termination or add a repeated-8-gram
or early non-EOS continuation. Report exact greedy sequence matches. Verify
all 12 original greedy streams against the saved METH-47 result before
scoring the packed arm. A failure rejects this row-scale layout; do not
choose an intermediate granularity from these same results.

Local RTX 3060 only; ≤20 minutes, ≤10.5 GiB peak GPU allocation,
≤20 GiB RSS, no T4. Stop on hash, shape, reload, original-stream or
resource mismatch. This experiment evaluates reconstructed factors in
PyTorch; it does not benchmark a C LUT, large-E routing, semantic fidelity
or end-to-end native throughput.
