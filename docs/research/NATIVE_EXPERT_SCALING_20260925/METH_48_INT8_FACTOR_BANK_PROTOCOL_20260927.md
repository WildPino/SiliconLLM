# METH-48: packed trained-factor bank on the retained Instruct candidate

**Uncertainty.** The METH-47 E128 factors are stored as fp32 and the
10B/100B expert ladder requires a bank whose bytes grow with available RAM
while selected-path traffic stays fixed. Previous ternary/I4 factor exports
on the older base donor failed a prospective top-1 gate. Test a higher
fidelity one-byte code on the current Instruct candidate before a CPU LUT
implementation. This is a factor-storage diagnostic, not a claim about
large-E routing or full native throughput.

Bind the METH-47 update-512 checkpoint SHA-256
`0c6f09562efff7e78921036fe6ab030d7377150fe9afbe06b3deb1c029e0aa59`,
the Instruct donor weights SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
and the frozen METH-45 external manifest SHA-256
`15a68db9a43a8066e70b1b261f819cab3d6779d8b52633bcce7c725d8d54acfb`.
The METH-47 external result SHA-256 is
`9640f086c00c03db7ab7ff28ae13436a3b8fdef18996d2313aab772ddc652c29`.
This set has been viewed and is only a conversion diagnostic. Its output
cannot independently promote METH-47 semantics.

For each layer and expert, quantize the entire `a[8,896]` and `b[896,8]`
factor separately with a symmetric int8 scale `max(abs(weight))/127` and
round-to-nearest-even, clamped to [-127,127]. Store the two code tensors and
two fp32 scale vectors per layer. Keep router and donor unchanged. Reload
the saved artifact, reconstruct factors in fp32, and compare both arms on
the same frozen 12 documents, 12 chat prompts and greedy continuations.
Verify artifact metadata, exact code/scale readback, checkpoint hashes and
the original arm's saved METH-47 greedy token streams before drawing a
conversion conclusion.

**Fixed gates:** quantized-minus-original pooled and per-category document
BPB ≤+0.002; next-token top-1 agreement ≥99% on the 12 prompt positions;
quantized generation may add no repeated-8-gram or early non-EOS case in
any category and may lose no EOS termination. Report exact continuation
matches separately. Do not pick a different scale granularity or checkpoint
after seeing the result. A failure closes this per-expert int8 export;
another quantizer needs a new protocol.

Measure artifact bytes and exact bytes per expert for E128, E27,355 and
E273,547 with L24/D896/rank8; the latter two are **shape projections**, not
trained or allocated banks. Report quantization error, elapsed time, peak
GPU allocation and RSS. Local RTX 3060 only, ≤20 minutes, ≤10.5 GiB GPU
allocation and ≤20 GiB RSS; no T4. Stop before scoring if any bound hash,
shape, reload or original generation parity check fails.
