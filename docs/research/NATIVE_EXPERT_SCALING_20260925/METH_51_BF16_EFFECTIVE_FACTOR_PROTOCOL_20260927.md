# METH-51: exact effective-factor export for the Instruct candidate

**Uncertainty.** METH-48/49 show that two one-byte trained-factor banks
fail next-token ranking, and METH-50 shows exact expert IDs do not fix the
row-int8 loss. The METH-47 forward casts selected fp32 factors to the BF16
hidden-state dtype before multiplication. Test whether storing exactly
those BF16-effective factor values provides a reproducible lossless
factor-bank anchor. This is an exact-representation test, not a new quality
sample or a CPU throughput result.

Bind METH-47 update-512 checkpoint SHA-256
`0c6f09562efff7e78921036fe6ab030d7377150fe9afbe06b3deb1c029e0aa59`,
Instruct donor weights SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-45 external manifest SHA-256
`15a68db9a43a8066e70b1b261f819cab3d6779d8b52633bcce7c725d8d54acfb`,
and METH-47 paired result SHA-256
`9640f086c00c03db7ab7ff28ae13436a3b8fdef18996d2313aab772ddc652c29`.
The viewed prompts are used for exact parity, not statistical promotion.

For each layer store `a[128,8,896]` and `b[128,896,8]` as BF16 in
`safetensors`, with checkpoint/model metadata. Read back every tensor
bit-identically. Restore into the existing fp32 wrapper parameter so its
unchanged forward casts the selected values to BF16. Verify that every
restored factor's BF16 bits equal the original parameter's BF16 bits.
Keep the donor and fp32 router unchanged.

**Fixed gates:** original-versus-reloaded logits must be bit-identical on
all 12 frozen prompts (all 2,091 positions); original greedy streams must
match the saved METH-47 reference, and reloaded greedy streams must match
original on all 12 prompts. Stop on any mismatch and report it. Include
tensor payload bytes per expert and E27,355/E273,547 shape projections,
not claims of trained larger-E quality. At top-4, compute selected factor
bytes/token independently of E.

Local RTX 3060 only; ≤10 minutes, ≤10.5 GiB GPU allocation and ≤20 GiB
RSS, no T4. This export excludes the donor, router, index and native
format. No CPU LUT or accepted-token rate can be inferred from parity.
