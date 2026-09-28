# METH-128: can an int8 head shortlist preserve the exact BF16 choice?

## Uncertainty and prior evidence

METH-127's six-thread FP32 full C profile charges 14.767 ms/token to the
tied vocabulary head. A BF16 exact head still addresses 272,269,312 bytes
per token; METH-59/91 show that substituting an int8 head throughout the
model can lose ranking/grounding quality. The untested variable is a two-pass
head: score every token with stored int8 head codes, then read original BF16
rows only for the shortlist and choose by exact score. This trades more
resident RAM for less active head traffic. It cannot by itself repair body
quantization or preserve full-vocabulary likelihood.

## Bound diagnostic

- Use the quality-gated METH-123 centered E1280 BF16 Qwen2.5-0.5B-Instruct
  model: donor source SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
  METH-56 parent checkpoint
  `8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`,
  METH-107 child checkpoint
  `15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`.
  Apply the METH-119 per-parent centered-B transform, no other changes.
- Use all 24 prompts in the METH-121 external manifest, SHA-256
  `7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366`.
  These sources are **viewed**: this is a representation screen only, never
  an independent model-quality promotion.
- Compare two already stored candidate heads against the original BF16
  tied head on the *same final hidden states*: METH-59 per-output-row R8
  core SHA-256
  `5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd`
  and METH-91 group-128 R8 core SHA-256
  `6f994c9ddf047adc789d3e1c841dd3be156cc578ed0d6c3f4045f97212e2accb`.
  Load the stored code/scale tensors, reconstruct each head exactly as
  specified by its exporter, and verify tied-head dimensions and source.
- For K = 1, 4, 16, 64, 256, count the positions whose exact BF16 head
  argmax is absent from the approximate top-K. Report margins/ranks by
  category and per prompt. Check that a K-candidate exact BF16 rescore
  reproduces the original argmax whenever the original token is included.
  The control is K = full vocabulary, which must recover every exact top-1.

## Decision and budget

The pilot gate for native implementation is **zero omitted exact top-1
tokens at K=64 over all prompt positions**, plus exact rerank parity on
those positions. Prefer the per-row format if both pass; use group-128 only
if it passes and per-row fails. If neither passes, reject K=64 and report
the smallest tested K with zero omissions, if any. This is a strict
development screen, not a claim of generation/task/semantic conservation.
A passing arm next needs new disjoint sources, full-model greedy/task audits,
an actual CPU two-pass head and end-to-end timing on a compact body.

Local RTX 3060 only; no T4. Stop on any bound hash mismatch, baseline
head mismatch, nonfinite scores, >15 minutes, >10.5 GiB GPU allocation or
>20 GiB process RSS. Record the true resident BF16+R8 head bytes, nominal
per-token code/scale plus K BF16-row bytes and runtime resources. Do not
call a theoretical byte saving a measured CPU speedup.
