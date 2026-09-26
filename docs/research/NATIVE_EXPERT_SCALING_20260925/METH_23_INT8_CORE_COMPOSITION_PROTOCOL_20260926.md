# METH-23: R8 int8 head/body composition diagnosis

**Prospective diagnostic, not a promotion test.** METH-22 prices the
Qwen2.5-0.5B tied fp32 head as the binding cost for a one-byte-body
native candidate. METH-19/21 bound a half-amplitude E128 adapter
that retains scoped donor-relative quality with the original BF16
core. This test asks whether a *different precision rule* leaves
head, body, and their composition close enough to justify a real
int8 export. It uses the previously scored METH-19 documents and
METH-20 prompts, so even a pass requires a new independent quality
set before promotion.

## Bound inputs and changed variable

Verify Qwen source safetensors SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, and half-amplitude
adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
Reconstruct the METH-19 document manifest SHA-256
`1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70`
and match its committed result SHA-256
`eabf4bbe95e326656f9c65589c4a35da1527855dcb645f733d0020ca431e437a`.
Reconstruct the METH-20 24-prompt manifest SHA-256
`0e8641615a3e29fba7d9a04eb568268719cc4ff036fac1f0b7ed5cf7d5240284`.
No documents or prompts may be selected by their previous scores.

Use the existing `t2_rules.r8_int8_rtn`: for each output row of a
matrix, scale `max(abs(weight))/127` clamped to at least `1e-12`,
round weight/scale to nearest even and clip codes to [−127,127].
The source is BF16 donor weights widened exactly to fp32 for
quantization. Dequantize `code*scale` back to BF16 for the PyTorch
quality diagnostic. Record exact int8 code and fp32 row-scale byte
counts and reconstruction errors. This **does not measure an int8
kernel** or achieve the compressed memory footprint during scoring.
Keep norm/bias vectors, E128 router and expert factors unchanged.

Score four cases, each with the donor (experts disabled) and the
same loaded factor-0.50 adapter (experts enabled):

1. Original BF16 core.
2. R8 on the tied embedding/output-head matrix only.
3. R8 on all 72 FFN and 96 attention matrices only.
4. R8 on tied head and all 168 body matrices together.

Restore the exact BF16 original before each case. Verify that the
original donor/adapter pooled BPB reproduces METH-19 within
`1e-5`. Score all 56 METH-19 documents with its EOS prefix,
512-token target stride, up to 512 left-context tokens, absolute
positions and full selected bytes. Report donor and adapter pooled
and category BPB, plus deltas to the original donor and each case's
own donor. For each case and arm, record top-1 IDs at every position
of the 24 fixed 256-token METH-20 prefixes (6,144 positions) and
compare with the same arm's original BF16 IDs. This ranking pair
addresses the prior E17 finding that loss and token ranking can
disagree inside a damaged quantization regime.

The **diagnostic quality gate** passes only if the head+body R8 case:

- raises donor pooled BPB by at most **+0.01** versus original donor;
- leaves adapter pooled BPB at most **+0.01** above original donor;
- leaves adapter BPB at most **+0.03** above original donor in each
  code, prose and technical category; and
- retains at least **95% exact top-1 IDs** for both donor and adapter
  against their respective original-core controls.

A pass makes an actual compressed export and a *new* independent
document/task/generation audit the next step. A fail identifies
which organ or composition needs a different precision rule. No
quality or ≥50 tok/s claim follows from this emulated run.

The historical E17 comparison changed a ternary head on a badly
degraded ternary body; E64 tested int8 against a post-hoc hard FFN
carve. This cell instead retains the full pretrained body, uses the
existing **R8 int8** rule, and composes the previously trained
residual experts. Those prior failures motivate the controls but
do not settle this changed configuration.

Run once on the RTX 3060; stop at 15 minutes wall time, 10.5 GiB
peak allocated GPU memory or 20 GiB RSS. No T4 requested.
