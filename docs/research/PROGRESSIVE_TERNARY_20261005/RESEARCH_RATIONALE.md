# Initial research rationale and source admission

5 October 2026. Prospective reasoning; no new donor numerical results.

## Why the objective is plausible and unresolved here

Weight distance ignores the input distribution and compensation between
coordinates. For a linear projection, reconstruction is
`L(Q) = ||X (W - Q)^T||_F^2`, whose curvature depends on `X^T X`.
The final prediction additionally depends on residual context, normalization,
readout and source probabilities. A better local fit must be tested rather
than assumed to improve that final prediction.

Data-aware rounding and progressive weight compensation are established
research directions. The first implementation is an adaptation and diagnostic
comparison, not a claim to invent ternary post-training quantization.

- [Adaptive rounding](https://arxiv.org/abs/2004.10568) optimizes rounding using
  unlabeled calibration data and a local reconstruction objective.
- [Second-order post-training quantization](https://arxiv.org/abs/2210.17323)
  supplies a scalable progressive compensation method and discusses ternary
  settings. Its reported results do not transfer automatically to this donor.
  [Reference implementation](https://github.com/IST-DASLab/gptq/blob/main/gptq.py)
  and [quantizer](https://github.com/IST-DASLab/gptq/blob/main/quant.py) were
  inspected as algorithm references; no source files were copied into the setup.
- [Post-training ternarization](https://arxiv.org/abs/2510.03267) combines
  iterative grid fitting with activation-aware alignment. Its asymmetric grid
  is distinct from the fixed symmetric `scale * {-1,0,+1}` studied first here.

These source summaries describe methodology, not evidence of a successful
conversion in this branch. Calibration, source-context prediction, generation,
native implementation and useful expert capacity remain separate tests.

## Pinned source identities admitted through public metadata

Donor: `Qwen/Qwen2.5-0.5B`, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`.
The parent METH13 binding and current public metadata agree on the original
`model.safetensors`: 988,097,824 bytes, SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
[Pinned donor source](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987).

Candidate diagnostic corpus: `Salesforce/wikitext`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, configuration `wikitext-2-raw-v1`.
Training parquet: 6,357,543 bytes, SHA-256
`e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
Test parquet: 732,610 bytes, SHA-256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
Metadata admission did not tokenize, fit or evaluate the donor. Select exact
sample rules prospectively. This public corpus is a diagnostic benchmark,
not a guarantee of novel project-wide evaluation or broad task coverage.

## First bounded question

Can progressive compensation and revisiting discrete ternary choices improve
held-out reconstruction and source-context prediction over direct rounding for
one unchanged-width projection, while using exactly the same final scales and
ternary code storage? The first decision is whether this recipe deserves a
larger nonlinear/expert test; it is not a complete-model promotion.
