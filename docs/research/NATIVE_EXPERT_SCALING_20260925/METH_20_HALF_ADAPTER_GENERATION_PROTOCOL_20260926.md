# METH-20: paired generation with the exported half-amplitude adapter

**Prospective relative-behavior screen.** METH-19's factor-0.50 E128
adapter passed a separate-document BPB gate, but neither generated
behavior nor exact reload of its persistent tensor artifact had been
tested. This run compares the frozen Qwen2.5-0.5B donor with that
exact adapter on deterministic prompts. Repeated text is a coarse
failure signal; passing this screen alone cannot establish useful
answers, task accuracy, native parity, or speed.

## Bound input and apparatus

The [runner](../../../benchmarks/donor_adaptation/s1/meth20_half_adapter_generation.py)
ran in `--prepare` mode without model inference and produced the
[24-prompt manifest](meth20_half_adapter_generation_manifest.json),
SHA-256
`0e8641615a3e29fba7d9a04eb568268719cc4ff036fac1f0b7ed5cf7d5240284`.
Seed `meth20-20020` selects eight code, eight prose and all eight
general-technical documents from the METH-19 manifest by source-ID
rank. Each prompt is the first **256 donor-tokenizer IDs** from its
selected 4,095-byte-or-shorter document span; the manifest binds
every source ID, selected text SHA and prompt-ID SHA. These documents
were used for METH-19's BPB audit, but not to train the adapter or
choose this generation sample from their scores. The generation
outcome cannot become another independent document-quality gate.

Verify the METH-19 document manifest SHA-256
`1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70`
and reconstruct its source selection. Verify donor safetensors
SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, and **adapter-only**
safetensors SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
The adapter metadata must bind the METH-16 source checkpoint, model
and factor 0.50. Load all 72 `layers.<id>.a/b/router` tensors from
that adapter file into the 24 Qwen decoder FFN wrappers; the donor
control disables the wrappers in the same BF16/SDPA model instance.

For each prompt and arm, generate at most **128 greedy tokens** with
the model's EOS and KV cache. No sampling or prompt change. Keep raw
continuation IDs and decoded text. Count a repetition failure when
any continuation 8-gram appears at least three times. Record token
length, EOS termination, distinct bigram fraction, and whether the
first continuation token matches donor. Count premature non-EOS
outputs under 16 tokens separately. Report per-category and pooled
counts and the full pair rows.

The relative screen passes only if, in the full set and in each
category, the half-adapter has at most **one more** repeated-8gram
failure than donor and at most **one more** premature non-EOS output.
The gate is fixed before any generation result. A failure stops this
adapter's promotion pending diagnosis. A pass permits a separate
pertinent task/quality assessment; it does not assert useful
generation or authorize native C export by itself. Absolute high
donor and student repetition must be reported even if relative gates
pass. No choice of prompt or output factor may be revised using this
sample and counted as fresh.

One local RTX 3060 run; stop at 15 minutes wall time, 10.5 GiB peak
allocated GPU memory or 20 GiB process RSS. No T4 requested.
