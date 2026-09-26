# METH-24: materialize the R8 core

METH-23 passed its diagnostic gate on reused texts. Freeze its rule and
export a real packed-only artifact before evaluating any new text. The
source is the pinned Qwen2.5-0.5B BF16 safetensors, SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The companion half-amplitude E128 adapter has SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.

Encode all 169 tied-head, FFN and attention matrices with the exact
`t2_rules.r8_int8_rtn` per-output-row rule used in METH-23. Store signed
int8 codes and fp32 row scales, without an uncompressed matrix copy.
Store all 121 norm/bias controls in fp32. Preserve the single tied
embedding/output-head identity as metadata. Expected matrix codes:
493,961,216 bytes; expected tensors: 459. Record organ byte counts,
matrix reconstruction errors, artifact byte count and SHA-256. Reload
every tensor and metadata field and require exact equality before the
export is accepted.

This step establishes artifact identity and physical storage only. It
does not certify new-text quality, a native kernel, or ≥50 tok/s. The
subsequent evaluation must load these stored codes and scales, rather
than independently quantizing the BF16 source again. Stop after 15
minutes, 10.5 GiB GPU allocated or 20 GiB RSS.
