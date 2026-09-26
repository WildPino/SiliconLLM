# METH-27: fresh-prompt generation from the stored R8 core

METH-25 passes the independent document/ranking screen but does not
establish useful generation. METH-20 found 16/24 repeated-8gram loops
for the BF16-core adapter on older prompts, versus 17/24 for its
donor. This cell changes the core precision only, using the fixed
packed-only R8 artifact SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`
and unchanged factor-0.50 E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.

Select 8 code, 8 prose and 8 technical documents by fixed seed
`252527` from the METH-25 manifest SHA-256
`20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77`.
Use each selected document's first 256 donor-tokenizer IDs as a
prompt. METH-25 used these prefixes for top-1 comparison, but did
not generate continuations; their document loss was already viewed.
This is a prospective behavior screen, not another independent
document-loss sample. The resulting 24-prompt manifest SHA-256 is
`94abd7544d4f887e476286774b2ce51bbc0c5c68f079907b88b3984589d62ec6`;
it binds every prompt ID hash before generation.

On the RTX 3060, generate up to 128 greedy tokens with EOS/KV cache
for each prompt and three arms in one model instance: original BF16
donor; stored-R8 donor; stored-R8 core plus the exact adapter. Rebuild
all 169 matrices from **stored int8 codes and fp32 row scales** into
BF16 without re-quantizing the source. Verify metadata, shapes,
control-vector equality and head tying. Preserve every continuation
ID and decoded text. Report repeated 8-grams appearing at least 3
times, early non-EOS outputs under 16 tokens, EOS termination, length,
distinct-2, first-token agreement and per-category counts.

The relative behavior gate passes only if R8+adapter has no more
than **one extra** repetition failure and one extra early non-EOS
output versus R8 donor in the pooled set and each category. A
separate absolute-usefulness screen passes only if R8+adapter has
at most **4/24** repetition failures overall and **2/8** in code,
with no early non-EOS output. Failure on the absolute screen holds
native promotion pending a generation repair; it does not invalidate
the measured document/ranking quality. No prompt, amplitude or rule
may be revised after results and counted as fresh.

One local run, stop at 15 minutes, 10.5 GiB allocated GPU memory or
20 GiB RSS. No T4 requested. This PyTorch BF16-reconstruction test
does not measure C inference speed.
