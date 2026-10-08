# Preserve the source decay heads in the compact core

8 October2026. New shape variable after first integer budget, before any target
value/forward/fit observation. Keep first budget immutable. A full source head
has64 channels with one A/delta/D shared over those channels and state256.
SSM width768 can be12heads*64 or48heads*16. The latter keeps all48 source
A_log/delta-bias/D indices and all source decay rates at initialization, while
selecting/projecting fewer channels inside each head. Width/depth/norm/input
changes still require learning; actual delta values are not guaranteed preserved.

Choose48heads*16 for first whole pilot. It changes input projection delta rows
12->48, adding10*512*36=184320 products and1080 scalar coefficients. Source
state size/cache/FFN slots/k/h do not change. Same F32 packed-only budget rules.
New total69.435904M products/424404608B packed/4075788800B Adam, teacher+Adam
lower envelope7185561832B. Reused earlier source metadata, no source values or
model call. New code commit precedes second integer observation/result namespace.

Training caveat derived from local source code: SSD prefill pairwise G tensor
scales with head count*N*chunk*sequence, so48 versus12 heads increases this
workspace by4 at fixed chunk size. Chunk16 versus32 or activation checkpointing
may offset this, but actual forward/backward/step memory must be measured.
This chooses recurrence information retention with negligible matrix-count
increase; it does not infer memory or whole quality from the symbolic counts.

Original budget and all other claims stay scoped. Native scan plus group1 gated
norm/projection/conv support and LUT-aware ternary learner remain missing.
