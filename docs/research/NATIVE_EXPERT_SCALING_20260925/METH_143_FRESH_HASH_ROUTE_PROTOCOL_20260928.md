# METH-143: new-source route-load audit of the fixed token hash

METH-142's unsigned 64-bit token-context hash passed all six fixed
development load cells, but their sources had already been viewed in
METH-141. This experiment decides whether its **unchanged** constants
generalize to a fresh 24-document source/fragment-disjoint manifest.
It still does not measure trained B usefulness, model quality or
native CPU speed.

Bind METH-142 result SHA-256
`e7a302671fc9e24290e39b8e0c1864c7b7b053f26057fa9c357d5f201e0decd3`
and its committed hash formula/seed/golden vector. Bind the original
METH-126 BF16 bank, METH-107 centered child checkpoint, donor and
parent checkpoints and tokenizer as in METH-136. Reuse the hash rule
without changing any constant, input tuple, tie or modulo behavior.
Require the golden vector `(123,45,67,89,3) -> 899`, exact original
E1280 teacher/control BF16 logits on eight bound prompts, and
`grandchild // 10 == source child` at every counted route.

Before route inference, deterministically select eight code, eight
technical and eight book sources with seed
`meth143-hash-route-external-143000`, using pinned Git ref
`882bb43f9118df1e4f79f85a6072111897bede9e` and PG19 train shard
`data/external/pg19/data/train-00007-of-00023-e9fea5c158e172e2.parquet`
SHA-256
`7dc048a9b88a7ed1ebb36f2c7f666d29686439a06396b87b2275234fc44c9fac`.
Use the METH-133 4,095-byte span selection and exclude source IDs
and first/middle/final 256-byte overlaps with every earlier audit
manifest through METH-133, H0 `calib.txt`, and identifiable METH-136
training prompts. Verify that the H0 PG19 source budget ends in its
first train shard, making shard 7 source-disjoint from that training
stream. Freeze source/text/token hashes and the selected rows in a
committed manifest before inspecting any hash route output. Reserve
PG19 shard 6 for a future quality audit; do not use it here.

On the 24 selected documents, run identical E1280 child routing in
nonoverlapping windows of at most 128 and 512 tokens, resetting
previous-token and position at each window. Apply the fixed hash
offline to each selected child. Evaluate each width separately.
Require in **both** widths and every layer candidate/control
maximum-to-mean load ratio <=1.25, at least 4,000 selected
grandchildren, and no grandchild receiving more than 25% of a
source child's selections when the child has at least 250. Save
all layer counts, top IDs and failed gates. Stop on any failure.

A pass licenses a separately frozen hash-routed full-model clone
parity and native C CPU-cost test. Only then consider B-only training
against the matched E1280 control and fresh held-out quality. The
route hash may balance traffic while still partitioning semantics
poorly, so load alone cannot support the user's 10× useful-expert
claim or the >=50 accepted tok/s/LUT target.

Use the local RTX 3060 only. Stop at 15 minutes, 10.5 GiB allocated
GPU memory, 20 GiB process RSS or 1 GB new disk. No T4.
