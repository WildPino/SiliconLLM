# PQT-SRC-007: prospective original float-context input preparation

6 October2026, before corpus download/tokenization and model functions.
Freeze independent calibration/development/held-out document inputs for the
original float google/switch-base-128 encoder-decoder path. No fit, original
model function, GPU job or native timing is admitted by this input operation.

Use allenai/c4 en validation shard00000-of00008, revision
f998d2cd8b92435980789e3ecb2f89b4c68bfe1e, exactly40,435,873 compressed bytes,
SHA256bc35d7c1b1d14b90cd3a394cccbcbe191935edd04bf42ee965379c6e2987a5f0.
The [dataset card](https://huggingface.co/datasets/allenai/c4) states ODC-BY and
references Common Crawl terms; retain pinned source-card and document URLs.
This source has no earlier admission in the research's recorded input IDs.
Independent splits here concern research optimization, not proof of absence
from pretraining or semantic overlap with other corpora. Do not make broader
capability claims from this input preparation.

Use the exact tokenizer/config files previously acquired at original model
revision86c815ec05361a33a8b49fc717277da9c0a4e711, checked against the inherited
complete acquisition report. Read main originals without mutation; copy only
small tokenizer/config files into owned outputs. Bind full bytes/SHA/source
commit before model execution. Fast tokenizer from owned local tokenizer.json,
Transformers4.57.6/tokenizers0.22.2, no network model/tokenizer loading.

Download the compressed corpus once into an exclusive numbered namespace;
verify full length/SHA before parsing. Bound download300s/64 MiB, chunks1 MiB,
decompressed observed bytes512 MiB, JSON line2 MiB, at most10,000 observed rows.
Stop after the first4,096 eligible unique document/token prefixes. Eligibility:
UTF-8 text<=64 KiB, nonempty URL, at least128 nonspecial prefix tokens. Reject
exact normalized text, URL or128-token-prefix duplicates. Document identity is
SHA256 of NFKC/casefold/whitespace-normalized full text. Keep raw selected text
and source URLs plus source row/identities; observe no original model output.

Rank candidates by SHA256("PQT_SRC_007_SPLIT_V1:20261006:" + document identity).
Bucket by its first8 hex digits modulo3; take the128 smallest ranking hashes
per bucket, respectively calibration/development/held_out. Require all384
documents, URL and token prefixes disjoint, with exact frozen order. Coverage
selection uses no model function or quality observation. Do not replace or
extend splits after numerical results.

Use first128 tokens per document, add_special_tokens=False. For each document,
seed Python Random from SHA256("PQT_SRC_007_MASK_V1:" + document identity), first
eight bytes interpreted little-endian. Draw sorted five cuts from1..18 for six
positive noise spans totaling19 tokens; draw five cuts from1..108 for six
positive nonnoise spans totaling109. Alternate nonnoise/noise, ending in noise.
Encoder replaces each noise span with extra_id_0..5 and appends EOS1:116 IDs.
Labels contain sentinel plus removed tokens per span, then extra_id_6 and EOS1:
27 IDs. Decoder input is pad/start0 followed by labels[:-1], with all real
positions visible. All masks ones; padding/capacity controls remain separate.
Retain original128 tokens, full mask positions, both segment-length arrays and
all actual IDs. This is explicitly specified span corruption aligned with the
[model's masked-language-modeling task](https://huggingface.co/google/switch-base-128),
not a claim of reproducing an official benchmark sampler. Independently prove
lossless reconstruction and disjointness before original functions.

Local preparation requires fresh live process admission, isolated pinned CPU
runtime/one thread/no gradient/model construction. Complete child<=600s, RSS<=
1 GiB, compressed corpus64 MiB, all outputs<=128 MiB. Capture complete child
and nested timers, full tokenizer sources/versions/input identities and first
failures. Metadata discovery001 stopped on a4 MiB cap; its observed4 MiB+1
response prefix was not saved or hashed. Preserve that evidence limitation.
Metadata discovery002 uses minimal fields/exact LFS pointer and passes; retain
all four complete raw source records. Neither metadata attempt saw corpus or
model functions. No silent repetition of completed namespaces.

Next separately freeze original float execution/reference/capture protocol.
Include CUDA provider qualification, complete checkpoint acquisition and
streaming I/O/scratch/deadline/allocator/RSS limits, fixed natural capacity64,
all route/drop/coverage/discard rules and independent original-weight references.
No capture or fitting is authorized by this input-only result. Later long T4
dispatch requires fresh account admission and delegated monitoring while the
coordinator waits dormant; native CPU timing still awaits owner availability.
