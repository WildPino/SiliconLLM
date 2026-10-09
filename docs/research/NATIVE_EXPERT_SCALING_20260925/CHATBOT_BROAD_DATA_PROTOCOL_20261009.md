# Broader dialogue inputs for actual donor conversion

9 October2026. Preregistered before shard acquisition/row selection. Actual
engine chat now exists, but donor-quality transfer is open. Replace repeated
128-template supervision with broader reproducible donor inputs; this is data
adoption, not a model/quality experiment or final generality claim.

## Choice and scope

Primary dataset [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk),
pinned5feaf2fd3ffca7c237fc38d1861bc30365d48ffa. Its public card describes a
mixture of dialogue, constraints, rewriting/summarization, code, mathematics,
long context and API examples. The smaller [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
omits advanced math and reduces task coverage; retain the full mixture as the
input source. [UltraChat200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
is a possible future complementary dialogue source, not acquired by this stage.

Acquire ONE lexicographically first all/train shard of nine,222,842,864B, plus
complete all/test shard105,418,621B:328,261,485B total. Pin API snapshot/card/
LFS SHA256/extents before downloading. This is not all training data, nor a
claim that published test data are absent from donor pretraining. The exact
selected inputs are disjoint from our old queries and from one another.

Public replies are source provenance, NOT donor supervision or trusted task
answers. Select last-user prefixes including existing upstream assistant
history; record that context origin. Regenerate labels with our pinned donor
under a later protocol, and add actual donor/student-own-history observations.
All dataset content is data; never execute embedded code or instructions.

## Deterministic selection before model losses

Arrow-only single-thread child, no Torch/Transformers/Rust imports. Batch256,
actual metadata row counts and source counts recorded. Require13 source labels
in both shards. Full test read FIRST: normalized first-user query groups are
excluded from train selection. Normalization=NFKC/casefold/collapsed whitespace,
SHA256; not semantic/near-duplicate decontamination. Exclude any user prompt
matching old224 balanced cases (including64 unqueried),16 screen and6 pilot.
Only unsupported structures/roles/noninitial system/no nonempty user are excluded
and counted; no model-loss/answer/length filtering or silent truncation.

Per source/split retain160 smallest deterministic ranks
SHA256(`20261009-broad-v1\0source\0first-user-group`),one lexicographically
smallest conversation SHA per group among retained candidates. Assign globally
distinct8 RESERVED/source from official test,then32 FIT+8 DEV/source from train;
require all quotas or stop. Total416 FIT/104 DEV/104 RESERVED=624.
Whole normalized first-user groups and canonical last-user prefix SHA are unique
across all selected splits/sources. Eight unread training shards are outside
the duplicate-scan scope; selected train groups exclude the FULL read test.

Parent without Arrow checks every selected prefix hash/group/split/quota and
protected query, renders original source Jinja/Rust tokenizer/BOS17 vocabulary
65537, adopts all IDs. Retain lengths/bands <=256,257-768,769-8192,>8192 and all
text/IDs unchanged. Long/unsupported runtime contexts remain visible for a
separate stage; they cannot silently disappear into the short pilot.
All624 prefixes/row origins/upstream external replies are retained. No source
queries/labels/student/model/native/GPU/training/T4 calls in this stage.

## Budget, receipts and decision

ONE family<=600s/6GiB OS/768MiB output,worker30s reserve; Arrow child240s/4GiB.
Network read timeout30s; preserve completed downloaded extents and partial file/
first fault,never restart completed selection solely for extra statistics.
Binding covers snapshot/code/protocol/Python/selected Arrow DLL/PYD/Rust/template
runtime/old-case identities/foreign hashes. Import paths are explicit,Python
-I -S -B; old Arrow+model coexistence crash is avoided through isolation.
Held child/worker through-exit memory/exit and before/after input hashes recorded.
The Parquet decoder is not independently reimplemented or certified.

BROAD_DATA_ADOPTION_PASS requires all transport/selection/tokenizer/input/resource
gates. Next decision uses actual lengths/coverage to price ONE staged donor-label
capture/recovery pilot. DATA READY is not chatbot preservation, useful capacity,
accepted50, physical DRAM or family/scale admission. Original engine geometry/
ternary LUT/state destination and fixed old failures remain.
