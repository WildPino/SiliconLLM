# First attention source acquisition: bounded bytes and public prefixes

This stage prepares only the source values needed for a first-layer GPT-OSS-120B attention experiment. It does **not** download experts, the full embedding/head, a complete checkpoint, or any Falcon outputs/logits. No model is instantiated and no GPU is used. The fixed revision is `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`; the local [config](large_family_metadata/config120.json), [index](large_family_metadata/index120.json), and [reference](large_family_metadata/reference_model.py) are prerequisites. The config SHA-256 is `933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566`, and the reference SHA-256 is `319ba3c6e4d827eb3115fe4838942af448376b513c02cea4b6b471bf6c21a346`.

Only `model.layers.0.input_layernorm.weight`, layer-0 `self_attn` Q/K/V/O weights and biases, and `self_attn.sinks` are acquired as complete BF16 tensors. The 2,880-wide embedding is acquired only for canonical token IDs used in the selected prefixes, one BF16 row per unique ID. No full embedding matrix or LM-head payload is requested. The output is `results/native_expert_scaling/first_attention_source_20261015/manifest.json`, schema `FIRST_ATTENTION_SOURCE_V1`, with exact-HF-name tensor descriptors `{name,path,shape,dtype,bytes,sha256,source_shard,source_offset}` and `embedding_rows` `{path,shape:[U,2880],dtype:'BF16',bytes,sha256,token_ids,row_lookup}`. `source_offset` is an absolute byte offset in the pinned shard. Extra custody fields may accompany these descriptors.

## Frozen selection before source-weight requests

The input corpus is `E:/SiliconLLM/results/native_expert_scaling/chatbot_source_coverage_20261015/corpus.json`, SHA-256 `1c42eb789f52258d7796bcaeb8b52387c648ffe803f8955ecea5c2f3bdbd9ad6`. The scanner hashes the complete file but semantically decodes only case ID, split, domain, public input messages, group/prefix provenance, and upstream row provenance; stored Falcon answers, output IDs and logit paths are skipped. It verifies 432 cases, 360 FIT/72 DEV, 12 domains and unique IDs/groups. The existing source-coverage plan (SHA-256 `67aa70f4b24189e407582262d3a01df57c0ca62bcce4b109f070741dbdb071fa`) supplies `first_user_key`, `conversation_sha256`, and the established `normalized_all_user_turns_v2` group scheme.

The **source-family holdout** is fixed before tokenizer or weight values are observed: sort the 12 domain names by SHA-256 of `first-attention-source-holdout-v1:` plus the domain, and assign the first four (`smol-magpie-ultra`, `metamathqa-50k`, `smol-constraints`, `self-oss-instruct`) to DEV only. The remaining eight domains are FIT only. Eligible cohorts retain original FIT records in FIT domains and original DEV records in DEV domains. Before selection, remove any record whose `first_user_key` or `group_key` appears in both cohorts. Within each domain, sort eligible records by ID and select the first three with distinct `first_user_key` and `group_key`. This yields 24 FIT and 12 DEV cases, with 36 distinct group and first-user keys. `eligibility_plan.json` is written before the first tokenizer request; it records cohort counts, all noncohort and duplicate exclusions, eligible counts, source-family order, and exact selected IDs. A collision or insufficient eligible records stops before weights. This is source-family-disjoint development coverage, not a claim that the original corpus's entire DEV partition holds out sources. The earlier two-FIT/one-DEV-per-domain candidate exposed a first-user collision; it is not the frozen selection.

The pinned revision's `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and `chat_template.jinja` are fetched as bounded metadata, with the already pinned `config120.json` copied into the isolated tokenizer directory. The isolated local `AutoTokenizer` renders each original public message history with `add_generation_prompt=True`, `reasoning_effort='medium'`, and the template's `strftime_now` fixed to UTC `2026-10-15`; the fixed rendering clock and actual host UTC date are logged separately. It does not load a model or trust remote code. If the local tokenizer/runtime cannot render the official template, the stage stops rather than inventing Harmony formatting. It records original canonical token count, every canonical token ID, and SHA-256 of all IDs encoded as contiguous little-endian U32, then retains the first at most 256 IDs and their separate U32 SHA-256. The original corpus `prefix_sha256` stays unchanged as provenance and is never repurposed to mean GPT-OSS token IDs. Every case records public messages, first-256 canonical IDs and the complete canonical ID list. The longest prefix shared by all 36 first-256 sequences is `common_prefix_length`; `evaluation_query_start` equals that value and every case must have at least one variable query token after it. The sorted unique retained IDs determine the embedding row set and `row_lookup`. `selection_plan.json` is written **before the first weight-range HTTP call**.

## HTTP, header, and custody limits

Each shard request uses an immutable revision URL and `Range` with `Accept-Encoding: identity`. The client verifies HTTP **206**, exact `Content-Range` start/end and declared total, identity encoding, exact length, and a stable ETag when supplied **before reading any range body**. A known wrong 200/range response is closed without reading its body and retried once with a cache-busting query specific to that requested byte interval; a second wrong response stops. Every attempt, response header, final URL, ETag, byte count and SHA-256 is written to `http_events.jsonl`, including pre-body refusals. Tokenizer files are ordinary bounded 200 responses. No mutable branch URL or unbounded download is allowed.

For each fetched shard, the client retrieves its complete safetensors header (within the metadata cap), requires exact equality between the header's tensor-name set and the local index's names for that shard, verifies every shape/dtype/offset byte count, contiguous nonoverlapping payload offsets, and total file length. Selected tensor ranges are fetched in chunks of at most 2 MiB, exported to separate files, and independently hashed. The embedding row plan merges only consecutive IDs or gaps of at most eight rows, with each HTTP span at most 2 MiB. It charges all unused gap bytes, caps them at 100 MiB, and stops before weight requests if more than 2,000 row spans would be needed. The total HTTP call cap is 2,048.

Caps are 16 MiB shard metadata, 64 MiB tokenizer files, 256 MiB weight-range response bodies, 384 MiB total network bodies, 512 MiB local output, 2 GiB controller OS peak, six logical CPUs, and 900 seconds. Exclusive output names prevent accidental overwrite. `owned_process.json` identifies the writer; a terminal receipt records success or failure. All input pins, including tokenizer bytes and pinned runtime modules, are sealed before and after acquisition. This stage has no source-value quality or accepted-token-rate claim.

Launch only after reviewing and committing this script and protocol, passing their exact Git revision to `--freeze`. The script checks normalized Git blobs before selection/acquisition. Execution follows the existing research authorization after source review and freeze.

## Runtime-alias preflight correction

The first launch at85b8980 stopped before any HTTP request or weight acquisition:
individual runtime package directories are junctions into the repository venv,
so requiring resolved files to remain under the alias root rejected the expected
installation. The corrected check compares each module with its exact expected
package entry point after resolving aliases; hashes still seal the actual files.
The original exclusive directory and terminal fault remain intact. The corrected
launch uses `--output results/native_expert_scaling/first_attention_source_repair1_20261015`;
this changes neither the data rule nor model revision, tensors or numerical gates.

## Physical vocabulary padding correction

The corrected runtime launch acquired the four tokenizer files, then stopped
before any weight request: the official tokenizer has200,019 IDs (maximum
ID200,018), while the checkpoint allocates201,088 embedding/head rows. Exact
equality was an invalid precondition for this padded vocabulary. The next
launch requires every tokenizer ID to address an existing source row, retaining
all physical model rows in accounting. No token ID, text, split or metric changes.

The tokenizer files already obtained from the immutable revision are SHA-256
pinned in the corrected source. `--tokenizer-cache` may reuse those exact bytes;
each hit records its input descriptor and zero new network-body bytes. No
cache file is trusted by path alone. This avoids a repeated28MB metadata fetch.
The next exclusive output is`first_attention_source_repair2_20261015`; all
earlier ownership/terminal receipts remain unchanged.

## Explicit tokenizer result contract

The next preparation stopped before weight requests because installed
Transformers5.13.1 defaults `apply_chat_template` to a dictionary result.
The corrected call explicitly uses `return_dict=False`; rendered text, token
IDs, clock and data rules are unchanged. A bounded pre-body HTTP403/416 refusal
may also receive the same single interval-specific cache-busting retry; the
refusal is retained, and no error body is read. The next exclusive directory
is`first_attention_source_repair3_20261015`; earlier faults stay intact.
