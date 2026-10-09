# Broader dialogue inputs: actual adoption and limits

9 October2026. Data adoption COMPLETE/PASS; no donor, learner, training or GPU
calls. This is an input preparation result, not chatbot quality evidence.

## Identity and reproducible result

Dataset HuggingFaceTB/smoltalk, pinned revision
`5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`: one of nine train shards and the
complete official test shard. API/card snapshot, pinned LFS extents, source
tokenizer/template, grouping, quotas and runtime are bound. The
[official dataset card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
describes the constituent sources; source labels here are provenance strata,
not measured competence categories or guarantees about answer correctness.

Repair freeze `457e790609af79fb95209665865d6664487c1f13`;
[binding](chatbot_broad_data_binding_repair1_20261009.json) SHA256
`d5dd9b8bd2a8c8fb89af2b6c197795e8c7081fc769862aa56c56778757f38e35`;
[result](chatbot_broad_data_result_repair1_20261009.json) SHA256
`b3519a65bae4d06ee7830517098e15abe34a3d58fba8ce9f4eae864bab4ac185`;
[terminal receipt](chatbot_broad_data_result_repair1_20261009.terminal.json).
The actual corpus is
`results/native_expert_scaling/chatbot_broad_data_repair1_20261009/corpus.json`,
13,946,319B, SHA256
`3ec0f6e1f32b417af631334d4f5d4615fc566e1eb6982f5f1258c7490a45a479`.

## What was adopted

624 canonical last-user prefixes:13 sources times32 FIT/8 DEV/8 RESERVED,
thus416 FIT/104 DEV/104 RESERVED. Sources:apigen-80k,everyday-conversations,
explore-instruct-rewriting,longalign,metamathqa-50k,numina-cot-100k,
openhermes-100k,self-oss-instruct,smol-constraints,smol-magpie-ultra,
smol-rewrite,smol-summarize,systemchats-30k. All messages and lengths are retained;
no answer, model-loss or length filtering. Public preceding assistant turns occur
in124 cases; their provenance is preserved. External final answers are retained
as public references, not pinned donor labels or ground truth.

| Split | <=256 IDs | 257-768 | 769-8192 | >8192 | Total |
|---|---:|---:|---:|---:|---:|
| FIT |267|75|43|31|416|
| DEV |72|11|13|8|104|
| RESERVED |67|17|12|8|104|

Minimum23/maximum20,874 input IDs. All48 longalign cases are8186-20874 IDs;
47 adopted prefixes overall exceed8192. A short-stage capture cannot claim
long-context preservation. These inputs remain intact for a separately costed
source/runtime/conversion stage; no easier replacement or silent cropping.

## Grouping correction and retained first failures

The [first attempt](CHATBOT_BROAD_DATA_FIRST_FAILURE_20261009.md) remains FAILED:
the launcher rejected its own conhost child, and its surviving Arrow reader later
found only5 first-user greeting groups in everyday test, below the8 quota.
Old reader PID/through-exit memory/exit receipt were not captured and stay absent.
Original binding, failures, progress and code bytes remain retained.

Repair1 changes dialogue identity to the ordered list of ALL normalized user
turns (NFKC/casefold/whitespace). Everyday has119 test/270 train full-sequence
groups while still having5 distinct first greetings in each shard. This is a
declared data-variable change, not a successful rerun of the old grouping.
All624 selected full-user groups and canonical prefixes are unique. Complete
official test groups are excluded from selected train;881 train rows collide
under the new identity. Old279 individual user-query keys remain excluded.
This verifies exact identity disjointness, not semantic independence or absence
from donor pretraining.104 new and64 old RESERVED cases remain unqueried.

## Actual cost and terminal state

Two original downloads total328,261,485B and are reused in place; new network
bytes0. Repair reads all54,948 test/115,991 train rows. Arrow24 runs in an
isolated child; canonical Rust tokenizer/Jinja adoption runs in the parent,
without importing Arrow there. Source/model/optimizer/GPU calls and reserved
queries are all0. All admission/input/resource gates PASS.

Family40.000s, worker38.719s; launcher26968/worker2080/reader24828, both children
exit0. Held worker peak216,903,680B; reader held peak230,424,576B through exit,
reader31.657s. Terminal receipt records8 output files/17,671,287B.
Session93404 is closed; no model/T4 worker remains.43 bound inputs are checked
before and after the family. Selected runtime hashes are not a full DLL-tree
certificate or independent Parquet/BPE implementation.

## Prepared next observation, not part of this adoption run

After terminal completion, `chatbot_broad_capture_cases.py` generated
`broad_capture_cases.json` in the adoption directory. This added file is a
separate post-adoption artifact, NOT one of the eight terminal-receipt outputs.
SHA256 `14050c2ce63c39b205bc007abc8b41933747ffb594b530ff3bf7375fa30b34e4`.
It fixes48 cases:12 sources times2 FIT/2 DEV,25th/75th eligible length ranks,
ID tie break, without model observations. Actual input lengths34-1252;
max256 new IDs and2304 total, up to1,610,637,312B full BF16 logit packets.
Longalign is deferred explicitly;576 other adopted inputs are unqueried.

[Next capture and controlled pilot](CHATBOT_BROAD_CAPTURE_NEXT_20261009.md)
preserve the compact original-LUT/ternary/SSM target. At this record's creation,
the separate capture worker/binding/execution are still UNEXECUTED. Broader
coverage is a testable transfer variable, not a demonstrated solution to
representation, sparse selection, quality or accepted50.
