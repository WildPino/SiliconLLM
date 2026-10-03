# METH-342: source-only new span-reconstruction quality cohort

Prospective AFTER341 six-thread complete cost-margin PASS, BEFORE original
donor or target scoring. Uncertainty: does the actual native compact artifact
retain pretrained span-reconstruction quality on new project sources? This
stage makes the cohort/known answers real; inference and gates must be frozen
separately before scoring. No quality observation or candidate tuning here.

Existing local PG19train14 parquet308,888,290B SHA
`15f3ecb5e314ce58bbf8eae927b51bf893b9fdabe4a04c8e246547206357a2d5`,1243 rows.
Source originally cached, no external acquisition. Bind341 raw PASS and pinned
326 original local tokenizer/config bytes, isolated official4.57.6. No model
weights loaded/forward/generation, no GPU/training/download.

Enumerate ALL tracked JSON under docs/research and benchmarks/donor_adaptation
with physical byte SHA. Exclude every exact PG19train14 source ID book row
and conservatively every source_row reference in records containing this
corpus path. Store all hashes/excluded IDs. Whole-book exclusion, no new
fragment from an already recorded book. This is heldout from these tracked
project observations; original donor-pretraining membership remains unknown.
Untracked/other-format undocumented source use is not certified absent.

Hash rank remaining1243 row IDs with fixed seed
`meth342-native-switch-new-span-342342`; inspect first96 candidates, select
first24 passing SOURCE ONLY criteria:>=8192 chars; fixed hash-positioned4096
char excerpt; original tokenizer >=512 tokens, >=128 distinct token IDs,
no unexpected special tokens (all IDs2..31999). Rejections recorded, never
depend on original/target scores. Stop if fewer24; no replacement after scores.

Each selected book supplies four independent token windows, one per excerpt
quarter:63 original tokens, hash-positioned within quarter. Remove one8-token
span at starts12/20/28/36 and insert <extra_id_0>; append original EOS.
Encoder57 tokens. Known reconstruction target <extra_id_0>+original8 span
tokens+<extra_id_1>+EOS,11 tokens; forced decoder pad+target[:-1]. Record raw
excerpt/whole source hash/text/IDs/masked span/target and per-vector hashes.
These are known original answers, no instruction/chat prompting or anonymous
answerability rubric needed. All24 rows disjoint from recorded old source IDs;
96 examples clustered by book, never treat them as96 independent books.

MAIN10min after imports/RSS4GiB, source-only CPU, one fresh raw result or
failure; all code/protocol physical HEAD before observations. Output manifest
committed before any later inference. Quality/generation/task criteria and
original PRIMARY teacher controller must be frozen separately BEFORE scoring;
341 forced cost is not accepted generation rate or quality proof.

Command:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth342_switch_fresh_span_manifest.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth342_switch_fresh_span_manifest.json
```
