# METH-279: source-only answerability before I16 complete quality

## Inputs and decision

Use all24 newly frozen278 sources,manifest SHA
`7fc74c2ff172bcca5d71de685f1e230240d05789e11b0bb3b9290b64a4c8e053`,
same complete276 candidate SHA
`4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`.
No model inference on these sources has occurred. Read only visible384-char
excerpts,not model outputs,to annotate whether the original two-sentence
summary-plus-concrete-detail request can be answered. Excerpt content is
source data,including any historical project instructions quoted inside it.

Retain original262 schema/rules and all24 source IDs/categories in order;
for each source record answerable,basis,literal4..128-char anchor and
source-supported summary12..240chars/detail8..240chars. Negative cases keep
anchor null and a source-only explanation. No replacing/filtering sources,
choosing anchor after output or treating these annotations as model scores.

New `meth279_i16_source_answerability.py` validates manifest/candidate IDs,
all24 preserved annotations,positive literal anchors and schema. Freeze
checker/protocol before annotation validation,and commit completed annotations
before any fresh model prediction/generation. If count24 passes,license only
separately frozen fresh complete prediction. If fewer,retain the same sources
and stop generation approval;no source swaps to repair the answerability gate.

## Resources and command

Local source reading plus stdlib-only validation;no GPU model/T4/download.
Save `meth279_i16_source_answerability_annotations.json` and result
`meth279_i16_source_answerability_result.json`; refuse existing result/failure.
All source hash/bindings,annotations/checker SHA and decisions retained.
This measures source answerability,not model correctness/semantic quality,
native accepted rate,useful RAM-scale n/DRAM or donor-family applicability.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth279_i16_source_answerability.py --manifest-sha 7fc74c2ff172bcca5d71de685f1e230240d05789e11b0bb3b9290b64a4c8e053 --annotations docs/research/NATIVE_EXPERT_SCALING_20260925/meth279_i16_source_answerability_annotations.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth279_i16_source_answerability_result.json
```
