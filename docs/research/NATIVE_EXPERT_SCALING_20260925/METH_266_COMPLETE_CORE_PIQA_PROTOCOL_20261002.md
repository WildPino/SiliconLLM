# METH-266: full same-archive PIQA regression

Freeze before observing any task score from the259 candidate. Execute only
if all265 full-prefix generation-health/K64 gates pass; bind its completed
raw SHA on the command line and require the same259 artifact/source/parents/
children/export/helpers. No refitting or task selection.

Reuse the original21/90 task scoring apparatus on all1838 cached PIQA
validation items, source SHA
`93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d`,
labels SHA `b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb`.
This is a repeatedly used regression task, not a new independent sample.
Archive all per-item token hashes, labels and both option scores.

Three arms in265 order: BF16 donor, original BF16 centered E1280, actual
saved259 core and mapped unique bank. Full BF16 head/no KV cache; unchanged
EOS+separately tokenized `Question: {goal}\nAnswer:` and space+solution.
Primary lower mean suffix NLL chooses option, exact tie option0; total NLL
secondary descriptive only. Candidate relative to each control must have
accuracy delta >=-.02 and paired-bootstrap lower fifth percentile >=-.05.
Reuse seed212121/20000 draws and90/122 thresholds. Failure closes the fixed
candidate path; no post-observation threshold/sample/precision adjustments.
Pass still requires separately frozen anonymous semantic review.

Local RTX3060/six threads, deterministic/highest/TF32off,70min after
imports,20GiB RSS/10.5GiB GPU; no T4/download. Run after265 exits, preserve
per128-item partial progress and failures. This is quality evaluation, not
a speed benchmark or shortlist NLL approximation. Native cache/whole-model
accepted rate, actual large-RAM n/CPU route/LUT/DRAM and multiple donor
families/scales remain unqualified.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth266_complete_core_piqa.py --generation-sha <completed265SHA> --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth266_complete_core_piqa_result.json
```
