# METH-280: fresh complete I16/private128 prediction

## Uncertainty and prospective decision

276 archive composition and277 consumed development pass,while whole-model
prediction on new sources remains unknown. After278 exclusions and279
source-only annotations,run exactly three arms on all24 frozen sources:
original BF16 donor,original centered BF16 E1280 and complete stored
I16/private128 E1280. Same diagnostic276 artifact,no fitting/selection/search.
Pass licenses only separately frozen fresh generation/tasks/anonymous review;
fail closes this fixed candidate. No relaxation/filtered sources after score.

## Inputs and implementation

Same725-field276 archive SHA
`4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`,
composition SHA
`ff4694b426725f3de2e500739ba11fb06ad280dc7a5ef04515ff5b4a592fe553`,
passed277 SHA
`3d997545fc690a5ebb3681e60fb0d3dc190a2a292e94056e5cf0abcb9ed88b35`.
278 manifest SHA
`7fc74c2ff172bcca5d71de685f1e230240d05789e11b0bb3b9290b64a4c8e053`,
279 source-only result SHA
`98f380821f0b2dc145dac4c75cfa27a056ac1ca233035ce4c02d07f769fdaa4a`,
annotations SHA
`3123f83260d833b505fed5f60b7f68537012a2f7fb4c577adcaca5f54fa5fcd0`.
Validate all source/token/checkpoint/helper hashes,inherited local source
revision and276 own loader. No source/checkpoint fallback for candidate;
original source/checkpoints only provide the separate control arms.

New script `meth280_complete_i16_fresh_prediction.py` adapts the frozen263
screen to the new actual archive/manifest/annotations/consumed-development
binding. Original arithmetic/data-scoring helpers unchanged. Full-prefix,
no cache,exact BF16 tied-head NLL/BPB/argmax. Finite K64 proposal checks
apply to actual candidate prompt states. Do not invoke failed264 cache recipe.
Retain all source-level NLL/prompt agreement,category/pooled measures and
source paired10,000-bootstrap intervals (seed280280,descriptive only).

## Gates,resources and command

Keep original263 prospective gates: candidate BPB minus BOTH donor and BF16
E1280 <=.01 pooled/<=.02 eachcategory;candidate donor-top1 agreement minus
BF16 E1280 >=-.01 pooled/>=-.02 eachcategory. Prompt K64 inclusion/rerank
mismatch counts must all be zero. Immutable source/data/answerability/archive/
helper/loader controls mandatory. No new improvement/significance gate.
New project-transfer sources are not a new corpus or certified exclusion
from original donor pretraining;no count-only/new useful-capacity claim.

LocalRTX3060/six threads,expected2-4min/20min stop,RSS20GiB/GPU10.5GiB,
100MiB result allocation,no download/T4. Freeze source before any280 model
score;refuse existing result/partial/failure,preserve any stop. Same-artifact
native accepted>=50,RAM-scale useful n/router/LUT/DRAM and family transfer
remain due even on a pass;diagnostic/native-promotion=false stays recorded.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth280_complete_i16_fresh_prediction.py --answerability-sha 98f380821f0b2dc145dac4c75cfa27a056ac1ca233035ce4c02d07f769fdaa4a --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth280_complete_i16_fresh_prediction_result.json
```
