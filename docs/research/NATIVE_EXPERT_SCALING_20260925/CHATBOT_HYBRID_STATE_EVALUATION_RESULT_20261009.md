# Complete fixed-state evaluation after interrupted whole recovery

9 October2026. TERMINAL/exit0,diagnostic recovery FAIL. Goal ACTIVE/INCOMPLETE.
Implementation/protocol freeze6966666540d951152cf0fe0b54d8d409758f4304;
[181-input binding](chatbot_hybrid_state_evaluation_binding_20261009.json)
SHA4ab6f86a8b07ea0c5d9bc04723b426ec1ecccc16059d3d9d05ee3bf3f1dfa7a3.
[Protocol](CHATBOT_HYBRID_STATE_EVALUATION_PROTOCOL_20261009.md),
[raw result](chatbot_hybrid_state_evaluation_result_20261009.json),
[terminal](chatbot_hybrid_state_evaluation_result_20261009.terminal.json).

## Actual complete result

Reused audited checkpoint286,Adamstep294,
SHA7b95e699a57c9bc0ed320bef016d95ba78ac1df2e79b9ea98a0c12a1ba78eec5.
No optimizer restoration/update or source/C call. Original no-grad AQ63,
ternary forward,F32/TF32off,case-zero/continuous SSM/SWA. All160/5746 NEW
full-vocabulary rows retained at this checkpoint. Earlier initial rows were
from checkpoint8 and were not replayed.

| Metric | Audited initial FIT | Final FIT | Audited initial DEV | Final DEV |
|---|---:|---:|---:|---:|
| Equal-case KL |24.0622705392|1.5134956062|24.7118650042|3.0809806443|
| Label-weighted KL |25.7844720289|1.4595673595|26.0431073407|2.3556498050|
| Greedy disagreements |4634/4643|1557/4643|1101/1103|459/1103|
| Disagreement fraction |99.8062%|33.5344%|99.8187%|41.6138%|

Final160/support and both relative recovery gates PASS. Four absolute/category
quality gates FAIL:DEVcaseKL>1,disagreement>20%,all-domainKL<=2 and all-domain
disagreement<=35% not satisfied. Decision INTERRUPTED_STATE_PREFIX_RECOVERY_FAIL.
Reported disagreements are next-token choices on donor prefixes,not a fraction
of semantically incorrect answers. No fresh own-history generation was measured.

| DEV domain | Equal-case KL | Disagreement |
|---|---:|---:|
| arithmetic |1.613533|110/316=34.8101%|
| code |1.893513|111/236=47.0339%|
| explanation |2.580093|71/195=36.4103%|
| history |3.005956|7/14=50%|
| instruction |4.415980|27/59=45.7627%|
| planning |3.753294|91/199=45.7286%|
| reading |3.542373|8/13=61.5385%|
| rewrite |3.843103|34/71=47.8873%|

All rows finite;CPU/GPU greedy IDs exact. Stable NumPy F64 source-bit-lift KL
agrees with GPU F32 per-label reductions within maximum5.346938579e-6 against
fixed1e-4. Different arithmetic in the same worker,not independent libraries.
New final support counts exactly input IDs*8/site/domain;atleast8 banks/site.
Missing historical training support remains missing;original512 run incomplete.

## Actual resources, command and decision

ONE family471.969s,worker OS peak through exit5,386,641,408B,
GPU allocated peak1,406,876,672B/reserved2,376,073,216B. All fixed resource/input
gates PASS;no faults/extra resource/T4. Session62892 andPIDs8596/27536 terminal.
Isolated Python3.12.10/shared launcher,args:

```
--binding docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_state_evaluation_binding_20261009.json
--binding-sha 4ab6f86a8b07ea0c5d9bc04723b426ec1ecccc16059d3d9d05ee3bf3f1dfa7a3
--freeze 6966666540d951152cf0fe0b54d8d409758f4304
--directory results/native_expert_scaling/chatbot_hybrid_state_evaluation_20261009
--out docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_state_evaluation_result_20261009.json
```

Actual learning substantially changes the model's output distribution,including
DEV recovery;it does not pass the registered calibration gates. This is not a
general impossibility or a completed test of512 updates. Defer automatic extension
of the same small-template recipe. Follow the [controlled group result](CHATBOT_HYBRID_GROUP_SUM_RESULT_20261009.md)
and [new construction proposal](CHATBOT_HYBRID_SHARED_PRIVATE_NEXT_20261009.md).
Do not promote to the prepared full-recovery native cohort path.

128 FIT/32 DEV,eight related templates,23 partial source replies.64 RESERVED
unqueried. Final fresh chatbot quality+>=50 accepted batch1 IDs/s on the same
original-LUT/ternary/SSM artifact,useful larger n/CPU routing mass/DRAM and actual
other family/scale demonstrations are still missing.
