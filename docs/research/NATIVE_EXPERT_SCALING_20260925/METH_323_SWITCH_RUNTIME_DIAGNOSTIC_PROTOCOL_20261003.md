# METH-323: capture all router/forward/capacity contracts

Frozen before observations,2026-10-03.322 first combined direct-router assertion
fails; raw6fab999e92752817932e81e3d6e304166421a312b1c995f3bc928b6d14ae102c
preserved039b6c9. Retain EXACT sourceSHA/seed322/Tiny config/router weights/
fixed inputs/three sparse shapes/full head-cache case and budgets from322.
No learned source weights/network/inference quality/native rate/GPU work.

Execution repair: convert first assertions to recorded BOOLEAN controls,
capture all three actual router shapes/mask values and expected mask. Continue
all three sparse/full-model calls even when direct gate FAILS. This cannot
turn failed322/323 gates into eligibility. Exceptions retained per case.

NEW diagnostic: independently compute per-sequence top1 from F32 classifier
softmax, argmax without extra singleton, one_hot[B,T,E], cumsum alongT and
capacity1. For every accepted token, apply its SAME actual tiny expert's dense
function times selected probability; dropped tokens outputZERO. This matches
the explicit contract in [official v4.57.6 source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/switch_transformers/modeling_switch_transformers.py).
We do not claim it has been qualified on pretrained weights yet.

Each sparse case must have nonzero reference, relativeL2<=1e-6 versus this
logical masked sum, and zero-output reference fault detected. Record mask,
error, returned shape/finite state and norm on dropped tokens. No favorable
tokens, capacity reset or changed threshold. This gate is prospective new
semantic evidence, not another reading of322's absent forward outputs.

All direct-router/shape/finite/logical-capacity/full-head-cache gates required.
Failure means current5.13.1 reference unsuitable under specified semantics;
separately qualify an isolated official older reference before checkpoints.
No global environment downgrade, source field change or final goal narrowing.
5min including imports/2GiB observedRSS, Torch1thread. Source/prior pins322
unchanged. Record all outcomes; metadata321 base eligibility remains scoped.

Command: `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth323_switch_runtime_diagnostic.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth323_switch_runtime_diagnostic_result.json`.
