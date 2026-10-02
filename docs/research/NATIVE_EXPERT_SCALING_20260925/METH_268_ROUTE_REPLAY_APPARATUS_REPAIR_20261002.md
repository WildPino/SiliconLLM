# METH-268 apparatus repair before first completed case

Initial session33649 exited1 before any completed case or persisted layer
measurement. Python treats `except BaseException as error` as a local
binding throughout main, shadowing the global relative-error function.
The first call to that function therefore raises UnboundLocalError. Source/
candidate first-prefix forwards and own-choice/first-layer same-input
identity guards ran, but no whole-case/replay result or summary was produced.

Preserve [failure](meth268_generation_route_replay_result.failure.json)
and empty-row partial. Change only exception variable name to `failure`,
leaving the numerical error function untouched. No input, sample, weight,
arithmetic, route-replay intervention, budget or guard changes. Freeze
this repair before rerunning once to a new output path:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth268_generation_route_replay.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth268_generation_route_replay_repair1_result.json
```

The initial stop is apparatus evidence, not a scientific route/candidate
failure or a reason to reopen267 semantic/native promotion.
