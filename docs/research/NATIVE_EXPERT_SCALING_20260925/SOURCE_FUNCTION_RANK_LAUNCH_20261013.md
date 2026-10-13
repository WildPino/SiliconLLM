# Bound FIT response-rank producer launch

Source/protocol/storage freeze 9b5b6e1 (initial f0dd108 preparation fault retained).
Binding source_function_rank_binding_v2_20261013.json, 106024B,
SHA256 c496a060d1098f1bea0ecf34b86d21e534803f29573a39132138b2c7b4d90115.
Preparation CLOSED0 with separate preparation terminal. Source goal remains active.

PowerShell environment OPENBLAS_NUM_THREADS=6, OMP_NUM_THREADS=6,
MKL_NUM_THREADS=6 before Python. Execute from repository root:

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_function_rank.py --run --freeze 9b5b6e1 --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_function_rank_binding_v2_20261013.json --binding-sha c496a060d1098f1bea0ecf34b86d21e534803f29573a39132138b2c7b4d90115 --directory E:/SiliconLLM/results/native_expert_scaling/source_function_rank_20261013 --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_function_rank_result_20261013.json
```

No source/GPU/native model rerun, DEV array input, kernel/quality/timing admission.
Caps source900s/audit1200s/OS6GiB/new E1GiB. All numeric and complete stored-data
audits required regardless of poor response tails. Auditor invocation will bind
actual report/terminal hashes after producer completion, before execution.
