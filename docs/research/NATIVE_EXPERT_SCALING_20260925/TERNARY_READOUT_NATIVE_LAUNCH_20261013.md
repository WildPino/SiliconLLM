# Bound selected-p6 native LUT correctness producer

Source/protocol/storage freezeb043503;original engine AVX bodies exact,
two bounded compiler commandsCLOSED0/preparationCLOSED0. Binding
ternary_readout_native_binding_20261013.json,1218099B,
SHA25669d16f93ffa7122f34f402984ff6a1e1bcd698ac86f71d15a85760fd712b8447.
Environment OPENBLAS/OMP/MKL_NUM_THREADS6,PythonCPU0..5,C callsCPU0 only.

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/ternary_readout_native.py --run --freeze b043503 --binding docs/research/NATIVE_EXPERT_SCALING_20260925/ternary_readout_native_binding_20261013.json --binding-sha 69d16f93ffa7122f34f402984ff6a1e1bcd698ac86f71d15a85760fd712b8447 --directory E:/SiliconLLM/results/native_expert_scaling/ternary_readout_native_20261013/run --out docs/research/NATIVE_EXPERT_SCALING_20260925/ternary_readout_native_result_20261013.json
```

Run900s/audit1200s/OS6GiB/newE2GiB. All144 supplied-input histories/all3
sites,source control full F32 bits,new LUT full F64/BF16 scores,producer
scalar/table/mutation controls. No recurrence/source/GPU/T4/RESERVED/optimizer
or timing admission. Complete audit regardless of numeric/quality failure;
independent launch bound to actual report/terminal hashes after terminal0.
