# METH-36: stored int8 router sketch passes independent route gate

The [prospective protocol](METH_36_INT8_SKETCH_PROTOCOL_20260926.md)
fixed the METH-35 rank-64/64-candidate rule before selecting
different inputs. Eight code, eight prose and eight technical
documents come from the pre-existing METH-17 selection, with
source IDs disjoint from METH-27. The selected [prompt
manifest](meth36_route_prompt_manifest.json) has SHA-256
`4bd2bcd666418bfc1b558a4ed6e57650dc31972b29f9a3f6671b99e573dad5ab`.
The R8 core and E128 adapter hashes were verified again. The
model stayed on its exact original route while the two shortlists
were audited at each input layer.

The 5,720,160-byte stored sketch artifact
`results/native_expert_scaling/meth36_e128_router_r64_int8.safetensors`
has SHA-256
`133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d`.
All 72 tensors passed exact save/reload comparison. It contains
5,505,024 fp32 basis bytes, 196,608 int8 router-sketch code bytes
and 12,288 fp32 row-scale bytes at E128. The relative squared
error of quantized sketch weights is `3.6773e-5`; route retrieval
is the decisive metric.

| Shortlist on new inputs | Exact top-4 IDs included | Full top-4 sets matched | Lowest category inclusion | Mean missed gate mass | Frozen gate |
|---|---:|---:|---:|---:|---|
| Rank-64 fp32 control | 589,457/589,824 = 99.938% | 147,089/147,456 = 99.751% | 99.920% | 0.000439 | pass |
| **Stored rank-64 int8** | **589,466/589,824 = 99.939%** | **147,098/147,456 = 99.757%** | **99.921%** | **0.000427** | **pass** |

The stored-int8 category inclusion rates are 99.921% code,
99.971% prose and 99.926% technical. The minimum layer
inclusion is 99.862%; there was no per-layer gate. The int8
arm happens to include nine more exact IDs than its fp32
control on this sample. Quantization does not establish a
general improvement; both clear the fixed ≥99.9% pooled ID,
≥99% pooled set and ≥99% each-category gate here.

The [export ledger](meth36_int8_sketch_export.json), SHA-256
`7c780f214c6a7d8f3647f2a60681ccf53da967768926f09cfb3df989f37fa1b9`,
and [route audit](meth36_int8_sketch_audit.json), SHA-256
`b3e0a2be71cc09f7547ec2841e26e85954be62a75bf3d663ba725797e0c23f54`,
retain source/artefact hashes, byte counts, per-layer/category
counts and route-stream hashes. Commands:

```
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth36_export_int8_sketch.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth36_route_prompt_manifest.json --artifact results/native_expert_scaling/meth36_e128_router_r64_int8.safetensors --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth36_int8_sketch_export.json
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth36_int8_sketch_audit.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth36_int8_sketch_audit.json
```

Export took 66.27 s CPU and ended at 1.378 GB RSS. The RTX 3060
route audit took 71.81 s, peaked at 2.275 GB allocated GPU
memory and ended at 3.310 GB RSS. No T4 was used.

**Decision:** the stored one-byte E128 sketch passes the
independent route-fidelity gate and warrants a route-replaced
language-quality audit and an exact C implementation with CPU
timing. It is not yet a useful-model or CPU-speed pass. At
hypothetical E273,547, sketch codes alone would address
420.17 MB/token across 24 layers, before scales, candidate
rescoring, selected experts and core. No distinct E273,547
experts were trained, and the parent R8+E128 model still fails
its generation gate.
