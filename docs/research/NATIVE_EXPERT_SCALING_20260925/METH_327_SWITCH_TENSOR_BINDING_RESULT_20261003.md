# METH-327: all source tensors and expert parameter tuples bound

**PASS, source binding scope.** Freeze2e05f67, observations9df3c97.
[Raw record](meth327_switch_tensor_binding_result.json) SHA256
e72fc17b5527dc34df5c00ed7c44498b2ee4fc5d338ed8b130fff101a30c20a7.
[Protocol](METH_327_SWITCH_TENSOR_BINDING_PROTOCOL_20261003.md), driver
benchmarks/native_expert_scaling/meth327_switch_tensor_binding.py.

Every six original whole files freshly SHA256-verified versus325/326, built-in
Torch2.6 weights_only/mmap CPU one shard at a time. ALL6392 tensors exact source
namespace/shape/stride/F32, contiguous, finite; actual complete tensor bytes
SHA256/min/max recorded. All four embedding/head physical copies have EXACT
same shape and digest; architecture ties now confirmed from acquired weights.
Actual serialized14,738,177,280 elements include these copies; subtracting three
tied copies gives **14,664,154,368 architecture-unique parameters**, matching321.
This count is not global numeric weight deduplication or useful knowledge size.

Every12 sparse groups has every256 source labels, exact sourceWI[3072,768]/
WO[768,3072]. Each group has **256 distinct canonical F32 parameter tuples**,
3072 total group-local tuples. Hash includes fixed operator/shapes/actualWI+WO
digests, not expert/group label, so copied weights would be detected. Different
parameter tuples do NOT prove distinct mathematical functions under hidden
permutations/scalings, real-state response diversity, exposure or useful capacity.

103.531s **main after imports**, maximum checkedRSS10,334,208,000B (not continuous
peak), endRSS416,096,256B. No GPU/model inference. All tensor/source hash checks
within20min/16GiB main guard; external Python import startup unmeasured.
Exec35705 exit0. Raw preserves source hashes and complete bank tuple list.

Permits exact source mapping/full reference controls and separate precision/
LUT path. Donor-relative untouched/generative/task quality, useful n scaling,
native routing/DRAM cost and SAMEartifact>=50accepted rate remain unproved.
