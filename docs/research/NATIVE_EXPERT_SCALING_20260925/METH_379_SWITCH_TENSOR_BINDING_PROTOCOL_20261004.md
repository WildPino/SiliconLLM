# METH-379: independently trained original base128 tensor binding

Prospective. Freeze and execute only after complete METH-378 PASS is retained.
Resolve whether the acquired original source actually supplies the expected
7.415B tied architecture and real learned bank parameters. Reuse METH-325
original archive/header identities and METH-327 whole-tensor binding machinery.
The changed source is independently pretrained `google/switch-base-128`, revision
`86c815ec05361a33a8b49fc717277da9c0a4e711`, not a base256 subset.

Three whole acquired archives must be rehashed against official LFS SHA256,
with exact size and complete acquired record. Qualified Torch2.6.0+cu124 built-in
weights_only/mmap CPU loading, one shard and CPU thread at a time, no GPU or
checkpoint code. Every source tensor must match the fixed325 namespace, F32
dtype, shape and stride; contiguous CPU tensor bytes hashed in 4MiB chunks,
all values finite, minimum/maximum recorded. Require3320 names and the declared
index byte sum. Four embedding/head copies must have identical shape and
canonical tensor SHA256. Architecture-unique count after the three confirmed
duplicate aliases must be7,415,217,408 parameters.

Require12 sparse banks, each labels0..127 with WI[3072,768]/WO[768,3072].
Hash fixed operation/dtype/shapes and both actual tensor hashes into a canonical
pair identity excluding bank/expert labels; report distinct counts per bank.
Distinct pair counts are observations, not a quality threshold. Parameters
alone do not prove function inequality or useful knowledge. Record the total
original expert-pair parameters separately from tied architecture capacity.

MAIN20min, sampled RSS<=16GiB every64 tensors, streaming archive chunks bounded
by the same wall guard. Expected<=3min based on327's103.531s for the larger
59GB source. No concurrent model/native timing or download, no T4/training.
Keep first partial failure before any changed experiment. Fresh raw output,
all code/protocol/header/acquisition/common-helper files committed and physical
bytes exactHEAD; acquisition digest recorded. Library identities recorded.

PASS licenses a separately frozen source-specific native export and independent
numerical/reference qualification. It supplies no inherited363 quality, no
subset371 equivalence, no accepted-token rate and no causal n gain. New whole
original-primary prediction/generation/task quality and SAMEartifact rate
remain necessary. This same-family twofold scale test leaves useful>256,
hierarchical LUT/routing, physical DRAM, other families and~100B open.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth379_switch_tensor_binding.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth379_switch_tensor_binding_result.json
```
