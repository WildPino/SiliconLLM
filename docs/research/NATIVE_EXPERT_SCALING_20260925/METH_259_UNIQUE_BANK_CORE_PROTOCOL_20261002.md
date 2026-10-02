# METH-259: frozen complete core with actual unique functions and route aliases

METH-258 proves1243..1280 effective codes per layer and preexisting aliases.
Change physical storage to honest unique dictionaries with alias lookup.
METH-257's all10-distinct export stays closed. No refit, new learned
function, rank/dtype/route adjustment or relaxed same-variant quality gate.

Bind258 raw SHA
`e5069a0ed59938562ec7d995e26391d812dd7fde8c94e7d208230c98b9525311`,
require all gates. Pin original252/253 raw/source fixture,211 core/export,
parent training/checkpoint, child checkpoint and original125 source states
through fixed constants. Locally cached config only, no donor model/core
acquisition. Use same six-thread/RTX3060 deterministic/highest/TF32-off
centering and BF16 execution casts as the original existing conditional bank.

## New complete artifact format

Format `M253_COMPLETE_CORE_UNIQUE_BF16_CORRECTIONS_ROUTE_ALIAS_V1`.
Keep all220 non-FFN/head-proposal fields and361 qualified source FFN
segments unchanged, including exact tied BF16 head and all24 private32/
mixed32/rank32 row-Q8/LUT FFNs. As in257, no group-Q8 FFN retained.

Per layer store six conditional fields: shared BF16 parent A[128,8,896],
unique BF16 B[n_actual,896,8], unchanged FP32 router/projection/child keys,
and int32 leaf_map[1280]. Traverse original leaves0..1279, canonicalizing
code SHA(A,B), or one exact-zero code independently of A; assign dictionary
index in first-occurrence order. Require every per-leaf code matches258,
dictionary count matches258, no duplicate code, and leaf_map is surjective
and in range. Most importantly, B[leaf_map] must be bitwise equal to all
original BF16 B values. Parent A equality/shared-sibling equality exact.
Hash/record representatives, codes, six tensor fields and all bytes.

The map uses int32 for portable actual PyTorch indexing,5,120bytes/layer,
122,880bytes total. This added indirection is not yet a qualified native
cost. Store one B for each composite code, not globally deduplicating B
with different A while falsely counting it as one function. A selected
parent still supplies its original A; canonical exact-zero aliases store
the same all-zero B. No function/source noise added to fill a bank.

725 tensors, actual payload/file each<1.6GB; all finite and readback exact.
Archive includes original config, actual function-count list, source/bank
provenance and used helper hashes. Loader constructs BF16/SDPA with
seed259259, deletes72 random FFN constructor parameters before execution,
copies218 non-FFN parameters exactly, consumes all725 fields and checks
tied pointer, maps/counts. No source weights/checkpoints loaded by loader.

## Exact execution controls, no quality targets

Use identical qualified stored source FFN before the unchanged conditional
bank in both control/candidate arms. Keep original top4 product-key routes,
scores/softmax, child-key argmax, exact conditional SiLU, BF16 A/B sums.
Only B address is B[leaf_map[selected_child]], A remains A[child//10].

For all256 original stored BF16 states in each of24 layers,6144 vectors:
original source fixture vs saved-loaded FP32 FFN outputs bitwise equal,
BF16 cast outputs bitwise equal, original parent IDs/scores/child IDs
bitwise equal, selected mapped BF16 B values exact, and complete old/new
conditional BF16 outputs bitwise equal. Old bank is the original FP32
checkpoint plus its actual BF16 execution conversion; this qualifies the
new dictionary/map arithmetic without comparing the changed core to full
donor quality. Proposal tensor bits unchanged. No target/logit/document/
generation/task scoring in this phase.

## Resources and next eligibility

Local RTX3060/six host threads,20min after imports,20GiB RSS/10.5GiB GPU,
>=4GiB free disk. Preserve artifact before loader checks and failure rows.
Freeze code/protocol before execution once. All apparatus gates pass only
licenses a separately frozen consumed-development full-model screen,
then genuinely new independent source prediction/generation/task tests,
whole-engine native parity/routing/LUT/DRAM and accepted>=50 batch1tok/s
on this same artifact. No inferred rate from components. Nominal1280
routes, actual30,556 unique layer-local codes, copied pretrained source
rows and useful semantic capacity remain distinct quantities. Arbitrary
RAM scaling/large-n quality, sparse GigaChat/10B/100B and second-family
transfer remain open. Source-prefix2539.318ms remains only its component.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth259_unique_bank_core_export.py --artifact results/native_expert_scaling/meth259_qwen05b_unique_source_core.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth259_unique_bank_core_export_result.json
```
