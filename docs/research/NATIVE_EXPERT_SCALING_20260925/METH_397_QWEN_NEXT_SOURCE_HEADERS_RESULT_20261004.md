# METH-397: original Qwen3-Next headers and active-cost result

Frozen protocol/controller at `2e7eb13`; procedural repair only after396 first
physical manifest checkout failure retained460db6c. Execution exit0, all five
metadata gates PASS. This is source applicability evidence, not model inference.

## Identities and complete acquisition boundary

Original `Qwen/Qwen3-Next-80B-A3B-Instruct`, revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. Config SHA
`2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303`;
controller `1a5e71410e446730a8a11d290cb2ccfaef60ed5696682d794d009acbf3bb85ee`;
protocol `931732236a05f08319e0f725719359c45696133d357080d230c5c823ffe5c07c`.
Raw `meth397_qwen_next_source_headers_result.json`, SHA
`5519e6b95f4a47d333fc662ab4d0e24126ef3f58f18dad08163575d53ed5d1f0`.
The raw record binds every downloaded metadata/header and declared source shard.

ALL41 actual safetensors headers /75,944 tensor names reconcile index Git blob,
config Git blob, dtype sizes, contiguous offsets and declared LFS extents.
ALL48 main banks have512 original gate/up/down tensor slots, shapes512x2048,
512x2048,2048x512. This does not verify value uniqueness or usefulness.
No source weight value payload downloaded. Declared whole-file LFS SHA values
are not hashes verified against acquired whole archives.

## Exact source and representation ledger

| Quantity | Count / bytes |
| --- | ---: |
| Main decoder tensor-name parameters |79,674,391,296 |
| Extra MTP tensor-name parameters |1,650,471,424 |
| All BF16 tensor-name parameters |81,324,862,720 |
| Actual tensor-value extent |162,649,725,440 B |
| Declared41-shard total, including headers |162,659,161,528 B |
| Main active parameters including lookup/head/router/controls |3,563,766,528 |
| Hypothetical main stored additive representation |21,443,340,288 B |

Both original BF16 and FP32 full-resident representations exceed this host's
85,845,110,784B total RAM. A streamed/offloaded original reference is unqualified,
not ruled out. No complete original runtime or packed-loader conversion tested.

SAME317 hypothetical full-width representation, selected10 per48 banks:

| Addressed descriptor | Bytes |
| --- | ---: |
| Two U8 indices per8 coded coefficients |800,194,560 |
| F32 row scales |9,102,336 |
| Independent palettes |7,127,040 |
| Full Q6_K head |255,252,480 |
| Full F32 routers and shared gates |201,719,808 |
| Other F32 controls |5,565,440 |
| One BF16 embedding row |4,096 |
| **Total** |**1,278,965,760** |

Total exceeds the frozen560,000,000B yardstick. The coded matrix coefficient
count is3,200,778,240. Source/header arithmetic is measured; this representation
is hypothetical and untrained. Addressed bytes are neither physical DRAM nor
latency. Core/head/router are included. Optional MTP stays separately accounted;
excluding it from standard decode is not pruning main pretrained capacity.

Old config-only main formula79,573,616,640 is100,774,656 below actual main count.
The complete81,324,862,720 manifest is now reconciled by main+MTP. The old formula
also misses the full-attention output gate projection:12*(4096*2048)=100,663,296;
shared sigmoid gates48*2048=98,304; Q/K norms12*2*256=6,144; DeltaNet norm36*128=
4,608; DeltaNet A_log/dt_bias36*2*32=2,304. These omissions sum100,774,656, exactly
the residual. Header shapes and donor_inventory.py's existing projection/norm
formulas identify these terms; do not use that old formula as exact source cost.

## Cost, decision and limitations

MAIN133.468s; maximum checked RSS116,895,744B;85 HTTP requests and16,655,410B
response bodies. Limits10min/1GiB/64MiB/250requests all met. Local metadata:
`results/native_expert_scaling/meth397_qwen_next_source_headers`. No model,
native timing, GPU or T4 run. Command unchanged frozen protocol:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth397_qwen_next_source_headers.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth397_qwen_next_source_headers_result.json
```

**Decision:** close unchanged full-width additive active geometry before full
source acquisition. Retain metadata for a stated new geometry; no generic80B
port or rejection of all transformations. Top-k renormalization is source-
specific and finite-precision/ties still require a numerical contract. Useful
larger n, actual LUT/DRAM, original-relative quality and same-artifact complete
rate remain open. Qualified original Switch128/256 artifacts remain intact.
