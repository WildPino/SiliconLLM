# METH-329: complete original14.7B source in native bridge

Prospective AFTER326 complete source and327 all actual tensor/alias/bank binding,
328 full tiny native encoder/decoder/cache/head gates PASS. Resolve actual source
native mapping and full-size numerical semantics before precision/LUT quality/
rate investment. Existing high-active Giga319 remains closed. No model shrink,
synthetic/duplicated source experts, source config injection or quality waiver.

Original google/switch-base-256 cdac1724c078ea4974b4087c59634799561e7835,
ALL14,664,154,368 unique architecture parameters/12x256 actual source banks.
Fresh exact SHA256 all acquired326 files, same original config/capacity64/F32,
local bound tokenizer/sentinel mapping(0pad/1eos/32099first/32098second).
Exact readonly F32 repack: small manifest only, no weight copy/transformation.
All6392 tensors mapped into original6ZIP storage records via local headers and
original tensor storage offsets; every bounds/compression/alignment checked.

New opt-in engine SILICON_SWITCH_F32_SOURCE_BINDING wraps unchanged328 C
arithmetic; adds native Windows BCrypt SHA256 of EVERY actual mapped tensor.
Require all names/shapes/lengths/byte SHA EXACT327, including all unselected
experts and four confirmed tied copies. Trim native working set every64tensors
during audit only to bound resident pages; no residency/speed inference.
Freeze source/controller/engine/protocol before compiler/audit/observations.
Same pinned clang/libomp/one CPU/no fast math as328, BCrypt linked explicitly.
Main wall20min AFTER imports, combined checked controller+child32GiB,120s
compiler guard. Report imports unmeasured; no startup-time performance claim.
Source hashes/model loading guarded at stages, native children sampled250ms.
No overlapping CPU timing/GPU. Exceptions preserve partial audit/cases/logs.

Official unmodified4.57.6 source324, Torch2.6 weights_only/mmap CPU source loader,
META construction +per-shard assign=True, all names/all originalF32 parameters
required, no META remainder; ties restored only AFTER actual aliases verified.
No checkpoint code/trust_remote_code/remote tokenizer or model fallback.

Fixed consumed engineering controls, not heldout quality:
1. Source: The capital of France is <extra_id_0>.
   Forced decoder: start0 + <extra_id_0> Paris <extra_id_1>
2. Source: The scientist measured <extra_id_0> after the experiment.
   Forced decoder: start0 + <extra_id_0> the temperature <extra_id_1>
Source/decoder IDs derived from verified original tokenizer, recorded; both<=64
so source capacity unsaturated, cached decode per-call semantics preserved.
Official encoder once then cached forced decoder steps; native same caches.

Require pooled encoder/decoder/full32128logit relativeL2<=1e-4, independently
ALL encoder initial/block/final and every decoder initial/block/final state
<=1e-4, nonzero reference norms, exact greedy top1 EACH forced step. Ordered
route choices/acceptance exact; probability maxabs<=1e-6. Whole zero-head
fault for each case must exceed1e-4. Tiny328 retains other semantic faults;
no need to assume pretrained relative/probability effects above that threshold.
All gates required, all numerical failures recorded without threshold repair.

Command isolated reference Python:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth329_switch_full_source.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth329_switch_full_source_result.json
```

PASS permits separate compact precision/LUT/native cost and whole quality work;
F32 source-compatible bridge alone does not conclude goal. Final SAMEartifact
heldout/generative/task quality and>=50accepted end-to-end, useful larger-n
selection/DRAM cost, actual128 comparison/other families/~100B remain unproved.
