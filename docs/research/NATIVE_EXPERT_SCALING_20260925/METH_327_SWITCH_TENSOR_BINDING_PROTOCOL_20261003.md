# METH-327: whole source weights and actual expert parameter tuples

Prospective, execute only AFTER complete326 PASS committed. Reuse frozen325
actual shapes/storage metadata and qualified324 reference prerequisites.
Resolve missing actual learned weight binding, tied physical copies and bank
parameter diversity for original14.664B base256. No model/source scoring.

Every six whole files rehashed streaming4MiB against325 LFS SHA; exact size and
complete326 acquired SHA record required. Torch2.6 official built-in
torch.load(weights_only=True,mmap=True,map_location=cpu), no checkpoint code/
trust_remote_code or custom tensor functions. One shard at a time, one CPU
thread, no GPU; release state/views after each file. Exact entire shard namespace,
all actual F32 tensor shape/stride versus325, contiguous source required.
Stream tensor view4MiB chunks, SHA256 every actual canonical tensor byte and
finite/min/max checks.20min wall,16GiB checkedRSS every64tensors/shard; endRSS
also recorded. No concurrent rate/model benchmark. Partial records preserved
with all completed tensor/shard hashes on any failure; no silent source/gate repair.

Require all6392 serialized names and declared F32 byte sum. Four aliases
shared.weight/encoder.embed_tokens/decoder.embed_tokens/lm_head must have EXACT
same shape and tensor SHA; architecture-unique parameters after confirmed tied
copies must equal14,664,154,368, not physically summed duplicate storage bytes.
ALL12 source sparse banks, every256 labels, wi[3072,768]/wo[768,3072]. Canonical
parameter-tuple SHA256 over fixed operation/dtype/shape/wiSHA/woSHA, without
bank/expert label in hashed content; record actual distinct tuple counts per
bank. Diversity counts are observations, not a frozen quality threshold.

Different parameter tuples do not prove different effective functions (ReLU
permutation/scaling equivalences possible) or useful extra capacity. Real-state
responses/source baseline/heldout quality still needed. Full source/reference/
native conversion, tokenizer/generation conventions, precision fidelity and
measured routing/LUT/DRAM/accepted>=50 on same artifact remain separate gates.

Command:
```
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth327_switch_tensor_binding.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth327_switch_tensor_binding_result.json
```
