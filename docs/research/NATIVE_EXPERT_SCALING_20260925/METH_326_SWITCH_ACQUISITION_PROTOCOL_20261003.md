# METH-326: complete original14.7B source acquisition

Prospective, after324 reference and325 all actual header gates PASS. Resolve
missing real learned weights for exact repack/function binding/source baseline.
Primary source only: google/switch-base-256 revision
cdac1724c078ea4974b4087c59634799561e7835. Frozen325 six-shard sizes/LFS SHA256,
58,856,206,797 archive bytes. Acquire original seven fixed side files(config,
index,generation config,special tokens,SentencePiece,fast tokenizer,tokenizer
config) from the same immutable revision, then six original shards in name order.
128 comparison30GB deferred until primary native feasibility; no substitute.

Freeze driver/protocol before transfer. Local output under
results/native_expert_scaling/meth326_switch_base256_source; fresh-only run,
no source overwrite. Initial free disk must cover all declared files plus32GiB
reserve (currently~498GB free). Sequential1MiB streamed blocks, request15s
connection/30s read timeout,90min overall/64GiB cumulative download cap.
Disk reserve checked per file and30s progress intervals. Memory bounded by
one streamed block/hash buffers; no Torch/model/GPU or concurrent CPU timing.
No blind restart/network retry or revision changes. Interrupted .partial files
and complete earlier shards retained with failure record; any later resumption
requires explicit immutable-prefix checks and separate record/protocol.

Require original resolve header commit and LFS expected SHA; acquired exact
size and complete SHA256 must match325 for EVERY shard and SentencePiece.
Small Git files require exact Git blob SHA1 from preserved325 API; config/index
additionally exact321 SHA256. Record actual SHA256 for every side file too.
Only verified files rename from .partial to final. ALL files must pass; no
model code loaded, no trust_remote_code/checkpoint pickle executed here.

Only complete acquisition pass permits separate all-tensor/function hashes and
full source-reference controls. Quality cohort untouched and not yet selected.
Original banks are real source availability, not yet proof of effective diversity,
preserved knowledge, CPU LUT/routing cost or>=50accepted tokens/s. Engine bridge,
full source quality, precision qualification and useful n scaling still required.

Command:
```
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth326_switch_acquire.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth326_switch_acquisition_result.json
```
