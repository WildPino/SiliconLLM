# METH-325: immutable manifest and actual source tensor metadata

Prospective, after324 ALL gates PASS.321 shapes were inferred; resolve actual
tensor dtype/shapes, complete shard namespace and physical storage layout before
large transfer. Fixed base256 primary14.7B, base128 comparison, exact321 revisions,
configs/indexes unchanged. Reuse qualified official reference; no quality scoring.

Retrieve pinned-revision HF model API with blobs=true, require exact commit,
all official index shards present, exact size and64hex LFS SHA256. Preserve raw
metadata. Inspect ALL6+3 original .bin shard ZIP metadata via HTTP Range only:
strict206, exact Content-Range/whole-size, original redirect x-repo-commit and
x-linked-etag matching LFS SHA. Whole-shard fallback rejected before body read.
Fixed64KiB aligned windows for ZIP central directory and data.pkl records only;
windows may contain incidental adjacent payload bytes, explicitly not zero
weight-byte acquisition. Bound each read2MiB, each pickle2MiB, metadata4MiB,
total network body32MiB/200requests/600s, endRSS1GiB. No full source weights.

Structural pickle reader supports only OrderedDict, known tensor rebuild
functions implemented locally, and symbolic supported Torch storage kinds;
all other globals rejected. No checkpoint Python/Torch code executed. Accept
only CPU storage tags, positive integer dimensions, valid stride/offset bounds,
stored/uncompressed ZIP records. All actual names equal official shard index;
all shapes equal321 map and all actual dtypes F32 for these original base donors.
Every declared storage exists exactly in archive and byte length equals dtype
width times elements. Complete names/serialized F32 byte total must equal index.
Record storage aliases within each shard, sum physical storage and archive size.
No total unique-weight claim from metadata alone; full tensor hashes still needed.

Freeze source/protocol before observations; any exception keeps completed shards,
ranges/header digests, metadata, stage/error and partial result. No silent resume,
retry/version/gate/source changes. Separate experiment needed for apparatus repair.
Only ALL models/shards pass permits a bounded full acquisition protocol. LFS
hashes are expected whole-file checksums until full transfer verifies bytes.

Command:
```
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth325_switch_source_headers.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth325_switch_source_headers_result.json
```

No native engine/routing/LUT/quality/rate conclusion. This prerequisite preserves
real source banks for later conversion and does not relax>=50accepted tokens/s,
donor-relative whole quality or useful larger-n/cross-family/~100B requirements.
