# PQT-SRC-001: original Switch expert tensor admission

Prospective, 5 October 2026. Read-only source qualification; no fitting,
evaluation, calibration selection or native timing. The separate active checkout
must remain untouched. All output lives in the progressive worktree namespace.

## Objective and exact input

Admit original F32 weights for decoder block 11 experts 8 and 105 from
`google/switch-base-128`, revision `86c815ec05361a33a8b49fc717277da9c0a4e711`.
These IDs were identified by the inherited consumed routing calibration;
they are development selections, not a fresh random sample or holdout.
Use the local original pretrained archive acquired by METH-378, not the METH-380
int8 export. Reference the frozen inherited METH-379 per-tensor binding and
retain its hash, source commit, controller/protocol hashes.

Four tensors: `decoder.block.11.layer.2.mlp.experts.expert_{8,105}.{wi,wo}.weight`.
wi shape [3072,768], wo [768,3072], each 9,437,184 raw F32 bytes. The binding
maps all four to `pytorch_model-00003-of-00003.bin`. Verify exact shape, dtype,
contiguity, finiteness, raw size and original SHA-256 for every tensor. Source
location: main checkout's `results/native_expert_scaling/meth378_switch_base128_source/`.
Record current archive length/mtime and historical qualified archive hash.
Avoid a new full 10 GB archive scan: independently verify each admitted tensor
against the original binding instead. Explicitly do not claim current whole
archive identity has been reverified.

## Isolation, resource bounds and retention

Separate owned CPU scientific environment with torch 2.6.0+cpu, NumPy 2.1.3,
Python 3.12.10; no installation in the other environment. One CPU thread,
no gradients/GPU/native benchmark, restricted `weights_only=True`, private
memory mapping. Touch only ~38 MB selected tensor storage and small archive
metadata; do not materialize the entire model. Check active local scientific
processes before the read; this is lightweight source qualification, not an
exclusive native benchmark. Do not stop or interfere with another process.
300s internal deadline checked between tensors; preserve first failure/partials.
No full archive checksum or intensive scan while another branch uses resources.

Exclusive output: `results/progressive_ternary/PQT-SRC-001/original_001/`.
Save four non-pickle NPY arrays and an admission manifest containing original
raw tensor hash, output file hash/size and provenance. Recheck output identities
in a separate standard-library pass. Large arrays remain outside Git, small
raw manifest plus retention location committed in this dossier. Freeze source
and this protocol before accessing numerical tensor values. A successful
admission enables a later preregistered experiment; it establishes no quality,
generation, expert capacity, full-model compression or native speed result.
