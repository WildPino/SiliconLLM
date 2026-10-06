# PQT-SRC-003: exact selective Switch byte reader qualified

6 October 2026. **SOURCE PREPARATION PASS; RESEARCH INCOMPLETE.**
[Protocol](PQT_SRC_003_PROTOCOL.md), [frozen binding](PQT_SRC_003_BINDING.json),
[original retention](PQT_SRC_003_ORIGINAL_001_RETENTION.json),
[exact Git proof](PQT_SRC_003_GIT_VERIFICATION.json).

## Finding

The standard-library reader admits all15 prospectively selected original F32
tensors, with every raw hash matching the inherited full tensor binding. It
parses and compares all3,320 actual tensor headers across the three original
archives without loading/mapping the complete checkpoint. The selected tensors
contain118,358,016 original bytes; their15 NPY files total118,359,936 bytes.
All values are finite. Previous NumPy-produced anchors for experts8/105 have
exactly matching raw bytes; NPY header spelling can differ between writers.

The new selection includes decoder11 expert pairs0,8,32,64,96,105 and three
encoder/router/decoder core tensors. Experts0,32,64,96 add original byte
coverage. None has been evaluated as a function or selected on measured quality.
No model forward, optimizer update, GPU job or native timing was performed.

## Qualification and identities

Bindings prepared from consumed METH-325/METH-379 records before original reads:
2,435,020 bytes, SHA256
`efb5e3889f93e8c1880095a3c7d94ff600803fea8a856297425ada6e6ae36584`.
Original checkpoint revision86c815ec05361a33a8b49fc717277da9c0a4e711.
Original reader source/qualification freeze810c599335a13cb7dd09c06ce78ab95adb36a346.

Controls001 and002 both pass, with complete fixtures and historical source
bytes retained, including the earlier reader version. Controls002 extends001
before original use:38 explicit malformed stops and two positive aliased-tensor
NPY comparisons against independently loaded weights_only Torch tensors and
NumPy. Stops cover unsafe globals, trailing pickle, storage IDs/dtype/device,
offset/stride/address/rank errors, duplicate/path/symlink/ancillary ZIP records,
compression, header/pickle/archive/storage lengths and hashes, byte order,
noncontiguous/offset views, nonfinite payloads, CRC corruption, and byte/time caps.
Payload identity remains mandatory independently of ZIP CRC.

Original run001 passes all gates. Per-shard tensor counts1084/1121/1115 match
the frozen namespace; complete pinned data.pkl hashes and all layout/storage
specifications match. Current stat identities remain unchanged through reads.
Whole archive SHA256 values in the report are historical qualified identities;
the complete29,859,401,788 archive bytes were **not rehashed in this operation**.
Selected payload identities were reverified, and only selected records were
interpreted as numerical weights.

Separate streaming retention checks all15 NPY layouts, lengths, raw source
hashes and complete output hashes again, without importing the byte reader.
Raw original admission report13,150 bytes, SHA256
`e7ee5d706ae3ecee528d5055146bafa226b569d6112b6217415629622d3aa34f`.
Git proof at e6316756fc326a1d358f98d9873f3b84d15ee9ce verifies64 distinct
source/binding/retention/raw-evidence files totaling2,619,516 bytes against actual
working/raw origins. Large NPYs stay in the owned results directory, outside Git;
their complete retained manifest makes loss detectable.

## Measured resources

Fresh local process admission passed before both control runs and original
reads. No simultaneous owned preparation process was launched.

| Operation | Process elapsed seconds | Peak process RSS bytes |
| --- | ---: | ---: |
| Controls001,33 malformed stops | 5.828 | 184,811,520 |
| Controls002,38 malformed stops | 2.797 | 185,380,864 |
| Standard-library original admission001 | 6.875 | 34,996,224 |

Original archive metadata reads713,257 bytes/21 calls; payload-phase reads
118,358,708 bytes/145 calls, including692 bytes of record overhead. Largest
metadata read163,373 bytes; largest payload read1,048,576 bytes. Four old anchor
NPY reads add37,749,248 bytes, separately recorded. Output writes118,359,936 bytes
plus the small report. Independent retention rechecks the selected output once.

Retention process elapsed controls0010.188 seconds/controls0020.203 seconds/
original0010.578 seconds. These reports instrument their own operation duration;
shell startup, authoring, initial metadata-binding preparation and Git commands
are outside those timers. No CPU-native latency/inference-speed claim follows
from source I/O durations. This SRC operation had no failed attempt or retry;
both historical control rounds remain charged and retained.

## Scope and exact resumption

This removes a bounded-source-loading obstacle. It admits neither a full float
Switch execution path nor original natural expert contexts. Existing SRC-002
I8-core contexts remain consumed and insufficient for that claim. No change to
any PQT quality decision or ternary promotion.

Keep completed original001/controls001/002 exclusive; do not rerun or overwrite.
Owned artifacts: results/progressive_ternary/PQT-SRC-003/original_001 and
controls_001/controls_002. Raw reports/fixtures/historical executable sources:
pqt_src003_original_evidence and pqt_src003_controls_evidence. Source apparatus:
benchmarks/progressive_ternary/switch_zip_source.py; scripts/progressive_ternary/
prepare_src003_binding.py, qualify_src003_reader.py, admit_src003_source.py,
retain_src003.py, verify_src003_git.py. Operational reader uses the owned standard
library Python3.12.10; fixture comparisons use the owned scientific environment
Torch2.6.0+cpu/NumPy2.1.3, one thread.

Next qualify a tiny synthetic full encoder-decoder with lazy expert loading,
tied weights and an independently checked SDPA attention path before any
original source context capture. Preserve the original router/capacity behavior
and eval arithmetic; freeze source/runtime/operator controls and then the
natural input/split/selection protocol before new original-model functions.
Any later original streaming workload needs newly declared read/time/VRAM
bounds: the118 MiB source screen allowance does not authorize a30 GB scan or
multiple full-model passes. Private T4 dispatch still requires fresh owner,
quota and all-addressable-kernel admission. Delegate actual long T4 monitoring
and wait dormant until meaningful failure or terminal wake. Native CPU timing
continues to require the unanswered owner availability window.
