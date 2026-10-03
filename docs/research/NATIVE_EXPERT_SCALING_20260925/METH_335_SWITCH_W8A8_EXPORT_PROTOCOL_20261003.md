# METH-335: complete original bank row-I8 export

Prospective after334 prescribed arithmetic qualification PASS. Resolve whether
an actual compact ALL-bank representation can be built without losing source
provenance/tensor names or collapsing code-and-scale parameter tuples. This is
a prerequisite for native integer arithmetic/cost, not pretrained quality.
Reuse327 all finite actual tensors/tied alias hashes and334 qualified source
architecture. Preserve all original256 labels in each of12 source banks.

Freeze code/protocol before observations. No native benchmark/model forward/
new sources/training/GPU/download. MAIN20min AFTER imports, checked16GiB RSS,
24GiB target file cap, require>=32GiB free disk. Existing projectTorch2.6 only;
no reliance on unsupported installed Switch5.13.1. Source never overwritten,
fresh complete original archive+sidefile hashes before conversion. On failure
retain raw stage/partial directory/payload; fresh paths, no automatic overwrite.

All6392 original source F32 contiguous finite tensors verified byte-SHA327
while streaming one Torch weights_only/mmap shard at a time. Row I8 for ALL
2D matrices except router, relative-bias and lookup embeddings. Scales:
F32(absmax/127), original F32 division then nearest-even rounding and clipping
[-127,127]. Zero row scale1/codes0; nonzero underflow/invalid scale FAIL.
Controls/norm/relative bias/router original F32. Three lookup embedding aliases
physically deduplicated only after exact original byte/shape proof. Head is
separate I8 view from the same original tied F32 source; count its extra bytes.
No learned extra capacity/fitting. All names retained in target manifest.

Before source work: fixed hand-coded zero/tie/extreme controls versus independent
scalar F64 division rounded F32 then Python nearest-even; exact codes required.
I64 independent integer extreme/cancellation at4096x127:66,064,384/0. This only
qualifies mathematical integer bound, not native AVX2 implementation.

Target layout: one actual weights.bin, each array starts64-byte aligned,
I8 row-major codes followed by F32 row scales; original F32 arrays. Read back
ALL6392 code/control and scale payloads and require recorded exact SHA/length,
valid scales and no-128 codes. Every source shape/name preserved. All12 banks
require labels0..255 and256 distinct WI/WO code-and-scale tuple hashes excluding
bank/label. This is parameter distinction, not effective/useful functions.

Binary manifest SWI8A001: same13 config u32+F32 epsilon+filecount/tensorcount
as328; length-prefixed absolute UTF8 payload path. Each length-prefixed name
then5u32(file,dimensions,rows,cols,encoding0F32/1I8),3u64(code/control offset,
scale offset,elements). Full target payload SHA and manifest SHA; raw metadata
pins source/model/revision/original config/every source and target tensor SHA,
target bank tuples and resource/command provenance. Target approximate:
record physical bytes, not an assumed compression factor or useful capacity.

PASS allows separately frozen native integer/scaling correctness and actual
full CPU cost preflight. FAIL retains and closes unchanged export. No heldout
quality, source-model forward, native compute/routing rate, LUT/useful RAM-n or
accepted>=50 claim. Original329/330/332 numerical failures unchanged; ORIGINAL
unmodified donor remains PRIMARY for NEW future untouched quality cohort.

Command:
```
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth335_switch_w8a8_export.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth335_switch_w8a8_export_result.json
```
