# PQT-SRC-003: selective original Switch source admission

Prospective source-byte qualification, 6 October 2026. This is preparation for
original float-model context capture, not a fitting experiment or quality test.
The original Switch model is an encoder-decoder trained with masked-span inputs;
future natural-context acquisition must preserve that full path and conditioning
([model card](https://huggingface.co/google/switch-base-128)). Existing inherited
I8-core contexts do not qualify as original float-model natural states.

## Question and frozen selection

Can a bounded standard-library reader reconstruct exact original F32 tensor
bytes from the three PyTorch ZIP archives without loading or mapping the whole
checkpoint? Source: google/switch-base-128 revision
`86c815ec05361a33a8b49fc717277da9c0a4e711`. Inherited METH-325 headers and METH-379
tensor hashes are consumed source evidence. Freeze their current file hashes,
all 3,320 header/tensor specifications and three archive identities in an owned
binding before original numerical payload reads.

Select these 15 tensors in the stated order, independently of tensor values:
encoder block0 attention q, encoder block11 router classifier, decoder block0
attention q; then decoder block11 experts 0,8,32,64,96,105, each wi then wo.
Exactly 118,358,016 original bytes. Experts8/105 are previously admitted anchors;
the other four extend source coverage only. No function, routing or quality
claim follows from a tensor hash.

## Gates, bounds and failure policy

Restricted pickle globals: OrderedDict, torch FloatStorage and the two tensor
rebuild functions mapped to structural metadata; no library import or arbitrary
pickle code execution. Accept F32 CPU storage only. Check dimensions, strides,
offset bounds, storage consistency, exact frozen names/layouts, uncompressed
ZIP records, unique safe names, non-symlink files and declared storage lengths.
Check complete pinned data.pkl hashes and all actual 3,320 metadata entries.
Accept selected tensors only as full contiguous, zero-offset storage records,
with exact original per-tensor SHA256 and finite F32 values. Write standard NPY
files with no pickle, stream at most 1 MiB at a time, and record raw/output hashes.
Do not interpret a current size/mtime check as a new whole-archive hash.

Before original reads, run tiny actual torch.save fixtures and independent
weights_only Torch/NumPy comparisons; qualify aliases, F32 byte order and NPY
output. Negative controls must reject unsafe globals, invalid persistent IDs,
offsets/strides/dtypes, changed pickle/payload hashes, malformed ZIP namespace,
compression, length/selection/layout mismatches and resource overruns.
Freeze executable source and qualification evidence in Git before original use.

One local sequential operation, one thread where Torch is used; fresh process
inventory must admit it. No GPU or native timing. Per read <=2 MiB metadata,
<=1 MiB payload; combined archive metadata reads <=8 MiB, combined payload-phase
archive reads <=128 MiB; each selected tensor <=128 MiB. Process peak RSS <=512
MiB for the standard-library original reader, elapsed <=180 seconds. Metadata
parse and each stream read enforce the elapsed stop. Output directory exclusive;
retain complete numbered first failures and partial outputs, never overwrite.
Source stat identity must remain unchanged through the read. No automatic retry
or claim of qualification after a failure. No full-checkpoint scan or mapping.

## Decision and next boundary

Pass only if every synthetic positive/negative control and every original
selection/header/hash/output gate passes, raw reports are retained, and source,
binding and report bytes have exact Git proofs. All reads and overhead count in
costs. This admits a byte reader for later separately qualified lazy full-float
execution. It does not admit natural contexts, expert function fidelity, fitting,
native speed or model quality. Original-context acquisition requires a separately
frozen input/split protocol and operator equivalence, including attention,
router capacity, encoder-decoder conditioning and tied embeddings.
