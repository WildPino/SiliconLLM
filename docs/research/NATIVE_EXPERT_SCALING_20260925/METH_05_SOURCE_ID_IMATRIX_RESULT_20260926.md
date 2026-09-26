# METH-05 result: BF16 calibration exposes rare experts too sparsely

**Verdict: HOLD_CALIBRATION_EXPOSURE.** The preregistered 106-chunk BF16
importance matrix has every required tensor and no non-finite values, but
misses the minimum 64 observations per routed expert slice. No low-bit GGUF
was written; there is no new quality or native speed result. This is a
calibration-coverage failure, not a donor-quality failure or a refutation of
the METH-04 type map. The [frozen protocol](METH_05_SOURCE_ID_IMATRIX_PROTOCOL_20260926.md)
sets the stop and later quality gates.

## Inputs and apparatus

- Source-bound BF16 GGUF: SHA-256
  `fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`.
  Q4 control: `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Non-held-out 48-document calibration: SHA-256
  `68d9327a823328b1104c843afe986efeefcdd88c911c02895d577f65bffd8f81`.
  The source tokenizer has SHA-256
  `b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe`.
  The [ID manifest](meth05_calib_ids_manifest.json) records the category
  interleaving. The local 54,633-ID file hashes to
  `3d611152b4a145f2219a34e7ad5bc2e8929a9a600321272e9e258b92924c250a`.
- Pinned llama.cpp revision `5b335f413`, source-ID patch SHA-256
  `22d705b7fa7f22417c6fb09c299527277e709713875c224b6cae96dbf8d77287`,
  patched `llama-imatrix.exe` SHA-256
  `05b1d7e9f8256220977012ac55ae658f7c58f8967288d74cd96088fbf22e55fe`.
  `--ctx-size 512 --batch-size 128 --threads 6 --no-ppl`; CPU Ryzen 5 3600X;
  no T4/GPU. The patch reads validated source-tokenizer IDs rather than the
  known divergent GGUF text tokenization.

## Coverage and cost

| Matrix arm | Full 512-token chunks | Required tensor entries | Minimum routed-expert count | Predeclared ≥64 gate |
|---|---:|---:|---:|---|
| Q4 resource probe | 1 | 254/254 | 0; 78 of 4,800 expert slices untouched | fail |
| Q4 category-interleaved | 16 | 254/254 | 2 | fail |
| BF16 resource probe | 1 | 254/254 | 0 | fail |
| BF16 category-interleaved | 16 | 254/254 | 3 | fail |
| **BF16 category-interleaved full available corpus** | **106** | **254/254** | **16** | **fail** |

The 106-chunk run processed 54,272 IDs (361 trailing IDs unused), returned
exit 0 and wrote a **31,645,856-byte** matrix, SHA-256
`ef5de87fc4fa0d1382ed2988e59e9472fb1d7fb2c9bed0e8edcced6ff9bd4f9c`.
It ran 09:22:52–09:43:05 local time, about **20 min 13 s**, below the
45-minute stop. The observed resident working set reached approximately
**21.25 GB**, below the 50 GB ceiling. These are local process observations,
not conversion/training cost or C decoding rate. See the
[complete normalized log](meth05_bf16_106chunks.log), SHA-256
`d12c73b0e39a7859842188b783191a401747102c3917e2e8bbfdf5176dc7a372`,
and the [machine audit](meth05_bf16_106chunks_audit.json), SHA-256
`e8ced042cc7d2c76b657210e5387ad2bf96263be12db368ab36a9e8e55d4b068`.

Every one of the 4,800 routed tensor/expert instances was touched and all
254 required matrix entries are finite. Only **two distinct expert choices**
miss the exposure floor, each repeated across its gate/up/down tensors:
layer 19 expert 29 has **16** observations; layer 20 expert 51 has **39**.
The next-smallest expert count is **199**. The minimum is therefore governed
by rare routes, not wholesale missing tensor collection. The 64-count floor
was fixed before observing the 106-chunk matrix; lowering it after this
result would make the proposed low-bit quality test less interpretable.

## Decision and next action

Do not quantize from this matrix under METH-05. Repeating the same 54,633 IDs
would raise counters without adding independent activation contexts. A new
non-held-out calibration extension should add genuinely different documents
and language coverage, including Cyrillic; the local Kubernetes website has
Russian and Ukrainian Markdown files, while the current heldout has no paths
under those language directories. Freeze file/content hashes and exclusion
against heldout, then run a bounded source-ID route/exposure probe. Require
the same ≥64 count floor before quantizing, and retain the frozen BF16/Q4/
candidate BPB, PIQA and rollout gates. General language quality remains
unproven even if calibration coverage improves. A Q4-derived matrix was
measured only as a resource proxy; the chosen primary path remains BF16.
