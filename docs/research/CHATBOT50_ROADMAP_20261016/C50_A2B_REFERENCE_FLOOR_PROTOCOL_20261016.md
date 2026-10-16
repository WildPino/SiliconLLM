# C50-A2/A2b protocol: reference artifacts and the llama.cpp noise floor

16 October 2026. Cells A2 (second half) and A2b of the [CHATBOT50 roadmap](ROADMAP.md)
§5. Pushed before the imatrix, the quantization and every A2b run (programme rule 5).

## Question

How far does pinned llama.cpp disagree **with itself** on one Ling-mini-2.0 Q4_K_M
artifact when only the build (generic vs AVX2), the thread count (6 vs 1) or the
batch path (512-token micro-batches vs one token at a time) changes? That
disagreement is the floor from which the A3 engine tolerance is computed (rule 2),
fixed before any engine output exists.

## Fixed inputs

- **Source checkpoint:** `inclusionAI/Ling-mini-2.0` @ `a810f64`, all 14 files
  hash-verified (`benchmarks/chatbot50/results/c50_a2_ling_mini_source_manifest.json`).
- **llama.cpp:** `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`, WinLibs GCC, Release,
  static libraries, OpenMP. Two builds: `build-avx2` (AVX, AVX2, FMA, F16C on,
  `GGML_NATIVE` off) and `build-generic` (all x86 SIMD options off).
- **Holdout:** `benchmarks/chatbot50/corpus/c50_holdout_v1` built by
  `benchmarks/chatbot50/c50_holdout.py`. The manifest is tracked; the text is not.
  - Roles: FIT 48+48+48, DEV 32+32+32, RESERVED 32+32+32 (code, technical
    markdown, PG19 prose). Each document has one 8192-byte span.
  - Roles are assigned from the document content hash.
  - The 230 documents used by any earlier project holdout are excluded.
  - Two spans that NFC normalisation would change are skipped in rank order. The
    source tokenizer normalises to NFC, so their IDs would not decode back to the
    span.
  - Token counts from the source tokenizer: FIT 327,578; DEV 208,736; RESERVED 219,666.
- **"Fresh" means this:** no project decision has used these documents. It does
  **not** mean they postdate the model. PG19, The Stack and public documentation
  markdown are very probably inside Ling-mini's pretraining data. Within one
  artifact, comparisons between runtimes do not depend on this. A4's
  quantization dBPB on memorised text may differ from fresh text, and A4 must
  carry that limitation.

## A2 conversion steps (no gate; recorded)

1. **BF16 GGUF:** `convert_hf_to_gguf.py --outtype bf16` from the pin.
2. **Vocabulary identity:** `c50_a2_gguf_check.py` must report zero ID→token
   mismatches between every GGUF and `tokenizer.json` over all source IDs. A
   mismatch stops A2b, because the runtimes are fed source IDs.
3. **imatrix:**
   - Input: all 144 FIT spans in `fit.jsonl` order, joined by a blank line.
   - Model: BF16 GGUF.
   - Tool: `llama-imatrix` (AVX2 build, 6 threads, `-c 512`, all chunks, default
     options otherwise).
   - Tokenization: `llama-imatrix` reads text, so llama.cpp's own tokenizer is
     used. Its agreement with the source IDs on the FIT spans is measured and
     reported. An imatrix is a calibration statistic, so a small tokenization
     difference changes the activation sample, not what the result means.
   - Stop rule: if the projected time exceeds one hour, the run is not started.
     The owner is asked instead (rule 7).
4. **Q4_K_M:** `llama-quantize --imatrix <FIT imatrix> <bf16> Q4_K_M`, with
   default options. The tensor-type mix and the per-token weight bytes of the
   artifact are recorded by `c50_a2_gguf_check.py`.

## A2b positions

- **Subset:** the first four items of each category of `fit.jsonl`, in file
  order (`c50_make_ids.py --role fit --per-category 4`). That is 12 documents and
  26,363 scored positions; the input file SHA-256 is
  `cf76819b8b5ee1636feea8badd7ae4b52d08b1bcc5dffcd87d8704722eeaa460`.
- **Scoring:**
  - Every position p ≥ 1 of each document is predicted from `ids[0..p-1]`; the
    first token is context only.
  - No special token is added.
  - Context is cleared between documents.
  - `n_ctx` 8192, KV F16, flash attention disabled.
- **Scorer:** `benchmarks/chatbot50/c50_llama_score.cpp`, linked against the
  pinned static libraries (`c50_build_scorer.sh`). It prints and asserts one
  `CONFIG` line and dumps the full log-softmax row of every position.

## Arms

| Arm | Build | Threads | Micro-batch | Role |
| --- | --- | --- | --- | --- |
| R0 | AVX2 | 6 | 512 | reference |
| R1 | AVX2 | 1 | 512 | thread floor |
| R2 | generic | 6 | 512 | build floor |
| R3 | AVX2 | 6 | 1 | batch-path floor (the decode path an engine runs) |

**Statistics.** `c50_compare.py` computes each pair (R0, Rk) on the same
positions. Every statistic carries a document-level bootstrap 95% interval
(10,000 draws, seed 20261016).

- top-1 disagreement;
- mean KL(R0 ‖ Rk) in nats, from the full rows;
- dBPB (bits over the bytes of the scored tokens);
- bit-identity of the target log-probabilities.

**Floor, defined now:**

- `floor_top1` = the largest top-1 disagreement point estimate over the three pairs;
- `floor_KL` = the largest mean KL point estimate over the three pairs.

The A3 gate then reads as the roadmap fixes it:

- top-1 disagreement ≤ max(2 × `floor_top1`, 0.2%);
- mean KL ≤ max(2 × `floor_KL`, 1e-4 nats);
- |dBPB| upper CI ≤ 0.002.

The A3 protocol names the llama.cpp arm the engine is compared with. The default
is R3, the same computation shape as engine decode. It is set before the engine is
measured.

**Not timed.** These runs measure numerics only, so a contended box does not void
them. Stop rule: if the first document projects an arm beyond one hour, that arm
is not continued and the owner is asked.

## A2b rate (external reference rate; no gate)

`llama-bench` (AVX2 build) on the Q4_K_M artifact:

- `-t 6 -p 0 -n 128 -d 512,2048,4096 -r 3 -fa off`;
- five fresh processes;
- the clean-box rules of C50-A1: a 45 s pre-flight, the foreign-occupancy bar
  4.39% per process, VOID processes kept and replaced at most five times.

Statistic: the median over processes of the reported generation rate at each
depth, with min..max. It runs in the same owner-provided clean-box session as
C50-A1.

## Registered predictions (scored, not gates; ROADMAP §5)

- The floor is above zero: a top-1 disagreement of order 0.1-1% for at least one
  pair.
- llama.cpp decodes Ling-mini Q4_K_M at ≤ 30 tok/s at depth 2048.

## What this cell cannot say

- The floor is llama.cpp's own sensitivity to these three changes on one artifact.
  It is not the quantization damage, which is measured in A4.
- Fidelity on FIT positions says nothing about DEV or RESERVED quality.
