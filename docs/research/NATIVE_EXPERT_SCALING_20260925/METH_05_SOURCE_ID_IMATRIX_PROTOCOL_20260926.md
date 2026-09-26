# METH-05: source-ID importance matrix and first GigaChat conversion

**Status:** decision criteria frozen before the full calibration and weight
conversion. The 1-chunk resource probes and 16-chunk Q4 coverage probe below
are apparatus observations, not quality measurements. This cell asks whether
the [METH-04 map](METH_04_GIGACHAT_LOWBITS_PROTOCOL_20260925.md) can become
real, quality-bearing weights using local resources. A converted GGUF remains
an intermediate artifact until `engine.c` runs it at the required rate.

## Bound inputs and reason for this cell

The source is the locally bound GigaChat BF16 GGUF, SHA-256
`fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`.
The prior source-binding audit found all 414 tensor payloads byte-identical
to the published BF16 GGUF; its differing metadata fields were classified as
nonsemantic. The source-converted file is still the explicit reference here.
The source tokenizer JSON has SHA-256
`b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe`.
The calibration split is the 48-document `strat01_gigachat_fresh_v2/calib.jsonl`,
SHA-256 `68d9327a823328b1104c843afe986efeefcdd88c911c02895d577f65bffd8f81`.
It is disjoint by item and source-content hash from its 96-document heldout.
It has 16 code, 16 prose and 16 technical/general documents, but no Cyrillic;
this establishes only a bounded domain pilot. Full-language capability
requires a new multilingual calibration/evaluation split.

The pinned llama.cpp `5b335f413` imatrix program tokenizes raw text with the
GGUF tokenizer, which differs from GigaChat's source text-to-ID mapping.
The [small source patch](../../../benchmarks/native_expert_scaling/meth05_source_ids_imatrix.patch)
allows an environment-bound file of decimal source token IDs. The
[preparer](../../../benchmarks/native_expert_scaling/prepare_meth05_imatrix_ids.py)
verifies source corpus/tokenizer hashes, each document's round trip and special
IDs, and alternates the three categories before flattening. Its 54,633-ID
exchange file has SHA-256
`3d611152b4a145f2219a34e7ad5bc2e8929a9a600321272e9e258b92924c250a`.
Its [manifest](meth05_calib_ids_manifest.json) records all document IDs and
the exact category order without copying the texts.
The tool still inserts BOS at each 512-token chunk boundary; documents can
cross those boundaries, a calibration approximation to record separately.

The first Q4 512-token probe loaded the source IDs, wrote all 254 required
tensor entries, and left 78 of 4,800 routed expert slices without any
activation. Sixteen Q4 chunks covered all 4,800 slices, but the minimum
count was only 2. Q4 calibration is an activation proxy, not the chosen
primary matrix. A BF16 1-chunk run cost about 4 minutes to initialize and
14.56 seconds for the pass, with approximately 18 GB resident observed.
These observations price the next local run; no T4 is scheduled.

The patch SHA-256 is
`22d705b7fa7f22417c6fb09c299527277e709713875c224b6cae96dbf8d77287`.
The patched `llama-imatrix.exe` SHA-256 is
`05b1d7e9f8256220977012ac55ae658f7c58f8967288d74cd96088fbf22e55fe`.
From the repository root, after applying the patch to the pinned llama.cpp
checkout and building `llama-imatrix`, the full collection command is:

```powershell
$env:SILICON_IMATRIX_SOURCE_IDS=(Resolve-Path results/native_expert_scaling/meth05_gigachat_calib_interleaved_ids.txt).Path
& C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\build-cpu\bin\llama-imatrix.exe `
  -m benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/GigaChat3.1-10B-A1.8B-source-bf16.gguf `
  -f results/native_expert_scaling/meth05_gigachat_calib_interleaved_ids.txt `
  -o results/native_expert_scaling/meth05_gigachat_bf16_interleaved_106chunks.gguf `
  --chunks 106 --ctx-size 512 --batch-size 128 --threads 6 --no-ppl --output-frequency 1000
```

## Frozen sequence and stops

1. Run **106 full 512-token chunks** of the interleaved source-ID file on the
   BF16 source, six CPU threads, batch 128, no perplexity/output-head imatrix.
   This uses 54,272 of 54,633 IDs; no heldout enters calibration. Stop the
   run if wall time exceeds **45 minutes** or process resident memory exceeds
   **50 GB**. Preserve any partial/failed log. Do not substitute the Q4 matrix
   silently if the BF16 run fails.
2. Apply the [matrix audit](../../../benchmarks/native_expert_scaling/audit_meth05_imatrix.py)
   to the full GGUF. Require all **254** IQ2_XS tensors present, finite
   values, exact matrix dimensions checked by the quantizer, and every routed
   expert slice reached at least **64** times. A lower count stops conversion;
   the calibration set or budget must be revised explicitly. The minimum is
   an exposure floor, not proof that 64 observations estimate a stable matrix.
3. Quantize the **BF16** GGUF with the exact METH-04 414-tensor map and this
   BF16 imatrix. Budget **45 minutes**, **50 GB** resident RAM and **4 GB** new
   GGUF output. No fallback, missing matrix or tensor-type mismatch is
   accepted. Re-read the actual GGUF header and active-byte ledger; require
   ≤540,000,000 addressed bytes/token before quality evaluation.

   ```powershell
   & C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\build-cpu\bin\llama-quantize.exe `
     --imatrix results/native_expert_scaling/meth05_gigachat_bf16_interleaved_106chunks.gguf `
     --tensor-type-file docs/research/NATIVE_EXPERT_SCALING_20260925/meth04_gigachat_tensor_types.txt `
     benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/GigaChat3.1-10B-A1.8B-source-bf16.gguf `
     results/native_expert_scaling/meth05_gigachat_bf16_imatrix_iq2.gguf IQ2_XS 6
   ```
4. Score the lexicographically first **three heldout IDs per category** (nine
   documents) on source BF16, published Q4 and converted candidate using
   the reused [source-ID GGUF scorer](../../../benchmarks/donor_adaptation/density/test_strat01_gguf_score.py),
   no truncation, BOS=1, document-level BPB. Bind new GGUF arms with its
   `--expected-model-sha256` option. This is an
   internal gross-failure screen. The
   [ID-only selector](../../../benchmarks/native_expert_scaling/prepare_meth05_pilot.py)
   froze this subset before candidate scoring; the
   [selection manifest](meth05_pilot9_selection.json) records its IDs. SHA-256
   `38298c84038b6f26a9c40f47347d5cf930c89ad2bc089e64057f5125c79bb2ac`.
   If each of the three category BPB deltas
   versus source BF16 exceeds **+0.20** and pooled delta exceeds **+0.20**, stop
   this low-bit map and report the failure; otherwise proceed to all 96
   documents. The [adjudicator](../../../benchmarks/native_expert_scaling/adjudicate_meth05_pilot.py)
   enforces the paired IDs, byte/token counts and stop rule. The
   nine-document screen cannot declare a quality pass.
5. For all 96 documents, use the existing paired stratified bootstrap
   (20,000 draws, seed `20260916`): require one-sided upper CI95 of
   candidate−BF16 **≤+0.02 BPB**. Also run the frozen GigaChat PIQA and
   document-rollout gates: PIQA correct ≥ceil(0.98×BF16 correct), and fewer
   than three candidate-only catastrophic rollouts. The source/published Q4
   arms are controls, not replacements for the candidate. The corpus was
   previously used to assess Q4, so this is internal fidelity evidence;
   later cross-family selection needs a fresh external holdout.

The result record must include commands, CPU/RAM and elapsed time, binary and
artifact hashes, matrix coverage, actual tensor types/bytes, per-document
scores, confidence interval, rollout/task outcomes, and explicit pass/fail.
Even a quality pass does not license a native speed claim: GigaChat operators,
source-tokenizer behavior and weight format must work in `engine.c`, then
quality and ≥50 accepted tok/s must pass on that same native artifact.
