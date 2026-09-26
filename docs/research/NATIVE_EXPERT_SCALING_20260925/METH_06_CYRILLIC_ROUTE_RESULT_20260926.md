# METH-06 result: route coverage recovered; IQ2 quality gate fails

**Verdict: STOP_GROSS_QUALITY_FAILURE for the frozen METH-04 type map.**
Distinct Russian/Ukrainian calibration raised every routed expert above the
predeclared 64-activation floor. The source-BF16 importance matrix then
produced a 3.208 GB GGUF with all 414 intended tensor types and
534,024,800 addressed bytes per token. On the frozen nine-document quality
screen, however, the candidate lost **+0.239465 BPB** versus source BF16.
Each category separately lost more than the predeclared +0.20 BPB gross
failure threshold. The [protocol](METH_06_CYRILLIC_ROUTE_PROTOCOL_20260926.md)
and [METH-05 quality stop](METH_05_SOURCE_ID_IMATRIX_PROTOCOL_20260926.md)
therefore end this low-bit map before the 96-document, PIQA or rollout gates.
This is not a native C speed or trained large-expert result.

## Inputs and calibration

The bound source GigaChat BF16 GGUF SHA-256 is
`fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`.
The source tokenizer JSON SHA-256 is
`b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe`.
The [new calibration manifest](meth06_cyrillic_calib_manifest.json) binds
90 distinct Kubernetes Markdown files (74 Russian, 16 Ukrainian), with
source-path and full-content-hash exclusion from both old calibration and
heldout. Its 64,322 source-tokenizer IDs have SHA-256
`7f5333d1d52a5c94ad0026ff0f38cc022a3aeeecf1b884085b56abe8c795c524`.
The original English/code/technical source-ID file has SHA-256
`3d611152b4a145f2219a34e7ad5bc2e8929a9a600321272e9e258b92924c250a`.
No calibration document came from the 96-document heldout; the
[selection manifest](meth05_pilot9_selection.json) froze the quality pilot
before candidate scoring.

On local Ryzen 5 3600X, six CPU threads, the patched pinned llama.cpp
`5b335f413` imatrix processed 125 full new 512-token chunks (64,000 IDs),
roughly 24 minutes wall time and about 21.3 GB resident working set at the
observed peak, under the frozen 40-minute/50-GB ceilings. The
[new-matrix audit](meth06_cyrillic_bf16_125chunks_audit.json) records all
254 required entries finite, all 4,800 routed tensor/expert instances
touched, and a minimum count of 31 on that new matrix alone. Its
31,645,824-byte GGUF SHA-256 is
`5c529f9009f0f241dfc50d0a1f598b0e183c137693a951f04963dc729a344d08`.
The two previously rare expert choices gained 78 (layer 19, expert 29) and
102 (layer 20, expert 51) activations. The [collection log](meth06_cyrillic_bf16_125chunks.log)
and [prior 106-chunk audit](meth05_bf16_106chunks_audit.json) retain the
underlying run evidence.

Only the **106 original + 125 new** matrices were merged. The 32-chunk
Cyrillic pilot was a prefix of the new run and was not counted again. The
[merge log](meth06_merged_231chunks.log) shows both input files loaded;
the [merged audit](meth06_merged_231chunks_audit.json) verifies exactly
231 chunks, all 254 required entries finite, and **94 minimum routed expert
activations**, above the frozen 64 floor. The merged 31,645,920-byte GGUF
SHA-256 is
`7a881fc7bb821388bbb019bca3cc2047272e0a1b5985020427a0302b7cfa8cbc`.
Layer 19/expert 29 has 16+78=94 and layer 20/expert 51 has 39+102=141.
This is a calibration-exposure pass, not a claim that 64 observations are
statistically sufficient or that multilingual quality has been measured.

## Conversion and actual byte gate

Using the merged BF16 matrix and the frozen
[414-tensor type map](meth04_gigachat_tensor_types.txt), `llama-quantize`
exited 0 after 1,353.407 seconds (22 min 33 s) on six CPU threads. The
[complete conversion log](meth06_gigachat_bf16_imatrix_iq2_quantize.log)
reports loading 308 matrix entries computed on 231 chunks and no fallback
or missing-matrix warning. The candidate GGUF is 3,208,362,784 bytes,
SHA-256
`fb19f9eca6825b61b95f4289136e3c57ba76e41454a96d5565c35a160239ebce`.
The [independent header audit](meth06_gigachat_bf16_imatrix_iq2_verify.json)
found all 414 planned types, no missing/extra tensors, exactly
3,202,277,120 stored tensor bytes and **534,024,800 active bytes/token**.
It passes the 540,000,000-byte active ceiling with only 5,975,200 bytes
of headroom. This is addressed-payload arithmetic, not measured DRAM
traffic or `engine.c` throughput. The large GGUF and matrices remain local
under `results/native_expert_scaling/`; their hashes and compact audits
are retained here.

The essential local commands, from the repository root with the pinned
binary and source BF16 path above, were:

```powershell
$env:SILICON_IMATRIX_SOURCE_IDS=(Resolve-Path results/native_expert_scaling/meth06_cyrillic_source_ids.txt).Path
& C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\build-cpu\bin\llama-imatrix.exe `
  -m benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/GigaChat3.1-10B-A1.8B-source-bf16.gguf `
  -f results/native_expert_scaling/meth06_cyrillic_source_ids.txt `
  -o results/native_expert_scaling/meth06_cyrillic_bf16_125chunks.gguf `
  --chunks 125 --ctx-size 512 --batch-size 128 --threads 6 --no-ppl --output-frequency 1000
Remove-Item Env:SILICON_IMATRIX_SOURCE_IDS
& C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\build-cpu\bin\llama-imatrix.exe `
  -m benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/GigaChat3.1-10B-A1.8B-source-bf16.gguf `
  --in-file results/native_expert_scaling/meth05_gigachat_bf16_interleaved_106chunks.gguf `
  --in-file results/native_expert_scaling/meth06_cyrillic_bf16_125chunks.gguf `
  -o results/native_expert_scaling/meth06_merged_231chunks.gguf --ctx-size 512 --batch-size 128
& C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\build-cpu\bin\llama-quantize.exe `
  --imatrix results/native_expert_scaling/meth06_merged_231chunks.gguf `
  --tensor-type-file docs/research/NATIVE_EXPERT_SCALING_20260925/meth04_gigachat_tensor_types.txt `
  benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/GigaChat3.1-10B-A1.8B-source-bf16.gguf `
  results/native_expert_scaling/meth06_gigachat_bf16_imatrix_iq2.gguf IQ2_XS 6
```

The merge tool emitted a deprecation warning about repeated `--in-file`,
while its log shows both files loaded and the independent header audit
confirms 231 chunks; the warning did not remove either input.

## Frozen donor-relative quality screen

The source-ID scorer used the same nine heldout documents, 11,091 payload
tokens and 36,855 UTF-8 bytes for each arm, with BOS=1 and KV reset per
document. The locally source-bound BF16 and IQ2 arms were each bound to their
GGUF SHA-256 before scoring. The [BF16 scores](meth06_source_bf16_pilot9_scores.jsonl),
[candidate scores](meth06_iq2_pilot9_scores.jsonl), and previously scored
[Q4 control](meth05_prior_q4_pilot9_scores.jsonl) are retained with the
[machine adjudication](meth06_iq2_pilot9_adjudication.json). The bound
source BF16 score matches the earlier BF16 reference exactly.

| Group | BF16 BPB | Q4 BPB | IQ2 BPB | IQ2−BF16 |
|---|---:|---:|---:|---:|
| Code (3) | 0.363524 | 0.376973 | 0.607039 | **+0.243515** |
| Prose (3) | 0.950031 | 0.961359 | 1.201196 | **+0.251165** |
| Technical/general (3) | 0.690800 | 0.703309 | 0.914516 | **+0.223715** |
| **All (9)** | **0.668119** | **0.680547** | **0.907584** | **+0.239465** |

The preregistered gross-failure rule stops when the pooled difference and
all three category differences exceed +0.20 BPB. All four do. The Q4
control differs from BF16 by only +0.012428 BPB on the same pilot; the
candidate's larger loss is not a scoring-input mismatch. This pilot can
reject this map but cannot certify another map or general model quality.
The 96-document CI, PIQA, greedy rollout and C speed gates were not run
for this failed candidate.

## Consequence for the large-expert objective

METH-06 resolves the **route-exposure** obstacle for this calibration
corpus, while showing that the chosen aggressive low-bit representation
does not preserve donor quality. More independent expert parameters can
fit in RAM as E grows, but that capacity does not reduce the active bytes
or rescue a quality-invalid precision map. A next donor-transfer design
needs a new, prepriced quality/cost mechanism with a frozen paired quality
gate; merely collecting more imatrix activations for this same IQ2 map is
not supported by this result. Separately, the CPU LUT/router cost and
learned quality at E≫128 remain unproven. No result here changes the
requirement for ≥50 accepted batch-1 tok/s and donor-relative quality on
one native `engine.c` artifact across families and scales.
