# METH-06: bounded Cyrillic calibration route probe

**Status:** frozen before evaluating new-source expert counts. METH-05's
106-chunk BF16 matrix was stopped by the preregistered ≥64-exposure floor:
layer 19 expert 29 had 16 observations and layer 20 expert 51 had 39.
The question here is whether distinct, non-held-out Cyrillic calibration
content can supply those missing routes at a justifiable local cost. This
does not observe donor-relative quality or change the low-bit type map.

## Source binding and selection

The [preparer](../../../benchmarks/native_expert_scaling/prepare_meth06_cyrillic.py)
examines local Kubernetes website Markdown under `content/ru` and
`content/uk`. It selects every file of at least 4096 bytes that has a
nonoverlapping 4095-byte UTF-8 window with at least 100 Cyrillic characters.
For each file it takes the window with most Cyrillic characters, breaking
ties by earliest byte offset. It reads the original 48-document calibration
and 96-document heldout only to exclude matching source paths or complete
source-content SHA-256 values. It verifies the source tokenizer hash,
round-trip and special-token exclusion, then interleaves RU/UK while both
remain. The [selection manifest](meth06_cyrillic_calib_manifest.json), SHA-256
`d6d41370896c624508b63aa16aaf07fd37306933c577edae537879660ecf9593`,
records 90 distinct source files (74 RU, 16 UK), 368,502 UTF-8 span bytes,
132,006 Cyrillic characters and 64,322 source-tokenizer IDs. The local
corpus SHA-256 is
`5af7e43f81d4c8d408e99a3c9de3e78ed2b3b2f7bbb91cd36944a0837275c735`;
the ID file SHA-256 is
`7f5333d1d52a5c94ad0026ff0f38cc022a3aeeecf1b884085b56abe8c795c524`.
The complete source file and content hashes are in the manifest. No candidate
quality scores were read to select these windows.

## Pilot, decision and resource stop

Run the first **32 full 512-token chunks** of the RU/UK ID file through the
patched BF16 `llama-imatrix`, six CPU threads, batch 128 and no perplexity.
The same source BF16/hash, patched binary/hash and 254-tensor map from METH-05
apply. The expected CPU cost is about 4 minutes of model initialization plus
8 minutes of decoding; stop at **20 minutes** or **50 GB** resident memory.
Keep the output/log and audit the two rare routed choices specifically:

| Routed choice | Prior count | Additional count needed for ≥64 | Pilot count needed to justify full new corpus |
|---|---:|---:|---:|
| layer 19, expert 29 | 16 | 48 | ≥12 |
| layer 20, expert 51 | 39 | 25 | ≥7 |

The pilot threshold extrapolates the first 32 of about 125 new chunks at
roughly constant route frequency. It is a cost screen, not an assumption
that the remaining documents have identical routing. If either count is
below its threshold, do not run the full new corpus under this cell; seek
broader non-held-out sources or a measured targeted-routing calibration
design. If both pass, freeze the full-run cost and process all new chunks,
merge with the METH-05 BF16 matrix, then require actual ≥64 observations for
every routed expert slice before quantization. Never multiply counts by
replaying the same IDs. Full quality and native speed gates remain unchanged.

**Full-run budget addendum, frozen after pilot counts and before full run:**
The pilot measured **30** additional layer-19/expert-29 and **38** additional
layer-20/expert-51 activations and met both cost-screen thresholds. Its
[matrix audit](meth06_cyrillic_bf16_32chunks_audit.json) and
[normalized log](meth06_cyrillic_bf16_32chunks.log) are retained; the matrix
SHA-256 is `cb751980db286bd5b0331153fefefb382fad1b4ce7e82d18582f40dbdbe495c9`.
Process all **125 full chunks**
(64,000 of 64,322 new IDs) from the same frozen source-ID file, starting a
fresh imatrix file. Allow **40 minutes** wall time, **50 GB** resident RAM,
and no GPU/T4. The pilot's observed 32-chunk run took about 6.5 minutes
including model initialization, so 125 chunks should fit with headroom;
stop and preserve logs if either ceiling is crossed. Merge the METH-05
106-chunk and METH-06 125-chunk BF16 matrices; **do not** also merge the
32-chunk pilot, which is a prefix of the full new run. The merged matrix
must have chunk count 231, all 254 required entries finite, and every
routed expert count ≥64. Only that artifact may feed the conversion gate.
