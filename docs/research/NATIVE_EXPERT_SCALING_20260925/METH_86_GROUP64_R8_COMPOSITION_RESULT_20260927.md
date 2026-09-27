# METH-85/86: grouped R8 FFN core export and fresh composition

The [frozen protocol](METH_85_GROUP64_R8_PROTOCOL_20260927.md) tests whether replacing METH-62's single FFN row scale with one FP16 scale per 64 weights repairs quality within the 560 MB ideal addressed-payload limit. The donor is `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`, source SHA256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; the distinct E128 product-key checkpoint is SHA256 `8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`.

## Stored artifact and cost

`meth85_export_group64_r8_core.py` writes the [export report](meth85_group64_r8_core_export.json) and a local stored safetensors artifact with SHA256 `c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a`. It reuses the exact METH-59 row-R8 tied head, stores 96 attention matrices as BF16, 72 FFN matrices as int8 with group-64 FP16 scales, and 121 controls as FP32. All 363 stored tensors reload exactly; every FFN BF16 reconstruction from reloaded codes/scales equals its pre-save reconstruction. Physical file size is 548,741,256 bytes. The ideal per-token addressed ledger is 548,701,184 core + 5,652,480 product-key router + 2,752,512 selected E128 factors = **557,106,176 bytes**, leaving 2,893,824 bytes below the 560 MB design limit. This is ideal byte arithmetic, not measured C traffic or rate. Export on local RTX 3060 took 18.19 s and peaked at 4.22 GB process RSS.

## Fresh development result

The [manifest](meth86_group64_r8_dev_manifest.json), SHA256 `0d4ac34beea84ecbcaf92bac088345ccab5cc3a2e3f7898710018248bf4bca0c`, fixes 24 source-disjoint documents and their chat prompts: eight each from code, PG19 prose, and technical documentation. It excludes prior source IDs and overlapping 256-byte fragments; the separate PG19 validation parquet supplemented the depleted PG19 test pool before scoring. The [result JSON](meth86_group64_r8_composition_result.json) binds all document/prompt IDs and the donor, checkpoint, artifact and manifest hashes. Inference used one local RTX 3060, 39.91 s, 2.87 GB process RSS and 2.45 GB peak GPU allocation.

| Measure | BF16 donor | BF16+E128 | grouped core donor | grouped core+E128 |
|---|---:|---:|---:|---:|
| Pooled BPB, 98,280 bytes | 1.250056 | 1.245813 | 1.250470 | 1.246357 |
| Donor top-1 matches / 4,169 prompt positions | 4,169 | 4,005 | 3,961 | 3,943 |
| Donor top-1 agreement | 100.000% | 96.066% | 95.011% | **94.579%** |

The grouped E128 arm's pooled document delta is −0.003699 BPB; code, prose and technical deltas are −0.001636, −0.005190 and −0.004272. All likelihood and each-category prompt floors pass. Pooled top-1 misses its 95% floor by 0.421 point. It trails BF16+E128 by 1.487 points, exceeding the allowed 1.0 point. Two of five preregistered gates fail. **Stop this core as a quality-valid native candidate.** Do not claim that the document improvement establishes retained generation or semantic quality.

## Posthoc attribution on viewed sources

`meth87_factor_scale_diagnostic.py` reused the METH-86 prompts after the decision. Halving the E128 B-factor output on the grouped core lowered pooled agreement to 94.219% and reduced the document benefit to −0.001888 BPB ([JSON](meth87_factor_scale_050_viewed_result.json)). Restoring the donor BF16 tied head, an over-budget diagnostic, raised grouped-core donor agreement to 96.210% but grouped-core+E128 only to 94.939% ([JSON](meth87_bf16_head_viewed_result.json)). This local result implicates both head precision and interaction with the expert residual; it does not isolate a unique cause. The prompts are viewed and cannot serve as independent promotion evidence for a follow-up.

No full `engine.c` inference, CPU LUT/kernel timing, varied-token DRAM measurement, semantic audit or ≥50 accepted tok/s result exists for this artifact. The next candidate needs an explicit quantization-aware expert/core adaptation rule with separate development and external sources, while keeping the large-E learned-quality and 10× CPU routing constraints visible.
