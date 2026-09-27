# METH-85/86: group-64 R8 FFN recovery

## Question and frozen decision

Can finer FFN scaling repair the METH-62 mixed-core quality gap without exceeding the 560,000,000-byte ideal addressed-payload design limit? The source is the pinned Qwen2.5-0.5B-Instruct donor, the expert bank is the METH-56 update-512 E128 product-key checkpoint, and the tied head remains the exact METH-59 per-row R8 representation. Attention stays BF16 and controls FP32. Each FFN matrix uses signed symmetric int8 codes in [-127,127], one FP16 scale per contiguous group of 64 input columns per output row. Quantization divides by the *stored* FP16 scale and rounds to nearest. Reconstruction multiplies codes by the stored scale then rounds to BF16.

The source tensor header gives 313,786,368 FFN weights. This format replaces 1,019,904 FP32 FFN row-scale bytes with 9,805,824 FP16 group-scale bytes, adding 8,785,920 bytes to the METH-62 mixed core. Ideal core payload is 548,701,184 bytes; adding 5,652,480 router bytes and 2,752,512 selected factor bytes yields 557,106,176 bytes per token. This is accounting, not a measured DRAM trace or native rate. The serialized file also has a header.

METH-85 exports a stored artifact. It must pass exact tensor reload and BF16 reconstruction readback. Stop if the ideal addressed total exceeds 560,000,000 bytes. One local RTX 3060, 15-minute wall budget, 20-GiB process RSS limit, and 10.5-GiB GPU allocation limit apply. No T4.

METH-86 freezes 24 fresh source-disjoint documents (8 code, 8 prose, 8 technical) and corresponding chat prompts before scoring. Compare intact BF16 donor, BF16 donor plus E128, grouped-core donor, and grouped-core plus E128 with the same token IDs. Development gates for the grouped E128 arm: pooled document loss ≤ +0.02 BPB versus intact donor; each category ≤ +0.04 BPB; pooled donor top-1 agreement ≥95%; each category ≥90%; pooled top-1 no more than one point below BF16+E128. Every gate must pass before an independent generation, task, and blind semantic audit. A development pass alone is not native or final quality promotion. A failure stops this format and is retained in the method record.

Before scoring, the first manifest build found only seven eligible PG19 test prose sources after prior-source and fragment exclusions. The selection pool therefore includes the separate PG19 validation parquet (SHA256 `81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef`), while retaining the eight-per-category gate and all source/fragment exclusions. This was decided before any METH-86 model score was observed.
