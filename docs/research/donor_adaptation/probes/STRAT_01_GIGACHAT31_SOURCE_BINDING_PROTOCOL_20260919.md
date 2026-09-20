# STRAT-01 GigaChat 3.1: full source-checkpoint → BF16 GGUF binding

Status, September 19, 2026: **PASS_SOURCE_BINDING**. The internal Q4−BF16 GGUF comparison passed its threshold, and the producer BF16 GGUF is now fully bound to the base tensors of the pinned source checkpoint. This pass does not cover MTP layer 26, which is absent from the base-only GGUF, or the fresh-heldout/task/engine gates.

## Frozen identities

- Sorgente: `ai-sage/GigaChat3.1-10B-A1.8B-bf16`, revisione `189fff27a1dee68473960c3d5bca53e0e07a3191`.
- Six shards `model-00000-of-00005.safetensors` … `model-00005-of-00005.safetensors`, 22.960.161.696 bytes total. Sizes and LFS SHA-256 values must match the API for the pinned revision; no `latest` file or moving revision is permitted.
- Reference pubblicato: `GigaChat3.1-10B-A1.8B-bf16.gguf`, revisione `97045b260251cfa86f5ad25638fa2dd074153446`, 21.356.281.984 byte, SHA-256 `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e`.
- Verification converter: **clean** checkout of `llama.cpp` commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. Do not use the modified MTP checkout without separating and recording the patches. Base-only BF16 output; MTP layer 26 must be explicitly omitted, as in the published 26-block GGUF.

## Verified acquisition

The snapshot is local at `benchmarks/donor_adaptation/density/results/strat01_gigachat_source_189fff27/`. All shards were reread in full after download and passed size and LFS SHA-256 checks: `00000` 4.986.065.392 B / `612f84d1…581b6`; `00001` 4.085.021.224 B / `5f104115…e2cf`; `00002` 4.085.022.040 B / `9a3f9ffc…e1df`; `00003` 4.085.022.240 B / `8fbb40d4…90071`; `00004` 4.085.022.240 B / `26c4261a…79b0`; `00005` 1.634.008.560 B / `cf53c1a4…6f8c`. Total 22.960.161.696 B.

The separate checkout `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4` is clean at `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`; SHA-256 for `convert_hf_to_gguf.py` is `c2b8d1f31a9007be75d6850aa1db47dedcaedee0eeebe00aa978eac6f2cc89ad`, and for `conversion/deepseek.py` `8f815e5e95c279f55ccca3e82ab82813a16d2996eb86db3c573c54095ffebe85`. The latter differs from the modified MTP copy, so experimental patches do not contaminate the binding conversion.

**Pre-GGUF apparatus addendum:** the converter's first launch indexed all six shards and mapped the 26 blocks, then exited with code 1 before creating the directory/output. The installed `transformers` applies strict validation and rejects `config.json:routed_scaling_factor` because the source serializes the numeric value as integer `1` while the current dataclass requires `float`. This is `VOID_TOOLCHAIN_PREWRITE`, not a result about the weights. The verified snapshot remains unchanged. A separate staging directory uses hardlinks to the same six shards and copies of the JSON files only; the sole modification is `routed_scaling_factor: 1 → 1.0`. This normalization preserves the mathematical value and is solely to allow the tokenizer/config to load; the hashes of the original and staging configs must be recorded in the final manifest. No other field or weight byte may change.

## Procedure and criteria

1. Acquire the source snapshot in a write-once directory, verifying each shard's size and SHA-256 against the pinned LFS metadata. Preserve the index, config, and tokenizer with provenance.
2. Convert the source checkpoint with the clean checkout, `--outtype bf16`, into a new base-only GGUF. Record the commit, converter Python-file hashes, command, output hash, and tensor inventory.
3. Compare the reconstructed and producer GGUFs by name, shape, type, and dequantized/exact value. Non-semantic metadata (for example, filename or producer string) may differ and must be listed; tokenizer, architecture, block count, and all tensors must match semantically.
4. For BF16/F32, require bitwise identity after any cast explicitly specified by the converter. For deterministic structural transformations (expert stacking, MLA split, transpose/GGML ordering), compare the canonical array reconstructed from the source, not just container bytes. No sampling.
5. If the clean converter does not support this geometry or produces a different mapping, classify as `VOID_TOOLCHAIN` and implement a streaming comparator from the pinned logic; do not promote sampled binding.

Allowed outcomes: `PASS_SOURCE_BINDING` only with complete coverage; `FAIL_SOURCE_BINDING` if at least one semantic value/shape differs; `VOID_TOOLCHAIN` if the apparatus is infeasible. A pass does not prove tasks, rollout, text→ID tokenization in the C engine, MTP, or throughput. Source files are large local artifacts and must not be added to Git.

## Result

The clean conversion produced `GigaChat3.1-10B-A1.8B-source-bf16.gguf`, 21.356.264.448 B, SHA-256 `fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`. The producer file remains 21.356.281.984 B, SHA-256 `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e`; the size difference is in metadata/header, not tensors.

The reproducible comparator `benchmarks/donor_adaptation/density/strat01_gigachat_source_binding.py` verified:

- 414/414 identical names, types, shapes, and sizes;
- all 21.350.179.072/21.350.179.072 tensor-payload bytes compared in full and identical, zero mismatches;
- a manifest of all 414 tensor hashes with canonical digest `c4d3e3dca919a6b24f8ca2a389d3f8624523a3c1acf19d685d910dd7693fd309`;
- all 48 common metadata keys identical except `general.name` and the internal `GGUF.kv_count` counter; the producer adds only `general.finetune` and `tokenizer.chat_template`;
- staging config SHA-256 `0c9f747879dcb35bac28ac4e864b849bd9995e0b8552405681878cb17d558dbe`, with the sole declared normalization relative to original SHA-256 `6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503`.

Full report: `benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/source_binding_report.json`, SHA-256 `0cd0699fb100e66eb89b37ed088a942228446c5b20939dcc682eec186772182f`. Outcome: **PASS_SOURCE_BINDING for the base model**. The producer BF16 can now serve as the tensor-level teacher for the source checkpoint on base gates; the claim does not extend to MTP, text→ID tokenization in the C runtime, tasks, or rate.
