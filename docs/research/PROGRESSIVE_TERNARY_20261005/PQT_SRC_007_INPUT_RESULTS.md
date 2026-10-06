# PQT-SRC-007: qualified disjoint original-tokenizer span inputs

6 October2026. Input preparation004 and independent audit pass. The frozen
384 C4 documents are ready for prospective original float-context execution.
No original model function, candidate fit, GPU job, quality measurement or
native timing has occurred. Input admission is not full capture qualification;
the whole research goal remains active.

## Source, selection and reconstruction

Original model google/switch-base-128 revision86c815ec05361a33a8b49fc717277da9c0a4e711;
small tokenizer/config bytes match the inherited complete acquisition record.
Every processed admissible text produces identical IDs through the raw fast
tokenizer and official T5TokenizerFast constructor, pinned Transformers4.57.6/
tokenizers0.22.2. Source originals were read-only, small copies owned separately.

Corpus allenai/c4 en validation shard00000-of00008 at
f998d2cd8b92435980789e3ecb2f89b4c68bfe1e:40,435,873 compressed bytes, SHA256
bc35d7c1b1d14b90cd3a394cccbcbe191935edd04bf42ee965379c6e2987a5f0.
One HTTP download2.515s; repairs reuse and rehash its exact owned bytes.
The pinned [card](https://huggingface.co/datasets/allenai/c4/blob/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/README.md)
states ODC-BY and references Common Crawl terms. Retain source URLs and license
attribution with raw documents. The model's [card](https://huggingface.co/google/switch-base-128)
identifies masked language modeling; the declared span sampler supplies that
conditioning without claiming an official benchmark implementation.

The source contains concatenated JSON objects without physical newline
delimiters. Bounded lexical framing handles quoted/escaped braces and nested
objects,64 KiB chunks and<=2 MiB record buffers. Preparation examines5,704
records, admits first4,096 unique eligible documents and discards1 oversized
text and1,607 short/special-token prefixes. No normalized document, URL or
prefix duplicate was encountered. The lexical reader decodes one extra record
before the loop stops; that record is excluded from selection. Read-ahead is
charged to decompression statistics. Neither the entire gzip stream nor the
entire dataset was parsed; complete compressed-file SHA is verified.

Fixed SHA ranking/buckets choose128 calibration,128 development and128 held_out
documents with disjoint normalized text identities, URLs and128-token prefixes.
No model output informed selection. Each input has19 removed tokens in six
positive spans,109 retained tokens in six spans, encoder116 IDs and labels/
teacher-forced decoder27 IDs. Both masks contain only real positions. Input
code/protocol/numbered framing amendment were frozen before004 at
8831dd17f005416f51cadab14642776924c63b11.

Independent stdlib audit binds all4,096 candidate records to the original gzip,
checks raw selected text/source identities and deterministic split ranking,
verifies sentinel IDs against actual tokenizer bytes and reconstructs every
original128-token input exactly from preserved encoder/label spans. All384
reconstructions and conditioning arrays pass. Source recheck reads13,172,736
decompressed bytes, maximum record buffer98,128 bytes. No tokenizer or model
is executed by this audit. A [supplementary seed audit](PQT_SRC_007_SEED_AUDIT_001.json)
also verifies all384 preserved span-length draws against their declared
document-specific seed and rejects a changed valid composition. It reads the
same retained documents, generating no new inputs;0.203s before report,
RSS33,959,936B. Its source is frozen atbc122645888b5dd63ce0cdcec0c9303e06042ff8;
this additional report/source is stored separately from the85-entry archive.

This establishes data independence between the frozen research partitions.
It does not establish absence from pretraining or absence of semantic overlap
with previous corpora. All subsequent baseline/candidate observations must be
logged as consumed evidence; later promotion requires appropriate fresh validation.

## First failures, repairs and costs

| Operation | Complete child seconds | Nested seconds before report | Process peak RSS bytes |
| --- | ---: | ---: | ---: |
| Inputs001, inherited serialization stop | 4.108917 | 3.500000 | 205,529,088 |
| Inputs002, newline framing cap stop | 9.092168 | 8.344000 | 336,973,824 |
| Inputs003, no eligible candidates under newline framing | 7.240598 | 6.422000 | 334,725,120 |
| Inputs004, qualified object framing | 24.914469 | 24.062000 | 372,736,000 |
| Independent source/split/reconstruction/retention audit | Not separately instrumented | 4.250000 | 79,405,056 |

Complete input children total45.356152s, including failures. Metadata discovery
001 stops0.797s on4 MiB response cap before corpus access;002 passes1.031s with
minimal fields/exact LFS pointer. First metadata response's observed4 MiB+1
prefix was not saved or hashed; its identity cannot be retroactively proved.
Keep this limitation explicit; do not describe all observed network bytes as
retained. No scientific corpus or model values were derived from that response.

Inputs001's own files match Git; inherited acquisition JSON differs only by143
CR bytes, raw7,058 bytes versus Git LF6,915. Preserve raw and canonical identity,
without editing inherited files. Inputs002/003 wrongly relied on newline framing.
[Repair001](PQT_SRC_007_INPUT_REPAIR_001.md) and [repair002](PQT_SRC_007_INPUT_REPAIR_002.md)
record pre-function changes. Scientific corpus/rank/eligibility/span/split/resource
limits are unchanged. No model result or preservation criterion was relaxed.

Framing001 rejects the other checkout's live Python before source reading,
0.453s/RSS22,667,264B. After authoritative disappearance, unchanged lexical reader
framing002 passes three positives/four malformed stops,0.625s/RSS22,691,840B.
Small additional command-level source-format observations precede these numbered
controls; their complete process/RSS instrumentation is unavailable, so total
research cost is not exhausted by the table. Shell startup, authoring/Git and
manual metadata probes are outside worker timers. No source/corpus redownload
or restart after observation timeout occurred.

## Exact retention and resumption

[Audit/retention](PQT_SRC_007_INPUT_AUDIT_RETENTION_001.json) preserves85 raw/source
entries/55,806,312 bytes in one ZIP_STORED archive55,824,724 bytes, SHA256
798ad097770c27dcb7f1d285103ff1304843d56cebbfcdc74e7c41d7548d92ff.
It includes complete compressed source, actual384 documents/IDs/masks/spans,
candidate manifest, tokenizer/config copies, four numbered input attempts and
controllers, metadata/format records and historical executable source closures.
[Git proof](PQT_SRC_007_INPUT_GIT_VERIFICATION.json) checks all retained entries
and nine files/55,912,272 bytes at8a13d697fad4d1b1a86cdea08030a77d5ad6df8e.
The missing first metadata prefix remains excluded from that exact-byte claim.

Keep results/progressive_ternary/PQT-SRC-007/inputs_001 through inputs_004,
source_discovery_001/002 and controller_inputs_001 through004 exclusive.
Do not regenerate splits, redownload source
or relabel the previous failures as successful runs. No live remote/local job.
Next implement/freeze original float execution/capture and independent
reference with complete-source budgets; see [execution direction](PQT_SRC_007_EXECUTION_DIRECTION.md).
Original natural expert coverage remains unmeasured. Native CPU timing awaits
the separately pending explicit owner availability window.
