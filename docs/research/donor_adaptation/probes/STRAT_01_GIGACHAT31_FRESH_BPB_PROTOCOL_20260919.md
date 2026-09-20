# STRAT-01 GigaChat 3.1: fresh-heldout BPB

Status, September 19, 2026: **PASS_FRESH_BPB**. V1 remains `VOID_CONTEXT_PREFLIGHT` with no forward pass; v2 is frozen and measured. The base model also has `PASS_SOURCE_BINDING`, so the BF16 teacher is fully bound to the base tensors of the source checkpoint.

## Corpus frozen before scoring

- Builder: `benchmarks/donor_adaptation/density/build_document_holdout.py`, current preset `strat01_fresh_v2`.
- Selection/span seed: `20260919`; maximum span **4.095 bytes**; minimum source size 4.096 bytes. For a byte-level tokenizer with non-empty tokens, 4.095 bytes cap the payload at ≤4.095 tokens and leave one position for BOS in the 4.096 context. The tokenizer preflight must still measure this and reject any violation.
- Quote: calibrazione 16 e heldout 32 documenti per ciascuna categoria `code`, `technical_general`, `prose` (48 + 96).
- Sources and deduplication remain as documented by the builder: local Python, local Markdown, and PG-19; global deduplication on the SHA-256 of the full content before selection.
- Both prior STRAT-02 JSONL files must be excluded: `calib.jsonl` SHA-256 `f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f` and `heldout.jsonl` SHA-256 `450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e`. Exclusion operates on both `source_document_id` and `source_content_sha256`; it must find 144 unique documents, and the new corpus must have zero intersection on both fields.
- Selection is based only on seed, category, split, and document hash, not model scores. After construction, hashes, IDs, and manifest become immutable.

### Pre-score addendum: v1 void

`strat01_gigachat_fresh_v1` was built write-once with 48+96 documents, 144/144 exclusions, and zero intersection; calibration SHA-256 `1b5d6313034adda8fa45c156c5a85b2856688e7437c45f98801c81781ebb811b`, heldout `4d90694f1caa7b66aea1a5172bd8663e493f55bd08a46dfb5f86232f482bb334`, manifest `dc1a43ea337265c5caf205e93239333260d723b35654bb8effab7667dcb5dcc1`. The source-tokenizer preflight, run **before any model**, accepted the entire calibration split (maximum 3.838 tokens) but rejected heldout at the first row exceeding the `payload+1 ≤4096` bound. No score or logit was produced. V1 remains `VOID_CONTEXT_PREFLIGHT` and is not modified.

V2 preserves the seed, ranking, quotas, and exclusions; only the mechanical span limit changes from 8.192 to 4.095 bytes, and output is written to a new directory. No model value was read to make this correction. The selected IDs must match v1; only spans/hashes may differ.

**V2 frozen:** `benchmarks/donor_adaptation/density/corpus/strat01_gigachat_fresh_v2/`, manifest SHA-256 `54d50cc64b5f874a4dec9bc7cd15afa416138fb2d139c493e37352ee7dbc250e`. Calibration: 48 documents, JSONL SHA-256 `68d9327a823328b1104c843afe986efeefcdd88c911c02895d577f65bffd8f81`, 54.633 source tokens and 196.560 bytes, maximum 1.927 tokens. Heldout: 96 documents, JSONL SHA-256 `04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e`, 109.989 tokens and 393.115 bytes, maximum 3.451 tokens. The 144 IDs match v1; the builder excluded 144/144 prior documents, and an independent check finds zero intersection for IDs and content SHA. `verify --check-source-files` and the full tokenizer round-trip pass.

Because the scorer, token mapping, alignment, batch-128/single-prefill, and two full runs had already been validated before v2 was created, the 48 v2 calibration documents do not require another forward pass: there is no parameter to calibrate, and doing so would add 50% compute without protecting heldout. The first v2 forward pass is directly on the frozen heldout, with Q4 and BF16 serialized and no apparatus changes.

## Apparatus and gate

Use the already validated GGUF source-ID scorer, BOS 1, empty KV per document, no payload EOS, `n_ctx=4096`, `n_batch=128`, 6 threads. Teacher: producer BF16 SHA-256 `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e`, now fully bound to the source checkpoint. Candidate: Q4_K_M SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

Calibration serves only to confirm executability; it does not change the corpus, scorer, threshold, or arm. On the full heldout, apply the previously fixed stratified paired bootstrap: 20.000 draws, seed `20260916`, sampling with replacement of 32 documents per category, linear quantile 0,95. `PASS_FRESH_BPB` if and only if the one-sided CI95 upper bound for ΔBPB Q4−BF16 is ≤+0,02. Also report categories, point estimate, and hashes; no document substitutions or new tolerance are allowed.

A pass authorizes the preregistered GigaChat tasks/rollout and execution path; it does not prove text→ID tokenization in the C engine, MTP, or ≥50 tok/s. A fail closes Q4_K_M for the current gate without automatically promoting other quantizations.

## V2 result

Both arms have 96/96 documents, with the same IDs/tokens/bytes, 109.989 tokens and 393.115 bytes. Q4 score SHA-256 `099bf46ba17b9e2f014e0cf6b9cb9782cc7fabe422162543c659e635d5372f8f`; BF16 `6e133ade9c174c662c2c64c1378c888c445f4a718e17779b93b0008ca9f85657`. Corpus, scorer, tokenizer, and parameters match in the manifests; the `llama.cpp` text-tokenizer mismatch is diagnostic on 95/96 in both, while inputs use the verified source IDs.

Preregistered adjudication: BF16 0,622631217933500 BPB; Q4 0,634954654131075 BPB; point Δ **+0,012323436197575**; one-sided CI95 upper bound **+0,013554844222859 BPB**, below +0,02. Category CI95 upper bounds: code +0,010866241725743, technical +0,014177883096846, prose +0,018224778128838. Outcome **PASS_FRESH_BPB**. Report: `benchmarks/donor_adaptation/density/results/strat01_gigachat_fresh_quality_v2/fresh_adjudication.json`.
