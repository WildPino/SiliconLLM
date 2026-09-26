# METH-17: donor-relative quality on externally sourced documents

**Prospective scoring protocol.** METH-16's trained E128 residual
student improved BPB on Qwen development windows, but those windows
share a corpus with earlier method selection. This test asks whether
the **same frozen checkpoint** retains donor-relative quality on
documents excluded from both Qwen calibration and heldout files.
Its outcome decides whether to spend effort on native export of this
specific checkpoint. No weights, router, precision or training
hyperparameters may change based on these results.

## Data binding before scoring

The [deterministic selector](../../../benchmarks/donor_adaptation/s1/meth17_fresh_transfer_audit.py)
has already run in `--prepare` mode without loading a model or
computing loss. Its [manifest](meth17_fresh_document_manifest.json)
has SHA-256
`7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8`.
It selected **60 distinct 4,095-byte-or-shorter spans**, 245,700
UTF-8 bytes total: 24 code, 24 prose, 12 general technical.

- Code comes from `.c`/`.py` sources under `benchmarks/phase*` in
  pre-METH-15 Git commit
  `45afab6847cfd328c105f73256441f36748dc323`. The 74 eligible
  files had ≥4,095 bytes; SHA-based seed `meth17-17017` chooses a
  span per file, then ranks source paths and retains 24. These source
  files were not the external Python corpus used to train METH-16.
- Prose and technical text come from the locally bound
  `strat02_document_holdout_v1/heldout.jsonl`, SHA-256
  `450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e`.
  That corpus was previously used in the sparse-donor GigaChat line,
  **not** in selecting or training METH-15/16. The selector ranks
  source IDs with the same seed; it takes 24/32 prose and all 12
  technical documents passing the overlap screen.
- The overlap screen checks the first, middle and final 256 UTF-8
  bytes of every candidate span against both Qwen corpus files:
  `calib.txt` SHA-256
  `10d4d28102625f399f715b6aa220b234c5015dcff199997ccafa06e5c59c89d0`
  and `heldout.txt` SHA-256
  `f46b0310c15faec59ca805d5688317d53b7655ae7099008fe5c3439460d58312`.
  Twenty of 32 technical external candidates were excluded; no
  selected span has a matching tested fragment. This is a bounded
  **partial-overlap screen**, not proof against every paraphrase,
  shorter common string or pretrained-donor contamination.

The selector had first considered the earlier `strat01_fresh_v2`
corpus; 40 of its 96 spans occur in full in the Qwen corpus and all
32 code rows share a tested 256-byte fragment. It is excluded from
this test. The current manifest was frozen before model scoring.

## Fixed model and scoring rule

Donor: `Qwen/Qwen2.5-0.5B` revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, source safetensors
SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`. Student: METH-16's
update-1024 adapter/optimizer checkpoint, SHA-256
`a9e74c9ceeb8a91fd04f9b9e35a981965db9ab824d2d25ea728dd940ea5365a4`.
Use RTX 3060 BF16/SDPA, frozen model `.eval()`, six CPU threads,
and **identical model instance and text IDs** with experts disabled
(donor) or enabled (student). No quantization.

For each document, tokenize the selected UTF-8 text with
`add_special_tokens=False`; prepend the donor EOS token as document
start and score **every text token**. Use 512-token target strides;
each window sees up to 512 preceding text tokens. Supply document
absolute position IDs. Sum natural-log loss by document and divide
by `ln(2) × selected UTF-8 bytes` for BPB. Keep all document rows
and category totals. The evaluator must reconstruct and compare the
manifest before model loading, and verify all source/checkpoint hashes.

Pool document nats and bytes for the primary paired delta. Bootstrap
documents **within each category**, preserving 24/24/12 counts,
20,000 draws with seed 171717. The one-sided 95% upper bound is the
95th percentile of paired delta BPB. The fresh-document gate passes
only if pooled delta is **≤+0.01 BPB**, upper bound **≤+0.02 BPB**,
and each category's pooled point delta **≤+0.05 BPB**. These bounds
are fixed before observing any model score. Report donor and student
absolute BPB, every document/category delta, token/byte counts and
CI even on failure. A pass supports export work; it does not prove
tasks, useful free generation, low-bit quality or native speed.

Run on the local RTX 3060 only. Stop at 30 minutes wall time,
10.5 GiB peak allocated GPU memory or 20 GiB process RSS. No T4
is requested. If an apparatus/hash error occurs before scoring,
repair only that error and preserve the invalid attempt in the record.
