# C50-S3 protocol: the teacher signal (option A), 50 M tokens of F0 top-32 log-probabilities

20 October 2026. Cell S3 of the [CHATBOT50 roadmap](ROADMAP.md) §5, Phase 1. Pushed before any teacher data is
kept, with the corpus builder, the corpus manifest, the launcher and the fidelity verdict.

## Decisions (§12, 19 and 20 October)

- **Option A:** top-k logits from F0 (Ling-flash-2.0 `96f9b7a` Q4_K_M without imatrix, `5348c73f…`) on the local
  machine, with runs that can be stopped and resumed at any time, and a study of GPU offload and lossless speed-ups
  ([sizing](C50_S3_TEACHER_SIZING_20261019.md)).
- **Corpus option (a), 50 M tokens, downloads authorized** (20 October): a mix shaped on S2's gap map, in three equal
  parts. The teacher run is started and stopped from my own terminal over the coming days.

## Corpus v1

- **Builder:** `benchmarks/chatbot50/c50_s3_corpus.py` (this commit). Manifest:
  `results/c50_s3_corpus_v1_manifest.json` (every hash and count). Output on E: (`s3/corpus_v1/teacher.tsv`, the
  teacher's input, and `meta.jsonl`, the source of every chunk).
- **Sources, pinned by revision and file sha256:**

  | Part | Source | Revision | File | License |
  | --- | --- | --- | --- | --- |
  | wiki (knowledge: nq, tqa) | `wikimedia/wikipedia` 20231101.en | `b04c8d1c` | shard 0 of 41 (420 MB) | CC BY-SA 3.0, GFDL |
  | edu (school knowledge: mmlu, arc) | `HuggingFaceFW/fineweb-edu` sample/10BT | `87f09149` | `000_00000.parquet` (2.15 GB) | ODC-By |
  | math (reasoning: gsm8k) | `openai/gsm8k` main train, then `meta-math/MetaMathQA` | `740312ad`, `aa4f34d3` | train parquet; `MetaMathQA-395K.json` | MIT, MIT |

- **Rules** (the builder's docstring is the full text):
  - items walked in sha256("20261016:<source>:<key>") order; GSM8K train first within math;
  - text forms: wiki "title, blank line, text"; edu the text; math "Question: q / Answer: a", with GSM8K's
    calculator notes removed;
  - Ling-mini-2.0's tokenizer, no special tokens, as every C50 panel;
  - chunks of 2,048 tokens; a last window under 64 tokens is dropped; each part stops at 16,666,666 tokens;
  - the output interleaves the parts by tokens (the next chunk comes from the part with the fewest tokens written),
    so any finished prefix of the run is a balanced mix.
- **Exclusions (decontamination).** An item is dropped whole if any 13-token window of it occurs in any evaluation
  input. The inputs are every `*/inputs/*.tsv` of the campaign and the `c50_holdout_v1` / `c50_holdout_ext_v1` id
  files: 25 files, 17,647 sequences. They cover nq, tqa, gsm8k, arc, mmlu, piqa, ifeval, chat, rollout, the A4 FIT and
  A4-bis documents, A6 DEV and the holdout splits, hence S2's donor-gap panel. A window counts only if at least 8 of
  its 13 tokens contain a letter.
  - **Why the letter rule:** this tokenizer splits digits one per token, so a plain 13-token window matched date ranges
    and number lists ("the 1950s and 1960", "1, 2, 3, 4,"). That was 1.4% of Wikipedia articles and 0.8% of FineWeb-Edu
    documents, all false positives on inspection.
  - **Planted control** (before the build; scratchpad, not evidence of the corpus): 249 evaluation sequences, decoded
    and embedded in unrelated prose, were caught 248 times with the letter rule at 8 (244 at 10, all at 6). The miss is
    a code sequence whose windows are mostly whitespace. On the first 1,000 items of each source, 1 Wikipedia article
    matched.
- **Counts:** built on 20 October before this push (it scores nothing), wall 155 s.

  | Part | Items read | Dropped as contaminated | Short last windows dropped | Chunks | Tokens |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | wiki | 14,729 | 12 (0.08%) | 422 | 17,569 | 16,666,841 |
  | edu | 16,131 | 7 (0.04%) | 89 | 18,816 | 16,666,730 |
  | math | 71,936 | 18 (0.03%) | 153 | 71,765 | 16,667,157 |
  | **total** | | | | **108,150** | **50,000,728** (49,892,578 scored positions) |

  `teacher.tsv` sha256 `b748733a0da0f631b26f428c220e48c116cafa0a8073d484041195fd233e6ac8`. Every prefix of 1%, 10%, 50%
  and 90% of the documents holds the three parts at 33.3% of the tokens each (checked on `meta.jsonl`).
  A first build interleaved the parts chunk by chunk; it was replaced before this push because the short math chunks
  made the prefixes unequal in tokens.

## The teacher run

- **Tool:** `c50_s3_teacher.cpp` (19 October), GPU build linked to the official release of the pinned commit
  (b11040 = `5b335f4`, CUDA 12.4), binary `c50_s3_teacher_cuda.exe` sha256 `8a079d53…`.
- **Production configuration (fixed now):** fp32 cuBLAS (`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`), `--ngl 99 --cpu-moe`
  (attention, shared expert, embeddings and head on the RTX 3060; routed experts on the CPU), `--ctx 2048
  --ubatch 2048 --topk 32 --threads 6`, 48 documents per shard. KV cache F16, flash attention off, context cleared per
  document: the numerics of `c50_llama_score`.
- **Record:** per scored position 24 + 8 × 32 = 280 bytes: document, position, target, the target's log-probability,
  the log-sum-exp, and the top-32 (id, log-probability). **Only the log-probabilities are the teacher signal.** The
  stored log-sum-exp differs between the GPU and CPU paths by ~0.8 per position (sd) while the log-probabilities agree
  (§12, 20 October), so raw logits are never reconstructed from it.
- **Launcher:** `c50_s3_teach.py` (this commit). Every start checks the binary's hash, the corpus file's hash against
  the manifest, and the model (hashed in full on the first start, then size and modification time), sets the backend,
  asserts the CONFIG line, and appends one line per start to `launches.jsonl`. Ctrl+C stops it; the same command
  resumes it from the last finished shard (the tool refuses a resume with another identity). It refuses to start
  before the fidelity verdict below is PASS.
- **Volume and time (ESTIMATE):** 49.9 M positions × 280 bytes = **14.0 GB** on E: (`s3/teacher_v1/`). At the wall throughputs measured on 20 October (95-166 tok/s), that
  is ~3.5-6 days of machine. Short items (math) fill the 2,048-token batch less, so the lower end is likelier.
- **Machine:** the run reads no rate, so it does not need an empty machine. It holds the model's file mapping
  (62 GB) and the GPU; no other long job or rate cell runs beside it (rule: one long job at a time).

## Fidelity rule (frozen now)

The wish is that the GPU path costs no quality. F0 has no BF16 reference that fits this machine, so the reference
is F0's CPU path, the one S2's gap map was measured with.

- **Panel:** S2's MMLU panel (`s2/inputs/mmlu.tsv`, 2,280 sequences, 129,316 positions, at most 935 tokens). It
  was not used in the GPU study of 20 October.
- **Reference:** F0's CPU scores of that panel from S2 (`s2/f0/mmlu.pos.bin`: A4's scorer, ctx 4096, ubatch 512).
- **Treatment:** the production configuration above, run with `c50_s3_teach.py --fidelity` (120 sequences per shard).
- **Statistic:** per sequence, GPU − CPU bits per byte of the targets; the mean over sequences with a bootstrap CI95
  (10,000 draws, seed 20261016); and the top-1 agreement over positions.
- **PASS iff** the CI95 lies inside **[−0.002, +0.002] BPB** and the top-1 agreement is **≥ 95%**.
  - 0.002 is under 1% of F0's document gain over T0 in S2 (0.235 BPB).
  - On T0 the GPU paths differed from the CPU path by +0.0016 and +0.0035, and were as close to BF16 as the CPU path
    (§12, 19 October): the bound is stricter than any loss seen there.
- **Verdict:** `c50_s3_fidelity.py` (this commit) writes `results/c50_s3_fidelity_v1.json`. Its logic was checked on
  synthetic shards built from the CPU reference (scratchpad, no engine ran): identical → PASS; every target −0.01
  nats (+0.0033 BPB) → FAIL; top-1 agreement 89.9% → FAIL.
- **If it FAILs:** the same check with `--ubatch 512` (otherwise unchanged), by an addendum before it runs; if that
  fails too, the CPU build (`c50_s3_teacher_avx2.exe`, bit-identical to A4's scorer, ~20 tok/s), which needs no
  fidelity check. No teacher data is kept from a configuration that has not passed.

## Registered predictions (scored, not gated)

- The fidelity check PASSes, with the CI95 inside ±0.001 BPB.
- The run's mean throughput over its first day is 90-170 tok/s.

## What S3 hands to S4

- 50 M positions of F0 top-32 log-probabilities on a decontaminated, balanced corpus, with the source of every chunk.
- S4's protocol fixes how the signal is used (distillation loss, the matched data of arms (a), (b) and (c)) and may use
  any finished prefix: the order is a balanced mix at every point.

## What this cannot say

- **The teacher is the Q4_K_M without imatrix,** the F0 of S2, not BF16.
- **The fidelity check is against F0's CPU path,** not against BF16; T0's evidence is that both paths are equally
  close to BF16.
- **Decontamination by 13-token overlap** catches copies and near-copies, not paraphrases. MetaMathQA rephrases GSM8K
  train items, not test items.
- **Wikipedia 20231101 and FineWeb-Edu are very probably in F0's and T0's pretraining.** The signal is F0's
  distribution on them, which is what distillation transfers.
