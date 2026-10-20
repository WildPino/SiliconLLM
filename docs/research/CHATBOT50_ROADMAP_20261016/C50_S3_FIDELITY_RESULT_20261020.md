# C50-S3 fidelity result: the GPU configuration FAILs; the teacher runs on the CPU build

20 October 2026. Protocol: [C50_S3_TEACHER_PROTOCOL_20261020](C50_S3_TEACHER_PROTOCOL_20261020.md), "Fidelity rule",
and its addendum 1 (this commit). Verdict file: `results/c50_s3_fidelity_v1.json`, written by `c50_s3_fidelity.py`.
Run data: `E:/.../s3/fidelity_v1/` (19 shards, `launches.jsonl`, `teach.log`). Diagnosis: `E:/.../s3/fidelity_diag/`
(scripts, logs, `diagnosis.txt`). No teacher data was kept.

## Verdict

**FAIL on both conditions.** S2's MMLU panel, 2,280 sequences, 129,316 positions, production configuration (fp32
cuBLAS, `--ngl 99 --cpu-moe`, ctx and ubatch 2048), against F0's CPU scores of S2. The model hash, the binary hash and
the CONFIG line all confirmed. Launched 15:22; shard wall 4,315 s.

| Statistic | Value | Rule | Inside |
| --- | --- | --- | --- |
| GPU − CPU dBPB, mean over sequences | **+0.01119** | | |
| its CI95 (10,000 draws) | **+0.01028..+0.01210** | inside ±0.002 | no |
| top-1 agreement over positions | **93.27%** | ≥ 95% | no |
| mean / max abs dlogp | 0.178 / 5.63 nats | reported | |

## Diagnosis (after the verdict; reported, not gated)

**84% of the gap is one context counted 2,280 times.** Every MMLU sequence starts with the same token ("Question"),
so position 1 (the prediction of ":" after one token) is the same context in every sequence. There the GPU path gives
the target 1.29 nats less than the CPU path on average (2,280 positions, sd 1.11, only 10 distinct values: the value
depends on the document's length, i.e. on the batch size the kernels see). The bootstrap over sequences treats it as
2,280 independent draws. This is a weakness of the panel I chose, found after the result; the rule is not changed.

| Positions | Positions | dBPB (CI95) | top-1 |
| --- | ---: | --- | ---: |
| all (the rule) | 129,316 | +0.01119 (+0.01028..+0.01210) | 93.27% |
| without position 1 | 127,036 | +0.00274 (+0.00188..+0.00361) | 94.88% |
| position ≥ 8 | 113,356 | +0.00314 (+0.00192..+0.00436) | 94.73% |

**Without that position the GPU path still loses +0.0027 BPB**, outside the rule's ±0.002. It is the size the GPU
paths showed on T0 on 19 October (+0.0016 and +0.0035); the bound 0.002 was set tighter than that measured spread.
The GPU study's +0.0006 on the FIT documents does not carry over: those are long documents (2,304 tokens on average,
CPU BPB 0.54), MMLU's are short (57 tokens, CPU BPB 1.10).

**Four variants on the first 120 sequences** (shard 0, 4,366 positions):

| Variant | dBPB vs CPU reference | position 1 dlogp | dBPB without position 1 | top-1 | against production |
| --- | --- | --- | --- | --- | --- |
| production (ngl 99, ubatch 2048, fp32 cuBLAS) | +0.01237 | −1.417 | −0.00097 | 93.63% | |
| ngl 99, **ubatch 512** | +0.01237 | −1.417 | −0.00097 | 93.63% | **byte-identical** |
| CUDA build, **ngl 0** (CPU, no GPU layer) | +0.00855 | −1.382 | −0.00431 | 94.07% | differs |
| ngl 99, default cuBLAS compute | +0.01233 | −1.417 | −0.00101 | 93.38% | differs (3% of log-probs equal) |
| **AVX2 build, ctx 4096, ubatch 512** | **0 (bit-identical)** | 0 | 0 | 100% | |

- **ubatch does not matter for a sequence shorter than the batch.** Only 4 of the 2,280 sequences exceed 512 tokens,
  so the protocol's first fallback (ubatch 512) would change at most 4 sequences of this result: its FAIL is decided
  by construction, and it was not run (addendum 1).
- **The position-1 gap is not the GPU's arithmetic.** The CUDA release build gives it on the CPU too (ngl 0), and at
  position 1 its CPU and GPU paths give the same distribution; the build used for S2 (`c50_build_scorer.sh`, AVX2)
  gives another.
- **The AVX2 build reproduces S2 bit for bit:** the protocol's second fallback needs no fidelity check, as written.

**Which one is right at position 1: neither.** F0 BF16 (206 GB, read from the USB drive by mmap; a few-token input
reads only the experts it routes to, ~37 min per run) on the first 4 tokens of sequence 0:

| Position (target) | BF16, AVX2 build | BF16, CUDA build ngl 0 | Q4 AVX2 | Q4 CUDA ngl 0 | KL(BF16 ‖ Q4) on BF16's top-32, AVX2 / CUDA |
| --- | --- | --- | --- | --- | --- |
| 1 (":") | −2.558 | −2.558 | −4.105 | −4.092 | 0.125 / 0.130 |
| 2 (" Find") | −8.266 | −8.259 | −7.976 | −7.924 | 0.0013 / 0.0028 |
| 3 (" the") | −0.185 | −0.184 | −0.171 | −0.166 | 0.0020 / 0.0015 |

After one token of context the quantized F0 is ~1.5 nats from BF16 in both builds, and the two builds agree on this
4-token batch. From position 2 both are close to BF16. The prediction after one token is ill-conditioned for the Q4
model, and the batch-size dependence moves it by up to 1.4 nats. This is one context, so it says nothing about which
path is closer to BF16 elsewhere.

## Registered predictions, scored

- **The fidelity check PASSes, with the CI95 inside ±0.001 BPB: WRONG** (CI +0.0103..+0.0121; +0.0019..+0.0036 even
  without the shared position).
- **First-day throughput 90-170 tok/s:** not scored. The GPU configuration did not run.

## What follows

- **By the protocol the teacher is the CPU build** (addendum 1): `c50_s3_teacher_avx2.exe`, ctx 4096, ubatch 512,
  6 threads, the numerics of S2's F0 scores. The launcher (`c50_s3_teach.py`) now runs it; its CPU path was tested on
  a 4-token input (CONFIG confirmed, output byte-identical to the diagnosis run).
- **Its cost (ESTIMATE):** about 20 tok/s, ~1.7 M tokens per day. The 50 M tokens decided would take ~29 days of
  machine; every finished prefix is a balanced mix, so S4 can start from any prefix.
- **The GPU path would take ~5 days (ESTIMATE)** but differs from S2's F0 by +0.0027 BPB on MMLU outside the shared position. Using
  it would override a frozen rule. That is not done here; it is offered as a decision (`docs/COMMUNICATION.md`).
- **In the corpus**, every math chunk starts with "Question:", so 71,765 positions (0.14%) sit at the ill-conditioned
  context; on the CPU build they are as in S2.

## What this cannot say

- Whether the GPU path is closer to or farther from BF16 than the CPU path, beyond the three positions above. F0 BF16
  does not fit in RAM, and from the USB drive a panel of long documents would take days.
- The panel's shared first token weighs the rule's statistic; another panel would have weighed it differently. The
  rule stands as frozen.
