# C50-A6 protocol, FIT stage: the precision ladder P1 → P4 on Ling-mini

17 October 2026. Cell A6 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed before any A6 arm
is scored.

**Status.** Tooling and GGUF builds can proceed. The scoring run takes ~5 h of CPU (6 arms of
~46 min plus 4 builds of ~9 min), so it waits for the owner. It is also only worth running if
the owner keeps Ling-mini after the A5 FAIL (COMMUNICATION item 3, option A).

## What changes from the roadmap's ladder

- **The head stays exact (264.1 MB).** A5 failed, so the ladder's P2 head lever is unavailable.
- **Remaining levers:**
  - int8 KV (P1);
  - IQ3_S routed experts (P3), with IQ4_XS as the risk register's fallback;
  - a BF16 router (P4).

## Arms

There is one fixed runtime for every arm, so the deltas are not runtime-path deltas (an A2b
lesson). The runtime is `c50_llama_score_a6_avx2.exe`, sha256 `84763138…`:
- the A2b scorer plus two printed options, `--kv f16|q8_0` and `--fa 0|1`;
- flash attention is **on** in every arm (llama.cpp needs it for a q8_0 V cache);
- 6 threads, ubatch 512, ctx 8192.

| Arm | GGUF | KV | Ladder step |
| --- | --- | --- | --- |
| R | P0 Q4_K_M (A2 pin `44de7315…`) | f16 | reference |
| K8 | P0 | q8_0 | P1 |
| E3 | Q4_K_M with `ffn_{gate,up,down}_exps` = IQ3_S | q8_0 | P3 |
| E4 | Q4_K_M with routed experts = IQ4_XS | q8_0 | P3 fallback |
| E3B | E3 + `ffn_gate_inp` (router) = BF16 | q8_0 | P4 |
| E4B | E4 + router BF16 | q8_0 | P4 on the fallback |

**Builds:**
- **Command:** each built GGUF is `llama-quantize --imatrix <fit24 imatrix> --tensor-type … <A2 BF16 GGUF> <out> Q4_K_M 6`, with the pinned llama.cpp `5b335f4`.
- **Imatrix:** the same as P0 (the first 24 FIT documents per category).
- **Check:** each output passes `c50_a2_gguf_check.py`, which checks the vocabulary and
  prices the weight bytes per token.

**Panel:** the A4 document panel. That is the 72 non-imatrix FIT documents (FIT items 25-48 per
category), 165,930 tokens, `docs.tsv` pinned by the A4 inputs manifest. The panel touches no
DEV or RESERVED data.

**Scripts:**
- `c50_a6_run.py` builds, checks, scores and asserts CONFIG: AVX2, threads, ubatch,
  `flash_attn=enabled`, the KV type and the model path. It is resumable and budgeted.
- `c50_a6_verdict.py` applies the rule.

## Statistic and FIT rule (fixed now)

- **Statistic:** paired dBPB of each arm against R, document bootstrap (10,000 draws, seed
  20261016), with the same byte count as A2b/A3/A4.
- **Bytes per token at context 2048:**
  - weights: per-token weight bytes from the GGUF check;
  - plus KV: 20 layers × 2 × 512 values × 2048 positions, at 2 B (f16) or 34/32 B (q8_0),
    which gives 83.9 or 44.6 MB.
- **FIT candidate:** the arm with the fewest bytes per token whose **upper CI95 ≤ +0.015**
  (ROADMAP §5 A6, applied on FIT).
  - If no arm beyond R qualifies, A6 stops at P0.
  - Top-1 agreement with R is reported, not gated.
- **Later (a second protocol, before it runs):**
  - the FIT candidate is confirmed once with DEV dBPB on DEV items **13-32** per category (60
    documents, not read by A5, which captured items 1-12);
  - the A4 task panels are run on that candidate.

## Disclosed before the run

- **The scorer smoke read one A6 quantity.** It used FIT document 0, an imatrix document
  outside the panel:
  - at default options the A6 scorer is bit-identical to the A2b scorer (`cmp`);
  - flash attention on changes 17 of 458 top-1s;
  - KV q8_0 with flash attention moved the mean target log-probability by −0.036 nats
    against the A2b scorer.

  That is one document. It is not a measurement of K8, but it was seen.

## Registered prediction (scored, not a gate)

| Arm | Predicted dBPB vs R |
| --- | --- |
| K8 | +0.003 to +0.015 (the smoke document suggests the upper half) |
| E3 | +0.03 to +0.06 (IQ3_S on all routed experts; Phase-B healing was budgeted for this) |
| E4 | +0.01 to +0.025 |
| E3B, E4B | within ±0.002 of E3, E4 |

- **FIT candidate:** K8 or E4B.
- **Budget: P4 does not reach B = 530 MB with any arm while the head stays exact.**
  - Ledger: 461 − 23.8 + 264.1 ≈ 701 MB with IQ3_S.
  - The measured GGUF byte counts replace the ledger.

## What this cannot say

- **No rate:** bytes are counted, not streamed.
- **No task quality:** that is the second protocol.
- **One donor:** it says nothing about other donors (option C).

## Addendum 1 (17 October 2026, after the builds, before any arm is scored)

**What went wrong.** The first E3B and E4B builds are VOID:
- They are byte-identical to E3 and E4 (same sha256, `05ca3057…` and `993afe58…`).
- The pinned `llama-quantize` never quantizes `ffn_gate_inp`
  (`src/llama-quant.cpp:306`, "do not quantize expert gating tensors"). It ignored the
  `--tensor-type ffn_gate_inp=bf16` override without a warning.
- The records are kept under `builds_void` in `results/c50_a6_run.json`.

**Fix.** E3B and E4B are now E3 and E4 with each router rewritten from F32 to BF16 by
`c50_a6_router_bf16.py`:
- It uses the pinned gguf-py, with round-to-nearest-even, the same rule as
  `ggml_compute_fp32_to_bf16`.
- It copies metadata, tensor order and alignment.
- It re-reads the output and checks:
  - 19 routers equal the BF16 rounding of the source;
  - the other 259 tensors are byte-identical.
- The runner builds them from the recorded base GGUF after checking its sha.
- A test build and the runner's build have the same sha (`43fb338d…`), so the rewrite is
  deterministic.

**Measured while fixing** (a property of the checkpoint, not an A6 quality number):
- **The F32 routers are exactly BF16-representable:** the maximum rewrite error is 0, because
  the source checkpoint is BF16.
- **So P4 changes no router weight.** It halves the router bytes, 39.8 → 19.9 MB/token.
- **The only numerical change is in the compute path.** ggml's BF16 matmul rounds the router
  input activation to BF16.
- **The treatment treats.** On the smoke document (FIT doc 0, off-panel), all 458 records
  differ between E3 and E3B. No statistic was computed.

**Measured weight bytes per token** (GGUF check; KV at context 2048 added as registered):

| Arm | Weights MB/token | + KV | Total at 2048 | Desk tok/s |
| --- | --- | --- | --- | --- |
| R | 783.4 | 83.9 (f16) | 867.2 | 34.4 |
| K8 | 783.4 | 44.6 (q8_0) | 827.9 | 35.9 |
| E3 | 700.4 | 44.6 | 744.9 | 39.6 |
| E4 | 748.9 | 44.6 | 793.5 | 37.3 |
| E3B | 680.5 | 44.6 | 725.0 | 40.6 |
| E4B | 729.0 | 44.6 | 773.6 | 38.2 |

- **No arm reaches B = 530 MB**, as registered.
- The ledger's ~701 MB for IQ3_S plus an exact head corresponds to E3 (700.4 MB of weights).
- The arms, statistic, rule and predictions are unchanged.

### Byte ledger of E3B by organ (measured, GGUF check of `43fb338d…`)

These are bytes per decoded token at context 2048. Routed experts count at 8/256 of their
stored 6,574.6 MB.

| Organ | MB/token | Share of 725.0 |
| --- | --- | --- |
| head (Q6_K, exact) | 264.1 | 36.4% |
| routed experts (IQ3_S) | 205.5 | 28.3% |
| attention (Q4_K) | 134.4 | 18.5% |
| KV q8_0 | 44.6 | 6.2% |
| shared experts | 36.1 | 5.0% |
| dense FFN (layer 0) | 20.4 | 2.8% |
| router (BF16) | 19.9 | 2.8% |
| norms, embedding row | 0.2 | 0.0% |

**Desk arithmetic** (not a rate):
- 50 tok/s at 100% of desk needs ≤ 576 MB/token; at the 0.85 of desk measured for a complete
  C engine, it needs ≤ 480 MB.
- E3B is 149 MB over the first figure and 245 MB over the second.
- Swapping the exact head for the A5 part-1b preview (94.5 MB, failed on DEV) would make it
  556 MB, which is 52.6 desk tok/s, or 44.7 at 0.85.

## Addendum 2 (17 October 2026, after A1, before any arm is scored)

**Why.**
- [A1](C50_A1_A2B_RATE_RESULT_20261017.md) measured IQ3_S on the eight-expert shapes at
  **12.45 GB/s** (FAIL against 30). Every other class passed at 35-41 GB/s.
- The A1 protocol fixed the consequence before its run: A6 prices a failing class at its
  measured bandwidth instead of 32 GB/s.
- Bytes alone no longer order the arms, because an IQ3_S byte costs 2.6× a byte of any other
  class.

**Change to the FIT rule.**
- **Cost:** desk ms per token at context 2048 =
  (bytes outside IQ3_S) / 32 GB/s + (IQ3_S routed bytes) / 12.45 GB/s + 2 ms.
- **FIT candidate:** the qualifying arm (upper CI95 ≤ +0.015, unchanged) with the lowest cost.
- **For arms without IQ3_S** this is the same order as bytes.
- `c50_a6_verdict.py` prints MB and desk ms for every arm.
- Arms, statistic, bound and predictions are unchanged. E3/E3B are still scored: their
  quality cost is worth knowing if the IQ3_S kernel is tuned later.

**Repriced arms** (desk, not a rate; T_FIXED stays 2 ms, although A1 measured a 2.70 ms residual):

| Arm | MB/token at 2048 | IQ3_S MB | Desk ms | Desk tok/s |
| --- | --- | --- | --- | --- |
| R | 867.2 | 0 | 29.10 | 34.4 |
| K8 | 827.9 | 0 | 27.87 | 35.9 |
| E3 | 744.9 | 205.5 | 35.37 | 28.3 |
| E4 | 793.5 | 0 | 26.80 | 37.3 |
| E3B | 725.0 | 205.5 | 34.74 | 28.8 |
| E4B | 773.6 | 0 | 26.18 | 38.2 |

- The registered FIT-candidate prediction was "K8 or E4B". It still stands, and is now
  favoured on cost.
