# STRAT-01 — Susono-10B-A1B-Base metadata screen

**Screen date: 2026-09-18.** Read-only metadata sidecar. No weights were
downloaded, no custom code was executed, no T4 was used, and no port,
benchmark, quality test, rate measurement, acquisition, or commit is
authorized by this screen. The main-agent decision is fixed: **metadata-only
candidate; not promoted to acquisition or port**.

## Identity and primary sources

- Model: [puwaer/Susono-10B-A1B-Base](https://huggingface.co/puwaer/Susono-10B-A1B-Base)
- Immutable Hugging Face revision:
  [`01031e1a808224a29c616e4f85bb94f2604a7a8d`](https://huggingface.co/puwaer/Susono-10B-A1B-Base/tree/01031e1a808224a29c616e4f85bb94f2604a7a8d)
- [HF API at the immutable revision](https://huggingface.co/api/models/puwaer/Susono-10B-A1B-Base/revision/01031e1a808224a29c616e4f85bb94f2604a7a8d)
- [`config.json`](https://huggingface.co/puwaer/Susono-10B-A1B-Base/raw/01031e1a808224a29c616e4f85bb94f2604a7a8d/config.json)
- [`model.safetensors.index.json`](https://huggingface.co/puwaer/Susono-10B-A1B-Base/raw/01031e1a808224a29c616e4f85bb94f2604a7a8d/model.safetensors.index.json)
- [Pinned model card](https://huggingface.co/puwaer/Susono-10B-A1B-Base/raw/01031e1a808224a29c616e4f85bb94f2604a7a8d/README.md)
- Official implementation fork, inspected only at commit
  [`a18d33fbbcfe595d0e9f5923899d3f6d5b4a8f6a`](https://github.com/puwaer/transformers/tree/a18d33fbbcfe595d0e9f5923899d3f6d5b4a8f6a):
  [`configuration_susono.py`](https://github.com/puwaer/transformers/blob/a18d33fbbcfe595d0e9f5923899d3f6d5b4a8f6/src/transformers/models/susono/configuration_susono.py),
  [`modeling_susono.py`](https://github.com/puwaer/transformers/blob/a18d33fbbcfe595d0e9f5923899d3f6d5b4a8f6/src/transformers/models/susono/modeling_susono.py)

The card declares Apache-2.0, English/Japanese, 24 layers, hidden size
2,048, vocabulary 151,680, full attention every fourth layer, GatedDeltaNet
elsewhere, 96 experts with top-4 routing, Engram at layers 3 and 7, and
four mHC streams.

## Stored parameter count

The immutable API safetensors metadata reports:

| Stored dtype | Elements |
|---|---:|
| BF16 | 10,610,034,912 |
| I64 | 303,460 |
| **Distinct stored parameters** | **10,610,338,372** |

The index reports `total_size = 21,222,497,504` bytes, consistent with
`10,610,034,912 × 2 + 303,460 × 8`. Thus the card's “~10B” is a rounded
description; the immutable metadata contains 10.610B stored elements.

## Large-organ active ledger

This is a source-derived weight-count ledger for one autoregressive token,
not a traffic measurement. The derivation uses the config and the official
implementation's parameter shapes. Biases and small normalization terms are
excluded from the named-organ total unless explicitly included below.

| Organ | Derivation | Active weights/token |
|---|---|---:|
| Routed MoE | `24 layers × 4 selected experts × (2DF + FD)`, with `D=2048`, `F=512` | 301,989,888 |
| Shared MoE expert | `24 × 3DF` | 75,497,472 |
| Full attention | Six layers because `24 / full_attention_interval=4`; each has `D×4096 + D×512 + D×512 + 2048×D + 2×256` for gated Q, GQA K/V, O, and Q/K norms | 88,083,456 |
| GatedDeltaNet | 18 layers; each has `D×8192 + D×32 + 2048×2048 + 6144×4 + 16 + 16 + 128` for qkvz, ba, output, depthwise convolution, decay parameters, and gated norm | 379,112,256 |
| Engram | Two modules. Each uses `16×672` looked-up elements, `16×672×672` head projection, `2048×672` gate projection, `672×2048` output projection, and `672×4 + 672` convolution weights/bias | 19,983,936 |
| Output head | `151,680 × 2,048`; untied because `tie_word_embeddings=false` | 310,640,640 |
| Router | `24 × 96 × 2,048` | 4,718,592 |
| mHC | Per layer: `8192` norm + `(4+24)` static coefficients + `8192×28` dynamic alpha + `2` scales + `4` static beta + `8192×4` dynamic beta + `1` scale; plus final `2048×8192` stream projection | 23,266,120 |
| **Named large-organ total** | Sum of the rows above | **1,203,292,360** |

Important accounting assumptions:

1. The routed MoE counts exactly four selected experts, while the shared
   expert and router run on every layer.
2. Engram counts only the 16 rows actually indexed per module and assumes
   those lookup rows are the locality-relevant payload. It does not pretend
   that the entire static embedding table is read per token.
3. The head is charged as a complete vocabulary projection on every token;
   the input embedding lookup is not charged as a complete table.
4. The active ledger excludes small decoder RMSNorms, biases, the input
   embedding row, and other small tensors. It is therefore not a full
   parameter-count identity and is not a measured memory trace.

## Bit payload scenarios and 50 tok/s floor

For the named-organ ledger only, using 1, 0.5, and 0.25 bytes/weight for
8-, 4-, and 2-bit payloads:

| Scenario | W8 | W4 | W2 |
|---|---:|---:|---:|
| Named active payload, GB/token | 1.203292360 | 0.601646180 | 0.300823090 |
| Corresponding payload at 50 tok/s | 60.1646 GB/s | 30.0823 GB/s | 15.0412 GB/s |

For comparison, multiplying **all 10,610,338,372 stored elements** by the
same byte assumptions gives 10.6103, 5.3052, and 2.6526 GB/token at W8/W4/W2,
or 530.52, 265.26, and 132.63 GB/s at 50 tok/s. That is a **hypothetical
all-stored-weights-streamed-per-token scenario**, not necessarily an upper
bound on actual traffic: caching, reuse, gather amplification, metadata,
kernel behavior, and residency can move real traffic in either direction.

There is no measured Susono tok/s or bandwidth result here.

## Training and quality evidence

The card lists targets of 300B pre-training tokens and 250B mid-training
tokens, followed by SFT and DPO phases. Those are targets, not verified
completion evidence. At this revision I found no completed-token counter,
final-step record, training log, loss curve, or independent evidence that
the 300B+250B target was reached. The card itself says the hobby project was
not pretrained or post-trained to a sufficient extent and that this is an
untuned base model.

No quality benchmark table, held-out BPB, task result, or rate result is
provided by the primary card/config/index sources. Quality is therefore
**unknown**, not inferred from the A1B label.

## Custom-code risk and disposition

Custom-code risk is **high**. The card requires a dedicated Transformers fork
and `trust_remote_code=true`, and points to dedicated Megatron-LM, SGLang,
and vLLM forks. The custom implementation includes Engram, mHC-lite,
GatedDeltaNet, routing, and fused-kernel paths. The model revision does not
pin those external runtime repositories to matching immutable revisions.

Compared with the existing [STRAT-01 sparse/hybrid screen](STRAT_01_SPARSE_HYBRID_METADATA_SCREEN.md),
Susono stores more parameters than Granite-4.0-H-Tiny-Base (6.939B) and
LFM2.5-8B-A1B-Base (8.468B), but its named active ledger is lower at 1.203B
than their 1.465B and 1.686B ledgers. That is enough to merit a deeper
metadata/compatibility screen, not enough to authorize weights or a port.

The current STRAT-02E W2-BF16 result remains a relevant blocker: its tested
PTQ W2 arms were non-competitive versus W4. That result does not decide
Susono's QAT or architecture, but it rules out treating the arithmetic W2
floor as a quality claim.

**Final decision: metadata-only candidate. Not promoted to acquisition or
port.**
