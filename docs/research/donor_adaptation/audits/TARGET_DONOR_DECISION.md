# Target-donor disposition: Qwen2.5-Coder-14B and Qwen3-30B-A3B-Base

**Status:** frozen technical disposition, 2026-09-15.  This is a compatibility and
upper-bound audit, not a measured donor benchmark and not an authorization to download,
convert, train, or launch anything.

## Scope and evidence discipline

This note keeps five evidence classes separate:

| evidence class | meaning here |
|---|---|
| real donor | pretrained weights evaluated locally under a declared estimand |
| converted real donor | a converted pretrained artifact with parity/quality evidence |
| synthetic integrated timing | an engine timing with synthetic weights; useful for mechanics, never donor quality |
| derived arithmetic | exact weight-count/traffic calculation; an upper bound, never a rate prediction |
| external official metadata | model-card/config/implementation facts cited only as metadata sources |

The deliberately generous ceiling used below, `B = 41.389179011072 G weights/s`, is the
largest `charged_G_w_per_s` value in the canonical synthetic E40 result
(`benchmarks/donor_adaptation/engine/results/e40_levers_exhausted.json`, lines 261–272).
E40 used the packed path, not either donor below; carrying its best effective rate over to a
one-byte stream is intentionally favourable to the candidates.  Therefore the quotients below
are rejection upper bounds, not expected one-byte performance.

The local candidate inventory is [TARGET_DONOR_LOCAL_INVENTORY.md](TARGET_DONOR_LOCAL_INVENTORY.md).
Use the [cross-axis no-duplication map](AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md) before assigning
an experiment.  In particular, `Qwen/Qwen3-30B-A3B` and
`Qwen/Qwen3-30B-A3B-Base` are distinct weight identities even where their architecture/config
dimensions coincide.

## Qwen2.5-Coder-14B: closest dense exporter-family control, not a direct target

Official metadata for `Qwen/Qwen2.5-Coder-14B` gives Apache-2.0, `D=5120`, `F=13824`,
`L=48`, `H=40`, `KV=8`, `HD=128`, `V=152064`, untied embeddings/head, and Q/K/V biases.
The card and exact config are external metadata sources only:
[model card](https://huggingface.co/Qwen/Qwen2.5-Coder-14B) and
[config](https://huggingface.co/Qwen/Qwen2.5-Coder-14B/blob/main/config.json).

The linear-weight accounting is deliberately explicit:

```text
QO = H * HD = 40 * 128 = 5120
KVO = KV * HD = 8 * 128 = 1024
attention/layer = QO*D + 2*(KVO*D) + D*QO
                = 62,914,560
FFN/layer = 2*(F*D) + D*F = 212,336,640
attention + FFN/layer = 275,251,200
charged linear weights/token = L*275,251,200 + V*D
                             = 48*275,251,200 + 152,064*5,120
                             = 13,990,625,280
output head = V*D = 778,567,680
embedding = V*D = 778,567,680
per-layer non-matrix terms = (QO + 2*KVO) QKV biases + 2*D norms
                           = 7,168 + 10,240 = 17,408
logical total = L*(275,251,200 + 17,408) + embedding + head + final_norm
              = 14,770,033,664
```

At the deliberately generous ceiling `B = 41.389179 G weights/s`, a one-byte dense stream has
the weight-only ceiling `B / 13.990625280G = 2.958 tok/s`.  The untied head alone takes
`778,567,680 / B = 18.811 ms`.  Both are **upper bounds**, before nonweight work, not rate
predictions.  The head therefore consumes nearly the entire 20 ms budget for 50 tok/s by itself.

`qwen_export.py` is structurally closest for this dense Qwen family, but it currently assumes
Q/K/V biases and loads the complete model; see [qwen_export.py](../../../../benchmarks/donor_adaptation/engine/qwen_export.py)
around lines 404–437 and 565–640.  Frozen disposition: this is an exporter-family control,
**not a direct target**.  Do not download or real-benchmark it unless a head-and-body structural
transformation has independently passed both quality and executable-speed gates.

## Qwen3-30B-A3B-Base: architecture-interesting, not a direct speed candidate

Official metadata for `Qwen/Qwen3-30B-A3B-Base` gives `D=2048`, `L=48`, `H=32`, `KV=4`,
`HD=128`, `V=151936`, untied embeddings/head, `attention_bias=false`, 128 experts, top-8,
expert intermediate size 768, `norm_topk_prob=true`, and sparse MoE in every layer.  Metadata
sources only: [model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) and
[config](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json).

The exact charged linear-traffic calculation is:

```text
QO = H * HD = 32 * 128 = 4096
KVO = KV * HD = 4 * 128 = 512
attention/layer = QO*D + 2*(KVO*D) + D*QO
                = 18,874,368
router/layer = 128*D = 262,144
one selected expert/layer = 2*(768*D) + D*768 = 4,718,592
head = V*D = 311,164,928
fixed no-expert floor = L*(18,874,368 + 262,144) + head
                      = 1,229,717,504
eight-expert traffic = L*8*4,718,592 = 1,811,939,328
total charged linear weights/token = 1,229,717,504 + 1,811,939,328
                                   = 3,041,656,832
```

With the same generous `B = 41.389179 G weights/s`, the one-byte weight-only ceiling is
`B / 3.041656832G = 13.607 tok/s`.  Deleting **all** expert work still leaves only
`B / 1.229717504G = 33.657 tok/s`.  These are upper bounds, not predictions.  Consequently,
the fixed attention-plus-head traffic itself precludes 50 tok/s in a one-byte implementation;
attention and head must change too.

### Present semantic and format gaps

The current carved FFN is not an exact Qwen3 MoE implementation.  It computes router logits,
chooses top-k groups, runs selected SwiGLU rows, and performs an unweighted down accumulation;
see [donor_engine.c](../../../../benchmarks/donor_adaptation/engine/donor_engine.c) around
structs 133–164, loading 1057–1101, carved FFN 1211–1265, attention 1285–1304, and dense FFN
1431–1450.  The official Qwen3Moe implementation instead performs fp32 softmax, top-k,
renormalization when `norm_topk_prob=true`, multiplies every selected expert output by its
routing weight, then sums.  It also applies learned per-head-dimension `q_norm` and `k_norm`
before RoPE.  The upstream implementation is an external metadata/semantic source:
[modeling_qwen3_moe.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).

The remaining exact gaps are mechanical but material:

- Qwen3 uses `attention_bias=false`, while the current reader always reads Q/K/V bias vectors and
  `qwen_export.py` asserts that those biases exist.
- The exporter assumes `config.intermediate_size=6144` plus `lay.mlp.gate_proj`, `up_proj`, and
  `down_proj`.  Qwen3 sparse blocks instead provide a router and expert tensors with
  `moe_intermediate_size=768`; it cannot export this model unchanged.
- The engine already represents `QO != D`, RoPE theta, and GQA dimensions.  Those facts do not
  remove the routing-weight, q/k-norm, optional-bias, or expert-layout gaps.

## Non-duplication and deferred work

| item | frozen status | reason / reopening condition |
|---|---|---|
| Qwen2.5-Coder-14B download or real benchmark | **DO NOT RUN** | Dense one-byte head/body upper bound fails far below 50; first require an independently gated structural transformation. |
| Qwen3-30B-A3B-Base weight download | **DEFERRED** | Its fixed one-byte attention/head floor is below 50 even with every expert removed. |
| Qwen3 shape-only timing | **DO NOT RUN** | It would duplicate earlier synthetic pricing and cannot prove executable semantic compatibility. |
| Qwen3 exact parity/speed benchmark | **DEFERRED** | Only after exact router probabilities/weighted combine, q/k norms, bias-optional reader/exporter, and expert layout exist, then fp32 parity against a pinned pretrained revision; even then it needs a separately quality-gated head+attention cut before being rate-relevant. |
| H2I Phase A | **DO NOT RERUN** | Canonical matched Phase-A evidence already exists. |
| E66 | **DO NOT RERUN** | Canonical 7B one-byte fidelity/rank evidence already exists. |
| E43 | **DO NOT RERUN** | Its speed half is permanently void. |
| E40 | **DO NOT RERUN** | Synthetic attention-lever exploration is closed; it is not donor quality evidence. |

The distinct roles of E40/E43/E60/E63/E66 remain as recorded in the canonical
[INDEX](../INDEX.md), [speed ledger](../SPEED_LEDGER.md), results, and probes: none substitutes
for a real-donor parity or rate result.  Synthetic integrated timing, derived arithmetic, and
converted-real-donor evidence must never be promoted across those classes.

## Current dependency gate

H1 S3 has now passed its frozen terminal adjudication: ternary eight-layer joint training reaches
`CARVE-IS-TRAINABLE` at HARD BPB 0.9234896239116439, versus applied-8L 1.0966361325809948. This
strengthens the plausibility of trained structure, but it does **not** authorize a 14B or 30B
download: H1 is ternary, eight-layer, and has no rank or rate measurement.

Await H2I Phase B v4's frozen **one-byte score+rank** adjudication. If it survives both gates,
that result decides whether the structural-adaptation mechanism is credible before scaling or
downloading. If it fails, do not scale the same mechanism. This is a dependency statement only:
it recommends no launch now.
