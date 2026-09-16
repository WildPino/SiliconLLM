# Pretrained sparse-donor screen after E63 and H4

**Status:** metadata plus derived arithmetic only, 2026-09-16. No weight download, conversion,
donor-quality test, or donor-speed test. This supplements, and does not silently rewrite, the
[frozen target-donor disposition](TARGET_DONOR_DECISION.md).

## Decision question

Can an already-pretrained, approximately 10B-total-parameter donor avoid a costly joint
attention/head/FFN transformation and directly enter the 50 tok/s one-byte budget on the
current `engine.c`? This is a candidate *screen*, not a donor result. "Active parameters" on
a model card is not the same as charged linear weights/token: the head is charged every token,
an embedding lookup is not, and routing/shared experts need explicit accounting.

The clean [E63 Part B result](../../../../benchmarks/donor_adaptation/engine/results/e63_part_b.json)
measured an **A10B synthetic** one-byte median of `49.37 tok/s`, not a pretrained model.
Using E36's `0.9283 G` charged weights/token, the descriptive effective throughput is
`49.37 × 0.9283 = 45.830171 G charged weights/s`. At that *observed* rate, a 50 tok/s model
would need at most `0.916603 G` charged linear weights/token. This is a screening reference,
**not** a universal speed ceiling or an admissible rate prediction for a new architecture.
It also corrects the older audit's E40 `41.389179 G weights/s` "upper-bound" label: E63
already exceeds it, because the path and format differ.

## Comparable full one-byte streams

These counts assume `QO = heads × head_dim`, `KVO = KV_heads × head_dim`, SwiGLU experts with
three matrices, all listed active experts evaluated, one byte per linear weight, and a full
output-head multiply at every token. They exclude small norms/biases and nonweight work; tied
embeddings remove a second *stored* matrix, not the output-head read. Any calculation that
omits routed weighting or other semantics is only traffic arithmetic, not an implementation.

| Publisher checkpoint | Charged attention | selected + shared experts | router | head | total charged/token | effective Gweights/s needed at 50 |
|---|---:|---:|---:|---:|---:|---:|
| [Ai2 OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125), [config](https://huggingface.co/allenai/OLMoE-1B-7B-0125/blob/main/config.json) | 268,435,456 | 805,306,368 | 2,097,152 | 103,022,592 | **1,178,861,568** | **58.943** |
| [Ai2 StdMoE_1b14b_1T_Preanneal](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal), [config](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal/blob/main/config.json) | 268,435,456 | 805,306,368* | 4,194,304 | 205,520,896 | **1,283,457,024*** | **64.173*** |
| [IBM experimental shared-expert checkpoint](https://huggingface.co/ibm-research/moe-7b-1b-active-shared-experts), [config](https://huggingface.co/ibm-research/moe-7b-1b-active-shared-experts/blob/main/config.json) | 251,658,240 | 566,231,040 + 188,743,680 | 3,809,280 | 77,194,752 | **1,087,636,992** | **54.382** |

Arithmetic, with `D` hidden size, `F` intermediate size, `L` layers, `k` active experts, `E`
router width, `V` vocabulary size:

```text
attention = L × D × (2×QO + 2×KVO)
experts   = L × k × 3 × D × F
router    = L × E × D
head      = V × D
IBM shared expert = L × 3 × D × shared_intermediate_size
```

OLMoE: `D=2048, L=16, QO=KVO=2048, F=1024, k=8, E=64, V=50304`, untied head.
Ai2 StdMoE: same `D,L,QO,KVO,F,k`, `E=128, V=100352`, untied head. Its card calls this
14B-total/1B-active and describes 127 routed plus one shared expert, eight active total.
`*` The table counts **eight expert evaluations including the shared expert** at the same
intermediate width. Confirm exact shared/routed combine in its custom implementation before
using this as a final artifact specification; if "top-8" instead means eight routed *plus*
one shared, add `100,663,296` weights/token, worsening the screen.
IBM: `D=1536, L=40, QO=1536, KVO=512, F=512, k=6, E=62, V=50257`, tied head, plus
`shared_intermediate_size=1024`. Its card says only "test model"; no practical-quality claim
is licensed by that metadata. The IBM shared expert is charged on every layer.

At the E63 descriptive `45.830171 G weights/s`, those totals correspond to **38.88, 35.71,
and 42.14 tok/s respectively** *if* the entire different model reproduced that effective
rate and incurred no additional costs. They are **not measured or bounded donor rates**.
OLMoE and IBM also fall below the approximate 10B total scale; the Ai2 14B checkpoint is
nearer in total size but has a larger one-byte traffic burden. Both Ai2 models are genuine
pretrained weights; IBM's public card is insufficient to promote quality.

## Other newly surfaced branch

[Meta MobileMoE-L](https://huggingface.co/facebook/MobileMoE-L-Base) is listed by its official
card as 5.3B total / 922M active, top-4 with 60 routed experts, 32 layers, `D=1280`, tied
embeddings, and QK norm. This is the only screened official card with an active count near
the descriptive one-byte budget, but the count does not establish charged bytes/token or
50 tok/s. Its weights are gated, licensed for noncommercial research, and its 5.3B total
size is substantially short of the approximate 10B target. It is at most a **smaller proof
of mechanism** candidate after exact config/access and engine-semantic review, not the final
target donor. No access request or weight download has been made.

A [community 10B/1B base claim](https://huggingface.co/puwaer/Susono-10B-A1B-Base) has the
right headline scale but explicitly says pretraining is insufficient and uses a hybrid
Full Attention + GatedDeltaNet backbone with Engram/mHC components. It is neither quality
evidence nor a drop-in `engine.c` donor. The source is the publisher's own card, not an
independent evaluation.

## Disposition / non-duplication

No newly screened pretrained donor simultaneously clears the exact-geometry, full-stream
one-byte traffic, current-engine semantics, and practical-quality gates on available
evidence. **Do not spend T4 time porting or downloading these weights on this screen alone.**
This is a *prioritization*, not proof that 50 tok/s is impossible. In particular, OLMoE
would need `58.943/45.830 = 1.286×` E63's descriptive effective weight throughput for its
full one-byte stream; an actual port could behave differently in either direction.

The shortest nonduplicative path remains to define a joint *deployable* geometry, evaluate
its quality with an exact teacher/reference and fixed gates, then measure that same geometry
in `engine.c`. H4 establishes that q/o adaptation learns but ends at a generation floor;
its fp32-latent diagnostic did not rescue that floor. E63 establishes near-50 on noise
weights, not donor usefulness. Neither result can be substituted for the missing joint proof.

Reopen a specific donor only if (1) an exact charged-stream calculation and engine-semantic
gap list are complete, (2) a preregistered smaller joint quality test gives a plausible
path to useful text, and (3) the expected speed gain pays for conversion/parity work.
Before any new full model download, freeze its revision, license/access, oracle parity
protocol, and go/no-go gates. Do not rerun E40, E63, H2I Phase A, or H4 terminal cells.
