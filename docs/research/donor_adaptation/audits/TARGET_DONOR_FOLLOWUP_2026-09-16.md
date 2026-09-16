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

**2026-09-16 format correction:** The clean [E63 Part B result](../../../../benchmarks/donor_adaptation/engine/results/e63_part_b.json)
measured `49.37 tok/s` on **synthetic mixed-format A10B**, not a full one-byte stream or a
pretrained model. Its `928,251,904` charged weights/token include only `106,168,320` carved
FFN weights in int8; the other `822,083,584` stay packed at half a byte. Thus the mixed
stream reads `517,210,112` charged weight bytes/token, versus `464,125,952` in the
packed arm. The median implies `25.535 GB/s` for this particular mixed stream. Dividing
its total *weight count* by time gives `45.828 Gweights/s`, but that mixes formats and is
**not a transferable full-int8 rate**. The former `0.916603 Gweights/token` donor budget
and the full-byte rate predictions below were invalid and are withdrawn. E40's
`41.389179 Gweights/s` is likewise a packed-path observation, not a cross-format ceiling.

## Comparable full one-byte streams

These counts assume `QO = heads × head_dim`, `KVO = KV_heads × head_dim`, SwiGLU experts with
three matrices, all listed active experts evaluated, one byte per linear weight, and a full
output-head multiply at every token. They exclude small norms/biases and nonweight work; tied
embeddings remove a second *stored* matrix, not the output-head read. Any calculation that
omits routed weighting or other semantics is only traffic arithmetic, not an implementation.

| Publisher checkpoint | Charged attention | selected + shared experts | router | head | total charged/token | full-int8 payload GB/s needed at 50 |
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

The final column is simply charged weights/token × 50 bytes/s under a full-int8 assumption;
it is **not measured achievable bandwidth**. E63 cannot predict these donors' full-int8
rates because 88.6% of its charged weights stayed packed.
OLMoE and IBM also fall below the approximate 10B total scale; the Ai2 14B checkpoint is
nearer in total size but has a larger one-byte traffic burden. Both Ai2 models are genuine
pretrained weights; IBM's public card is insufficient to promote quality.

## Other newly surfaced branch

[Meta MobileMoE-L](https://huggingface.co/facebook/MobileMoE-L-Base) is listed by its official
card as 5.3B total / 922M active, top-4 with 60 routed experts, 32 layers, `D=1280`, tied
embeddings, and QK norm. The active count does not establish charged bytes/token or
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

### Addendum 2026-09-16 — exact StdMoE revision and a mixed-precision gate

The read-only [publisher API record](https://huggingface.co/api/models/allenai/StdMoE_1b14b_1T_Preanneal?blobs=true) currently resolves to revision
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`: `safetensors.total =
13,568,641,024` distinct stored parameters, all `F32`, in **11 shards totaling
54,275,336,216 bytes**. The card declares Apache-2.0 and a 1T-token pretrained
standard-MoE checkpoint. This exact total replaces the card's rounded “14B” for
memory planning; it is not an active-parameter count. The pinned
[config](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal/blob/d2a4949c9d4ad6cf47fbac131f7e020077332b21/config.json)
has `D=2048`, `L=16`, `E=128`, `k=8`, `num_shared_experts=1`, `F=1024`,
`V=100352`, untied embeddings, full MHA and 4096 maximum positions. The
roadmap's proposed 8K/32K context checks would therefore require a separately
validated context-extension change; the native checkpoint alone does not meet
them. The pinned
[model implementation](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal/blob/d2a4949c9d4ad6cf47fbac131f7e020077332b21/modeling_emo.py)
selects `top_k - num_shared_experts` routed experts and then the shared expert:
**seven routed plus one shared, eight FFN evaluations total**. This resolves the
asterisked ambiguity in the earlier table. Because `always_active_experts=null`,
the pinned implementation uses its legacy shared path: it applies **separate
softmaxes** to 127 routed logits and the single shared logit. The shared
coefficient is therefore exactly `1`, while the seven routed coefficients
are their probabilities in the full routed-127 softmax, **without top-7
renormalization** (`norm_topk_prob=false`). A port that softmaxes all 128
logits together or renormalizes the selected seven changes the model. Its
`1,283,457,024` charged large
weights/token still excludes scales, metadata, KV/cache and nonweight work.

The arithmetic for a *hypothetical* W4 always-active stream and W2 expert stream
is: attention + router + head = `478,150,656` weights/token at 0.5 B each;
selected experts = `805,306,368` weights/token at 0.25 B each; total payload
`440,401,920 B/token`. All-W4 would be `641,728,512 B/token`; all-W2
`320,864,256 B/token`. The mixed payload alone needs **31.46 GB/s to fit the
roadmap's 14 ms streaming allowance**, or **22.02 GB/s to fit the entire 20 ms
50-tok/s period** with *zero* time left for everything else. At assumed
40/28 GB/s it consumes 11.01/15.73 ms. These are conditional divisions, not
measured bandwidth on this donor, a W2 quality result or an engine forecast.
E63's synthetic mixed stream has a different format/layout/shape and cannot
supply that missing rate.
The config-derived *stored* large-organ cross-check is 12,884,901,888 expert
weights + 268,435,456 attention + 4,194,304 router + 205,520,896 head +
205,520,896 distinct input embedding = 13,568,573,440, just 67,584 below
the publisher safetensors total (small norms/biases). If the input embedding
stays F32, the theoretical mixed packed storage payload is 4,282,384,384 B
plus scales/metadata/padding. The embedding is **stored** but only one row is
looked up per token; the distinct untied head is charged each token. Packed
storage is not training-memory usage or an implemented loader.
The pinned [safetensors index](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal/blob/d2a4949c9d4ad6cf47fbac131f7e020077332b21/model.safetensors.index.json)
independently lists 6,259 tensors over 11 shards: 6,144 expert matrices
(`16×128×3`), 64 attention matrices, 16 routers, separate head/embedding
and 33 norm tensors. Its `metadata.total_size=54,274,564,096 B` is exactly
`4×13,568,641,024`; the shard files total 772,120 B more from container
overhead. This confirms the stored-parameter arithmetic without touching
weight payloads.

On the local host the read-only check showed 65,483,676 KiB (~67.06 GB)
free physical RAM and
~921.03 GB free on `D:` at screening time. The F32 checkpoint therefore fits
on disk, but a naive full-F32 Python load would leave only ~12.78 GB of then-free
RAM before runtime, temporary copies and activations. The next R1 quality probe
needs a bounded-memory streaming/quantized loading plan and an exact teacher
baseline; it must not be launched with an unchecked `from_pretrained` allocation.
No weights were downloaded; the pinned remote model code was inspected as
text, not saved to the worktree or executed.
The exact HTTP bytes at this revision have SHA-256
`f1bd419d8dd926cf7d15131b8192938a0feda1b003586320fa56de20360e4bf2`
for `configuration_emo.py` (11,686 B) and
`26f57354940655db673b35d47e9c6b1a85900068f41d91cb08afed4ec53219d6`
for `modeling_emo.py` (55,477 B); these are pre-execution identity checks,
not a security audit. The modeling file decorates RMSNorm with
`use_kernel_forward_from_hub`; the local Transformers integration honors
`USE_HUB_KERNELS=NO`, which must be set before the first isolated import so
the teacher path does not silently fetch an unpinned kernel.
The current local `.venv` reports `torch 2.6.0+cu124`, `transformers 5.13.1`,
`safetensors 0.8.0` and `huggingface_hub 1.16.4`, while the pinned config
records `transformers_version=4.57.1`. A **local-import-only** compatibility
check (no remote code executed) tried the symbols imported by the pinned
`modeling_emo.py` and failed: `ImportError: cannot import name
'OutputRecorder' from 'transformers.utils.generic'` in the local 5.13.1
environment. Thus this environment cannot run the publisher's pinned model
code as written; this is an operational preflight failure, not a model-quality
result. Use an isolated, pinned compatible Transformers environment or an
explicitly reviewed/validated compatibility adaptation, then audit/import
the exact remote code **before** any 54 GB weight download or teacher score.

**Isolated import preflight completed, still without weights.** A temporary
package target outside the repository installed `transformers==4.57.1` and
its compatible dependencies while reusing the existing Python 3.12.10 and
Torch 2.6.0. With `USE_HUB_KERNELS=NO` set before import,
`AutoConfig.from_pretrained(..., revision=d2a4949..., trust_remote_code=True)`
reported `model_type=emo`, `E=128`, `k=8`, one shared expert; importing
`modeling_emo.EmoForCausalLM` succeeded. The files **actually imported**
matched the SHA-256 values above byte-for-byte. The default HF snapshot
contained only `config.json`, `configuration_emo.py`, and `modeling_emo.py`,
no safetensors shards. A second offline, no-weights check instantiated
`EmoForCausalLM(config)` entirely on PyTorch's `meta` device: it exposed
exactly **13,568,641,024 parameters across 6,259 meta tensors**, matching
both the publisher API and the pinned safetensors index, without allocating
weight storage. This closes Python import and model-shape consistency only;
forward/oracle parity, dependency security review, bounded-memory
teacher loading, quality and rate remain unmeasured.

**Decision update:** this candidate's *stored* scale exceeds the
illustrative 10B target, and its **mixed**
payload has nonzero idealized headroom at 40 GB/s under the 14 ms allowance.
It is promoted only to a **single-donor precision/teacher-baseline brief**, not
to download, T4 training, C port, quality or speed PASS. That brief must pin the
revision and remote-code hash, establish same-tokenizer teacher BPB/rollout,
test the actual W4/W2 operator by organ on held-out data, and specify a
bounded-memory loader plus a 20 ms full-token budget. If W2 experts or the
necessary bandwidth fail, this R1 branch stops or changes geometry explicitly.

No newly screened pretrained donor simultaneously clears the exact-geometry, full-stream
one-byte traffic, current-engine semantics, and practical-quality gates on available
evidence. **Do not spend T4 time porting or downloading these weights on this screen alone.**
This is a *prioritization*, not proof that 50 tok/s is impossible. Crucially, **none of
these full-int8 candidates has a comparable measured bandwidth or speed limit** here;
the earlier E63-based rate rejection was invalid. An actual port could behave differently
in either direction.

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
