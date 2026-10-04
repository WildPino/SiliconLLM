# M411: actual Granite3.1 MoE headers and one complete I8 applicability ledger

Freeze controller/protocol BEFORE tensor-header acquisition/outcomes.
New uncertainty after410 joint Ling format cost rejection: can another pretrained
family offer a smaller mandatory active path under the explicit conservative
byte allowance, with an original operator contract compatible with a credible
complete transfer? This does not reduce the final real~10B/~100B/useful RAM-scale
capacity goal to a small model. Existing Switch7.415B/14.664B qualified sources
remain; a tractable second-family case tests procedure generality first.

410 retained84e7610, raw bd345a1b3239c5f319432541059006fb94cb6a002593e52d80890a0058707f87:
new256+16/static32KiB-pair/Q4 passes ALL8 numeric/complete-byte gates, fails14ms,
ALL18 rep medians>20ms. Do not acquire Ling/fitting under its rejected formats.
GigaChat full-width LUT/latent and StdMoE unchanged W4 cost/quality closures remain.
Granite4H-Tiny6.939B is a different hybrid architecture whose913MB active Q4
header screen already failed; this is NOT an unchanged repetition of that source.

## Fixed sources and prospective evidence

Two original IBM Granite3.1 base MoE sources, chosen from official configurations;
API/model listing consulted preparatorily to pin revisions and two-shard layout,
no safetensors headers or values observed before freeze:

- ibm-granite/granite-3.1-1b-a400m-base,
  revision408b6e90baab8cf24f4aa9f8e19703ffa0a53b29:
  D1024/FF512/24layers/32 experts/top8/heads16/KV8/vocab49152.
- ibm-granite/granite-3.1-3b-a800m-base,
  revisiond4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a:
  D1536/FF512/32layers/40 experts/top8/heads24/KV8/vocab49152.

Both declared SiLU gated packed experts, biasless GQA full causal RoPE,
tied embedding/head, norm epsilon1e-6, attention multiplier.015625,
embedding multiplier12/residual.22/logits divisor6, BOS/EOS/PAD0,
context131072; respective RoPE theta1500000/10000000. Original card throughput/
benchmark scores are NOT host/native quality/rate proof. Configs remain actual
shape hypotheses to verify against ALL source headers. Main config operator
multipliers must be retained later; no automatic Switch loader compatibility.

Hash/store, NEVER import/execute, installed official Transformers4.57.6 module:
modeling_granitemoe.py SHA6bd8829f07902a061168d0f600774b95fca9382aa9e102a6c1def37f95b614ed;
configuration_granitemoe.py SHA6d930c9775bc203cfd55f6cdf7c637f4ad762bb0b6326d03d7ee35be7832c5f5.
This is code inventory only. Original runtime/backend/cache/generation reference
must later be independently qualified; package/module presence is insufficient.
Code inspected: expert input[num_experts,2FF,D]/output[num_experts,D,FF],
top8 RAW logits then selected-logit softmax (no full-bank probability denominator),
SiLU first half times second, sorted expert accumulation. This can alter which
router approximation is eligible versus Switch, but no retrieval is implemented.

## Actual acquisition and analysis

Reuse407 strict requests/range/dtype/offset/EOF logic, SHA-bound helpers and
previous410 record. Read pinned API blobs=true, config/index/generation config/
complete tokenizer files/special tokens for both sources; bind all small assets
to immutable Git blob identities. Require exactly two declared safetensors
shards each, stored-index ALL names equal ALL actual headers. Each response
requests ONLY prefix0..7 then declared header8..7+length. Enforce206/exact
Content-Range/total/identity encoding; no weight-value byte requests. The
whole-shard SHA is LFS DECLARED, cannot be freshly verified without values.

Every dtype/shape/product/offset/contiguous extent/EOF/index count must reconcile.
Classify ALL tensor names, no silent leftovers: per-layer four attention matrices,
two norms, router.layer.weight, packed input_linear/output_linear, embedding/
final norm and optional serialized lm_head tied alias. ALL actual tensors BF16,
ALL24x32 and32x40 expert slots have exact declared packed shape. No uniqueness/
ordered byte/function/usefulness inferred from header dimensions or slots.
If lm_head present, named counts include it; architectural unique count subtracts
one head CONDITIONALLY on declared tying, actual value equality remains unverified.
If absent, original class declared tie uses embedding as head; actual finite/
original outputs still unqualified. No MTP/extra tensors dropped to fit ledger.

Compare complete tokenizer bytes across sources; this is tokenizer identity,
NOT same hidden space, causal expert-count comparison or transferable functions.
Store ALL API/header/assets/module snapshots and raw config/tensor ledger.

## ONE explicitly hypothetical target, no performance or quality promotion

Row-I8 absmax127/F32 row scales and A16/I64-chunked accumulation reuse the
qualified Switch arithmetic recipe where applicable; all attention/expert/head
rows retain full width. F32 router/norm controls, BF16 lookup. Tied original
embedding/head counted once as SOURCE knowledge; TARGET stores BF16 lookup
and I8/F32-scale head separately, both storage copies charged. Source error/head
precision and whole composition unverified. No codebook fitting/reduced expert
count/duplicated experts/synthetic teacher or implicit inherited quality.

Count actual stored/selected expert coefficients and complete attention/head/
router/norm/lookup/scales separately. Include all full head rows, one lookup
row (not whole embedding/token). Report full target storage, original BF16/F32
value/RAM extents and nominal fit, without assumed allocator/KV/cache capacity.
Conservative active descriptor<=560000000B licenses ONLY a new complete CPU-cost
and source-specific original operator/reference protocol. Above560MB closes THIS
unchanged row-I8 geometry before values/native implementation. A positive ledger
never licenses generic incompatible-cost porting or proves50 accepted IDs/s.

## Gates, stop and retained output

ALL small Git blobs/indices bound, ALL4 shard headers/dtypes/offsets/extents/index
exact, ALL expert shapes/parameter/active/stored ledgers reconciled, tokenizer
identity and reference-module hashes stored/not executed. A failure stops and
retains FIRST raw outcome before any new numbered procedural repair.
Interpret each source's byte eligibility separately; neither512/256/10x expert
scaling nor larger donor source knowledge inherited. Two scales have different
core/geometry/training, so no causal n conclusion.

MAIN<=600s/checkedRSS<=1GiB/response body<=64MiB/<250 HTTP requests,
individual connect/read15/30s, header<=4MiB/shard, small bodies bounded. Expected
~30s/~100MB; CPU/local reference-file copy plus public metadata network only,
ZERO source values/model jobs/GPU/T4/training. No timed inference overlap.
No engine changes in411. Existing default0ff9705 tail and qualified binaries intact.

Primary preparatory sources:
[IBM1B configuration](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/raw/408b6e90baab8cf24f4aa9f8e19703ffa0a53b29/config.json),
[IBM3B configuration](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-base/raw/d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a/config.json).

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth411_granite_moe_source_headers.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth411_granite_moe_source_headers_result.json
```
