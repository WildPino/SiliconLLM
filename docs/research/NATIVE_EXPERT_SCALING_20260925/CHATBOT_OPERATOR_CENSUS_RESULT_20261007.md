# Entire chatbot operator accounting and transfer decision

7 October2026. Goal ACTIVE/INCOMPLETE. This is a new implemented analysis stage
of pretrained CHATBOT -> compact conversion -> export -> engine -> own-history
quality and SAME-artifact accepted50. No new model inference or weights read.

## Result and practical decision

Exact names/shapes/dtypes and independent element/byte/MAC conservation close
for both actual source headers and the existing Qwen276 C archive. The complete
cost surface explains why more residual experts alone cannot finish the goal:

| Batch1 decode, full head | Source Qwen | Actual Qwen276/295/296 | Source Giga, main only |
| --- | ---: | ---: | ---: |
| Stored header named elements, including buffers/copies |494032768|886086401|11479750784, including MTP|
| Stored tensor payload bytes |988065536|1329221892|22959501568, including MTP|
| Matrix coefficient-product terms/token |493961216|508086272|1628078080|
| Logical coefficient byte budget/token |988067328|709514884|3256351872|
| Direct attention products per context token |43008|43008|319488|
| Full vocabulary head products/token |136134656|136134656|197001216|
| Full head coefficient bytes/token |272269312|272269312|394002432|

These are dimension/code deductions, not measured CPU instructions, DRAM traffic
or throughput. MAC convention counts one matrix coefficient-product term as
one MAC, including integer and float paths; differing kernels need actual
clocks. Nonlinear/reduction/topK/quantization/control work remains extra.
Logical bytes include a duplicate embedding-row read of the tied head and an
explicit LUT/alias upper budget. They cannot be used as physical transactions.

Qwen's current archive decreases this coefficient byte budget by28.19%, while
increasing matrix terms by2.86%. It keeps ALL4864 FFN features/layer, then
recomputes128 private BF16 gate/up rows and adds residual32 and conditional
functions. This is reusable complete export/import machinery, with additional
selective capacity, but its always-active source FFN has not been replaced.
The Q8/F16 head proposal accounts for136438528 loaded bytes and zero actual
head work: the295/296 profile still reads the full tied BF16 embedding.

Giga's entire attention projections cost650051584 products/token (39.93% of
the matrix total); dense+shared+selected routed FFNs cost778567680. Head costs
197001216 and source routing2457600. Reducing only routed FFN width leaves a
large attention/head floor. This agrees with the retained301–304 native cost
attribution; it does not replay or supersede those clocks.

**Decision:** use small Qwen as the FIRST complete compact transfer case.
Replace the active source FFN with a jointly adapted shared plus conditional
SwiGLU representation, while pricing head and attention from the start. Keep
Giga as the second family with explicit MLA/core differences. Do not resume
the old independent-parent/shared-output-space recipe or donor runtime port.
The proposed replacement is untrained and has no capacity/quality admission.
First bind canonical chat fixtures, then select a finite complete converter
trial; avoid more precision, affinity or selection-only ladders.

## Source conservation and family contracts

Qwen:290 tensor names, tied embedding/head136134656 elements; attention44040192,
dense gate/up/down313786368, norms43904 and QKV biases27648. FFN gate/up/down
are bias-free SiLU; attention Q/K/V biases are present, O bias absent. There is
no pretrained expert router to preserve: the added parent/child router is part
of the adaptation and needs its own useful-capacity qualification.

Giga:5323 names bind exactly. Main-generation named elements10672535616;
additional MTP807215168. The latter includes separate embedding/head copies,
projection/norms and a full shared/routed layer. Header accounting neither
proves those copies numerically distinct nor credits MTP with free accepted
tokens. Main generation excludes the MTP path; speculation is a separate gate.

Main Giga routed banks store9437184000 elements, but top4/64 consult589824000
per token; shared FFNs147456000 and first dense layer41287680 remain active.
Per-expert K/E fractions in the descriptor ledger distribute a bank TOTAL,
not a uniform-visit assumption or individual-tensor access upper bound. All
config-based formulas independently agree with the enumerated operator terms.

Giga routing reads all64 logits per MoE layer. Choice uses sigmoid plus the
correction buffer and grouped topK. Mass uses the ORIGINAL four selected
sigmoid scores divided by their selected sum+1e-20, then routed scale. A full
E softmax denominator is absent. Winner/search still grows with total E for
this source algorithm. Target n and useful additional functions are unproven.

Existing Qwen router uses8x16 factor axes/top4 each,16 candidate products/top4,
ten child keys per selected parent and one child argmax. Its mass is over the
four selected parent scores, BF16-rounded. This is not a qualified million-n
LUT, full128-parent softmax or normalized1280-child mass contract. The fixed
catalog has30556 distinct stored leaf slots according to retained276 evidence;
numeric uniqueness was not rechecked from values in this census.

## Cache and context, with actual backend boundary

Qwen source code stores GQA K/V before head repetition:6144 elements per cached
token across24 layers. Source cache dtype is unobserved here; two-byte context
2048 would use25165824 bytes, four-byte50331648. The actual C profile allocates
float32 arrays,100663296 bytes at maxseq4096. BF16 rounding does not make those
arrays two-byte storage. Dynamic direct attention adds43008*context products.

The bound local DeepSeek-v3 code expands MLA K/V before `Cache.update`:
319488 elements per cached token across26 layers. At context2048, two-byte
cache would be1308622848 bytes; at8192,5234491392. Actual source inference dtype
and allocation are not measured. Dynamic direct attention adds319488*context
products. This corrects the overly broad "compressed KV cache" requirement in
the previous source preflight: keep its original report bytes and qualify the
backend explicitly. MLA architecture alone does not establish compressed cache.

A latent MLA backend could retain14976 elements/cached token for this config.
That is an UNIMPLEMENTED alternative needing absorbed projections, changed
attention arithmetic, cache/source parity and physical allocation/cost evidence.
Smaller cache does not by itself prove lower compute or faster accepted decode.

## Algebra and information: the next representation

For these bias-free SwiGLU channels,

    f(x) = sum_j d_j SiLU(g_j*x) (u_j*x).

Partitioning channels is exact only if the omitted sum is reconstructed.
Overlap/redundancy can store different useful views without proportionally
increasing selected work; copying addresses does not supply such a reconstruction.
Both g_j*x and u_j*x carry input information. Router scores alone need not
determine them; retain full x as the conditional function input initially.

SiLU(z)-z=SiLU(-z) gives a quadratic core. A raw output tensor Q_d is

    Q_d = sum_j d_dj sym(g_j u_j^T),  f_d(x)=x^T Q_d x + residual_d(x).

Materializing D different DxD matrices uses D^3 coefficients/layer. Keeping
all original factored gate/up terms preserves their active source-sized cost.
A genuinely cheap quadratic route therefore needs a new shared factorization
or regional approximation with explicit residual information. No exact cheap
linear ReLU fold applies. Original528's29% rare-domain failure closes only that
single-anchor ReLU recipe, not arbitrary redundant conditional representations.

One concrete target family for a NEW trial is

    shared_s(x) + sum_(k in selected(x)) w_k(x) B_k[SiLU(G_k x) * (U_k x)].

With shared width h0, selected K, conditional width h1 and hidden D, its FFN
matrix budget is3D(h0+K*h1) per layer plus router work, when shared_s is SwiGLU.
Storage grows with distinct stored G/U/B functions, while selected work stays
fixed. Sharing G/U inside a parent and using different leaf B functions is a
separate representation variant whose output span/exposure must be checked.
Head and attention still contribute the measured dimension floor above.

This differs from310/311/316: those fixed independent parent/region output
spaces failed; complete Giga mixture TEST medians53.79%/66.77%/21.56% and joint
span oracles46.36%/59.08%/18.11% remain closed. Jointly learned nonlinear input
functions and shared core are not licensed by those bounds, and also not
generally refuted. Do not repeat PCA16/spherical10/shared256/regional128 fields.
Existing Qwen128->1280 utility can guide initialization, but cannot prove a
new compact base FFN retains the donor's knowledge.

If a head approximation Hhat is proposed, its cost must include every produced
logit/normalizer. At fixed hidden state, infinity logit error delta protects
the winner only when2delta is below the donor margin; log-partition error is
at most delta, and target negative-log-probability error at most2delta. Hidden
state perturbations and changed own-history choices are separate errors. A
low-rank head or a smaller local RMS cannot substitute for fresh dialogue/task
quality and accepted end-to-end speed on the same exported artifact.

## Provenance, apparatus and retained limits

Executable [census](../../../benchmarks/native_expert_scaling/chatbot_operator_census.py),
[protocol](CHATBOT_OPERATOR_CENSUS_PROTOCOL_20261007.md),
[binding](chatbot_operator_binding_20261007.json), freezea23514a45b1698063ce567b5057410b38ab0d83b.
Header-only source SHA/length provenance; the original full source payloads
were not rehashed. Existing archive725 descriptors match all fixed C catalog
names/types/shapes/absolute offsets; prior full archive SHA4f9b9c7a... is reused.

| Job | Retained output SHA256 | Actual executor | Reported duration/OS peak snapshot |
| --- | --- | --- | --- |
| Qwen source+archive |a4293870cb38706fc60d94562d9579653ddff1a1387c366c307a56b4b262bbc2|0b8d35 exit0,0.9792052s span|.172s /28614656B|
| First Giga, output-cap fault |e8f3a18733a6fc6ec7ddae3a39b8cd6415b0142ef3daa3ba726b78527a0b79c4|70948d exit1,.8896455s span|UNKNOWN Python instance/peak; no retained counts|
| Giga compact representation repair1 |94e8a43db9b64d691ff9531b3cfd8eeb40b8062e1eb24f4198a11c0b5ba5236a|13ab96 exit0,1.3881841s span|.266s /34582528B|

First Giga reached report serialization; the pretty JSON exceeded2MiB and its
failure serializer also failed. The first actor/resource/count report was
lost. Preserve [fault](chatbot_gigachat_operator_census_20261007.failure.json).
Repair1 changes ONLY JSON representation, preserving all keys/counts and cap:
1630025-byte report, [source](../../../benchmarks/native_expert_scaling/chatbot_operator_census_repair1.py),
[repair protocol](CHATBOT_OPERATOR_CENSUS_REPAIR1_20261007.md),
[binding](chatbot_operator_binding_repair1_20261007.json), freeze53ed04bba05020cb957e0ebf4eca90a2f1f7a0a4.
The FAILED Giga metadata computation is repeated because no first counts were
retained. Completed Qwen and all source/model/native experiments are not replayed.

Resource snapshots are taken before final JSON serialization. Final whole-job
OS peaks were not retained, so full peak qualification is UNKNOWN; executor
wall spans bound elapsed duration, not RAM. Arithmetic/provenance conservation
is established; original120s/256MiB whole-process protocol admission must not
be inferred from the partial peak snapshots. Future launchers must retain an
OS process handle/peak through exit; no scientific replay to fill this gap.

[Windows closure](chatbot_operator_windows_terminal_20261007.json), d97f0f/exit0:
known instances6192/create1791376151.987146 and28000/create1791376352.5080044
closed, typed UTC Event1000 queries available/zero relevant faults, positive
controls179810/179791. First failed Giga instance remains UNKNOWN. Zero tensor
values/model/native calls, foreign hashes and isolated empty cache preserved.
No conversion/quality/50/s/general-family/large-n/physical-DRAM admission.

## Exact next action

[Canonical interaction and finite compact transfer plan](CHATBOT_CANONICAL_COMPACT_TRANSFER_NEXT_20261007.md).
Use these existing ledgers; do not rerun this census, old native295/296/297,
528 or closed independent output-space/cost recipes. Goal remains ACTIVE.
