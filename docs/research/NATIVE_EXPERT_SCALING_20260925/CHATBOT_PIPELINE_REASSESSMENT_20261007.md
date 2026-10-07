# Reassessment: pretrained chatbot to the Silicon C engine

7 October2026, following the human's explicit reminder. Goal ACTIVE/INCOMPLETE.
Final input is an instruction/chat pretrained LLM; final output is a usable
converted chatbot running in the project engine. A runnable component, an
infilling result or metadata inspection is not that pipeline.

## View of the complete project

We have substantial source-contract, transformation, export, C arithmetic and
local/full-model verification machinery. We do not yet have one admitted
compact CHATBOT artifact with fresh dialogue/task preservation AND>=50 accepted
end-to-end batch1 IDs/s. Useful large n/RAM, LUT winner AND normalized mass,
physical DRAM and actual family/scale variants remain joint requirements.

The recent operational index gave most detail to successive Switch local
experiments. Its best whole result is important, but Switch infilling cannot
establish conservation of an instruction-following chatbot. Restore the complete
chatbot pipeline as the primary decision surface; local experiments must change
a named conversion decision or provide a rigorous rejection of a candidate.

| Evidence line | Actually established | Remaining boundary |
| --- | --- | --- |
| [127 Qwen Instruct C composition](METH_127_FULL_C_REFERENCE_RESULT_20260928.md) |Full reference core/attention/KV/FFN/E1280/logits;64-position same-stored-artifact parity;16.818 decode IDs/s with6 threads |FP32 core;prefill excluded;not the quality-valid BF16 candidate;below50|
| [183 learned child route](METH_183_E1280_CHILD_ROUTE_ALIGNMENT_RESULT_20260930.md) |Content-coupled selection beats all9 rotations in consumed24-source diagnostic |Small dense donor;not unlimited useful n or native whole quality/rate|
| [259 saved core](METH_259_UNIQUE_BANK_CORE_RESULT_20261002.md)/[284 C loading](METH_284_NATIVE_ARCHIVE_RESULT_20261002.md) |Complete artifact/source fields/unique function maps/native conditional lookup implemented;no donor-weight fallback |Full dense source FFN rows remain active;scoped arithmetic verification|
| [295 native prediction](METH_295_NATIVE_PRIMARY_PREDICTION_RESULT_20261002.md)/[296 generation](METH_296_NATIVE_GENERATION_RESULT_20261002.md) |Actual phase60 execution of the1.329GB276 artifact;fresh prediction gates and real greedy cache/health checks |Teacher forcing/health do not prove semantic chat preservation or accepted throughput|
| [297 native semantics](METH_297_NATIVE_SEMANTIC_RESULT_20261002.md) |Complete anonymous finite panel:41 unsupported/27 severe/0 missing versus donor52/34/1 and BF16 E1280 40/30/0 |Fixed criterion41>40 fails;recipe remains CLOSED;no unchanged native PIQA/K64/rate promotion|
| [511 Switch whole result](METH_511_WHOLE_HYBRID_HEAD_RESULT_20261007.md) |Fresh bounded infilling quality and SAME-artifact warm accepted prose52.55861/lower50.46847 |Not chat;source-sized/full WI/fixed128 router;first request27.14;all-book cost fails|
| [Giga conversion/source binding](../donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md) |Actual10.67B base-generation weights and family operators;MTP separate |Generic donor-port line stays frozen;no compact full-chatbot quality/rate|
| [397 large Qwen Instruct](METH_397_QWEN_NEXT_SOURCE_HEADERS_RESULT_20261004.md) |Actual41-shard headers,79.674B main inventory and source-specific cost screen |No tensor values or inference;not an actual80B conversion|
| [528 atoms](METH_528_VARIABLE_ATOM_FIRST_FAULT_RESULT_20261007.md) |New original ReLU-atom banks/masks;first resource fault and prefix sealed |No full fidelity/eligibility/C audit;ReLU-specific,not a chatbot pipeline|

Preserve every original failed gate. In particular, donor-only297 counts do not
retroactively turn its stricter frozen six comparisons into a pass. New methods
need justified prospective changes and fresh data, not favorable regrading.

## Pipeline stages and current implementation boundary

1. **Identify the chatbot and interaction contract.** Pin source revision,
   weights/config/tokenizer, chat-template provenance, roles/history, BOS/EOS,
   sampling/stopping and supported contexts. A public name ending in Instruct
   is not sufficient evidence of the target's preserved behavior.
2. **Analyze and select the transformation.** Family-specific original operators,
   source-informed useful functions, coverage/error mechanisms and complete
   active byte/MAC/router/core/head/cache budget. Reject unsuitable fixed
   candidates before costly full assembly when a sound bound allows it.
3. **Convert and export the complete artifact.** Store every required core,
   function/router/mass/head parameter plus interaction metadata. Include exact,
   approximate and learned steps/provenance. No source checkpoint at inference;
   no random/uninitialized organ or silently omitted operator.
4. **Verify actual native chatbot behavior.** Fixed-history numeric comparisons,
   own-history autoregressive dialogues, multi-turn/system-role behavior,
   grounded instruction/task outcomes and negative controls on fresh excluded
   data. Evaluate the complete loaded C artifact, not a Python surrogate.
5. **Qualify the SAME artifact's total cost and scaling.** Accepted output IDs,
   prefill/decode/router/head and declared frontend costs, cold/first request,
   actual DRAM/workspace, useful distinct n and family/scale applicability.
   Metadata completeness or local RMS cannot substitute for these gates.

The engine's included family backends are genuine project C execution paths;
they remain individually scoped prototypes. Simply wrapping old scripts into
one command does not establish a converter. Pipeline tooling should expose
unsupported/missing stages and return their actual status.

## New concrete implementation: offline donor preflight

[Tool](../../../benchmarks/native_expert_scaling/chatbot_donor_preflight.py) and
[protocol](CHATBOT_DONOR_PREFLIGHT_PROTOCOL_20261007.md) frozen at
`ba407d15b53015c7b948d79c54d2955fb5948528`, then executed ONCE per local donor.
The reusable tool reads configuration/tokenizer metadata and safetensors
headers, optionally producer GGUF metadata. It executes no template, tokenizer,
tensor value, donor/model/native C, training or old scientific observation.
It reports missing pipeline stages explicitly; it is not the converter itself.

| New report | Observed source metadata | Retained resource |
| --- | --- | --- |
| [Qwen source contract](chatbot_qwen_source_contract_20261007.json) |290 tensors,494032768 BF16 named elements;dense SwiGLU/GQA;HF template present;generation EOS[151645,151643] |.500s/34254848B Python OS peak|
| [Giga source contract](chatbot_gigachat_source_contract_20261007.json) |5323 tensors,11479750784 BF16 named elements including807215168 at/beyond base depth;MoE SwiGLU/MLA |1.578s/37691392B Python OS peak|

Qwen report SHA `258313ec5e4800615d30e44ed61fd9e90086a295205fb3322054117a3c3fc225`;
actual executor chunkce53b4,exit0,wall1.3583s. Giga SHA
`88f25eb6055c0b4e7b06c5c755d1c3aded3ab1ac05574febac29eee6d2aabc81`;
chunk1cc1dd,exit0,wall2.4318s. Retained samples precede final report serialization.
Both obey60s/256MiB/2MiB,CPU10. No large-weight hash or tensor-value read.
The original `bytes_read` counter is a MISLABELED HASHED-EXTENT SUM:
7295720/18122233B. It excludes extra GGUF parse-pass/Git pipe reads. Reports
stay immutable; source now labels this `hashed_input_extent_bytes`. Total
elapsed/OS peak includes parsing. No replay to repair a counter label.

Revision is declared by caller; config SHA and read extents are actually checked.
Header hashes do not freshly verify full source weight identities or uniqueness.
Earlier source/value qualification remains separate. The output decision is
always `SOURCE_CONTRACT_INSPECTED_PIPELINE_NOT_QUALIFIED`.
[Actual executor receipts](chatbot_preflight_executor_exits_20261007.json) retain
all three exits0. [Terminal event receipt](chatbot_preflight_windows_terminal_20261007.json),
SHA `0ba158d05d8a24c786cf31bd09b676f486def3a519dde2397f2667638e59287c`, verifies
both original Python process instances closed, two typed UTC queries available,
positive event controls179810/179791 and zero relevant Event1000 records.
No source inspections were replayed to repair the counter label.

**Interaction findings:** Qwen's original296 deterministic assay stops only at
151645 under its own declared policy; native code does not implement both IDs
of generation_config. This is a scope difference, not retroactive296 invalidity.
Giga's local HF tokenizer config lacks a chat template/generation config, while
the producer Q4 GGUF has a template (SHA
`e0b8ec1e172a124fe688e705a6e48fc30c575c23dae7388e43007f150d8343a3`). The preflight
extracts its6076027B metadata prefix without tensor descriptors/values. It does
not automatically adopt that template or prove HF/GGUF text-to-ID equivalence.
Both issues must be explicitly resolved before default/chat API parity claims.

## Algebraic boundary: ReLU results do not automatically transfer to SwiGLU

Switch's channel is d_j ReLU(i_j dot x). Qwen/Giga channels instead have form

    a_j(x) = d_j SiLU(g_j dot x + b_gj) (u_j dot x + b_uj).

Qwen bias/config details and source precision need exact per-tensor qualification;
Giga's ordinary gated FFN uses the bias-free form. In real arithmetic a retained
subset V always gives an exact omitted-function decomposition

    f(x)-g_V(x) = sum_(j outside V) a_j(x).

This separability transfers. ReLU's negative halfspace contributes EXACT zero;
real SiLU(z)=z/(1+exp(-z)) is nonzero for every finite z!=0. Its up projection
also carries input information. Physical codec/LUT underflow or saturation can
create zeros, but that is a separately qualified numerical contract, not a
source halfspace theorem. Counts/cheap widths observed in528 do not transfer.

SiLU(z)-z=SiLU(-z) does hold exactly. However folding a positive-reference gate
as z changes a SwiGLU channel into a PRODUCT of affine forms. Bias-free folded
output coordinate has quadratic term

    x^T Q_d x, Q_d=sum_F D_dj sym(g_j u_j^T).

It is not523's linear Lx. Materializing all Q_d naively uses cubic size in hidden
dimension; retaining factor products or approximating within regions requires
its own cost/coverage proof. Extra expert storage is allowed, but cannot hide
those active products or full dense core/head costs.

At fixed history, errors across layers obey a bound of form
e_(l+1)<=L_l e_l+epsilon_l ONLY when those L_l/epsilon_l are actually established.
For a linear head, logit perturbation is H delta_h+delta_H h+delta_H delta_h,
plus separate runtime rounding errors. Argmax is protected by
2||delta_z||_infinity < donor top1-minus-top2 margin. A local1% RMS supplies none
of these uniform hypotheses. After a choice changes, own-history generation
must be evaluated; a shared teacher history is no longer the realized state.

## Chosen next action and limits

Keep528's full checkpointed continuation as a conditional option. First use a
[prospective necessary-failure certificate](METH_528_PREFIX_NECESSARY_FAILURE_NEXT_20261007.md):
FIRST independently verify completed parents1..12, then compare their error
energy LOWER bound with the FULL original-domain source energy. If even this
prefix, with outward guards for the original F64 decision, proves failure of
an unchanged whole-domain1% gate, reject THIS recipe
without computing114 additional parents. This is algebraic early rejection,
not a sample-only threshold or scope reduction. If inconclusive, finish the
original bounded checkpointed comparison; do not infer a pass from a prefix.

After this ONE decisive comparison, prioritize a family-specific SwiGLU transfer
contract and complete CHATBOT validation. Bind canonical role/history/stop
fixtures before new chat evaluations; do not reuse Switch cohorts as chat
quality. No further nearest-plane precision ladder or unchanged297 promotion.
The next transfer preflight must price the entire causal core/head/cache/router
and coverage, not only a compact selected expert. Full strategy remains open
to nonlinear regions, redundancy, factorization or jointly trained adaptation
when a source-specific analysis justifies them.

No heavy source-family port or new resource is launched by this reassessment.
All old jobs stay terminal, foreign tracked/untracked work preserved. Goal
ACTIVE; no percentage/completion claim. The result of this turn is a corrected
pipeline decision surface and one implemented/observed reusable source stage.
