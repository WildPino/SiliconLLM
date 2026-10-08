# Whole nonlinear transfer connection: implemented prerequisites

8 October2026. Goal ACTIVE/INCOMPLETE. [Prospective protocol](CHATBOT_WHOLE_TRANSFER_PROTOCOL_20261008.md)
was frozen BEFORE both actual phases. The previous goal turn was PROGRESS:
one real quadratic converter and independent necessary rejection. This turn
implements a new connection toward the complete CHATBOT pipeline; no admitted
compact chatbot or full adaptation yet. [Next initializer](CHATBOT_WHOLE_INITIALIZER_NEXT_20261008.md)
is the exact point of resumption.

## Reusable code and actual implementation boundary

[Whole model module](../../../benchmarks/native_expert_scaling/chatbot_whole_transfer_model.py)
now provides a 3D BF16 -> initialized2D F32 compact SwiGLU -> BF16 MLP adapter,
an ALL24 replacement API with source-config/tied-head/core/parameter-alias guards,
masked T=1 output KL and explicit fully resident F32 Adam allocation. The API
preserves218 non-FFN parameter objects/180246400 elements and permits only the
165150720 compact shared/private trainable elements. Source dense FFN fallback
is absent from adapters. Routing remains frozen F32; compact functions readfull x.

Successful ALL24 replacement and full optimizer allocation have NOT executed.
The new test executes one actual initialized block restored from immutable
closed structural-fit coefficients, on NEW synthetic inputs, with a fixed
synthetic downstream head/teacher logits. This validates an interface, not a
pretrained capacity substitute or source fidelity. The old local recipe remains
closed. Source-derived ALL24 initializer/global finite fit/export/C operator/
canonical native chat remain missing.

Inspected old donor_adaptation/s1/meth136_matched_sparse_train.py contains
whole teacher/student training/nonreentrant checkpoint/KL supervision, but
uses hierarchical augmented teacher/source-sized FFNs and output-B factors.
Its operational line stays frozen; no old learner is resumed or duplicated.
Reuse the design when writing the actual new whole fit.

## New conversion-memory and native-storage ledger

Use immutable source census and original geometric product counts; this is NEW
optimizer/gradient/output-liveness/F32-router accounting, not old census/geometry
replay. Independent teacher and student non-FFN copies are conservatively
budgeted. All gradient/moment buffers are explicitly reserved by this resident
optimizer design, including currently inactive leaves. This is a lower bound
for THAT allocation design, not a universal lower bound for sparse/offloaded
training. The current whole adapter supports E16; E160 is a dimension scenario
of extending the same resident design, not an executed supported whole model.

| Quantity | E16 | E160 scenario |
|---|---:|---:|
| Trainable F32 G/U/B elements |165150720 |1354235904 |
| Fully resident weights/gradients/Adam moments + teacher/frozen core/router bytes |3993823744 |23019642880 |
| Persistent lower-bound screen versus12GiB |Not excluded; peak UNMEASURED |FAIL fully resident Adam |
| Complete proposed matrix MAC/token |246935552 |246966272 |
| Native logical coefficientB/token with F32 router |495418368 |495545088 |
| Proposed native payloadB |693597440 |3072274688 |

Both necessary3/5 complete matrix/logical gates PASS; no physical DRAM/rate
inference. Source attention/head remain complete. Context attention43008*context
MAC and native F32 KV4096=100663296B reused separately. One F32 logit tensor
at128 tokens is77791232B; teacher+student named logits155582464B. Attention,
SwiGLU activations, log-softmax/exp/backward, temporaries/allocator/workspaces
are additional. The ledger is NOT a training peak bound or GPU admission.

Actual one-block constructor has6881280 parameter elements and29728 F32 buffer
elements, matching the new count. It registers unused C=1 child centers/norms;
these are charged to conversion. Proposed native C=1 export may omit them,
but that format is not implemented. F32 router adds explicit delta to previous
BF16-router dimension ledger; old results remain immutable.

## FIRST actual adapter/output-gradient qualification

NEW [1,4,896] exact dyadic grid, BF16 input/output; NEW17-column fixed head and
teacher/mask[1,0,1,1]. Actual input autograd gradient nonzero/finite, shared and
selected leaf gradients finite, unselected leaf gradients absent, routing
buffers not differentiated. Teacher gradient absent; masked logit gradient
EXACT zero. Independent F64 scalar fsum log-softmax/analytic logit-gradient
formula checks saved NEW logits without re-evaluating the field:

- Torch KL .07414790987968445; independent scalar .07414790396416726.
- Loss absolute gap5.915517187204955e-9 versus fixed2e-6.
- Maximum logit-gradient gap6.727064315315001e-9 versus fixed2e-7.
- ALL6 invalid interface cases reject: uninitialized block, layer24, rank2/F64
  input, empty/negative mask.

This is the Torch autograd propagation mechanism through dtype casts, NOT an
exact derivative of a discontinuous BF16 quantizer or a whole donor-response
Jacobian. The independent analytic certificate is at F32 logits. Mixed-precision
training uses surrogate cast gradients and still needs actual endpoint tests.
No GPU initialized, HF/source/full model forward, old field/input replay or
optimizer update. ALL24 assembly/native equivalence are not tested here.

## Frozen records, resource measurements and closure

Actual source freeze `dff00bc3473465bf076920e0759e5e87fea3247d` for both phases.
Inventory binding SHA `9b2aac175d74c8c3a2196905be7cf8423b4d3fcaa0ef6b5280186232047c7c25`,16
inputfiles/five runtime roots. [Raw inventory](chatbot_whole_inventory_20261008.json)
SHA `effcd2a171a6ba96f0b68f22824ddaa26adbcb8b609691b6f6b09caa0e432077`;
[terminal with exact command](chatbot_whole_inventory_20261008.terminal.json).
Compute-before-final-serialization.219s/family1.875s, worker OS peak28119040B
THROUGH EXIT/family57667584B, exit0; empty output directory.

Interface binding SHA `ce801ed41a5d5f4909d700795a21b2922ee7ab8c5c275b7d74640f103896e9f3`,18
inputfiles/27 restricted runtime roots. [Raw mechanics](chatbot_whole_interface_20261008.json)
SHA `e261195f06e8391140cf891f8bf7c9c8f07b2a239d84649c971d08ef16ad8a48`;
[terminal/typed output manifest](chatbot_whole_interface_20261008.terminal.json).
Compute5.688s/family23.875s, worker peak503840768B THROUGH EXIT/family536702976B,
exit0; five C-order F32 logits/mask/gradient witness arrays total15808B.

All original worker/family/RAM/output/log/input-before/after gates PASS; no
phase fault or repair. Launcher final receipt/stdout tail remains outside last
memory snapshot as declared. [Administrative closure](chatbot_whole_transfer_terminal_20261008.json)
SHA `59d19c4512996c7f774a211fd97dc906a6c7c399ea509c74e934e8339f0e0d2c`:
all4 known instances closed, positive Event1000 controls179810/179791,zero matched
OS faults. Three foreign tracked SHA unchanged; no scientific process live.

## Decision

E16 whole source initialization is not excluded by this resident-memory lower
bound; its real peak still needs measurement. E160 fully resident Adam is
rejected, not inference useful n/RAM or CPU-offloaded methods. The new adapter/
output-gradient mechanics qualify the next ALL24 initialization/assembly
prerequisite. Implement it next; do not clone layer12 across layers or treat
synthetic mechanics as chatbot preservation. Full goal remains unachieved.
