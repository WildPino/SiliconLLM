# Source-head moments for a changing decoder and causal transfer

12 October2026. **PROPOSED/UNIMPLEMENTED/UNBOUND**. Selected next after complete
all-label oracle and entropy-mixture audits. Current upper budgets fail;dual
family gives only tiny positive lowers below budgets. Neither head adequacy nor
head obstruction proved. Avoid indefinitely optimizing fixed per-label oracles;
make the supervision usable for a flexible decoder and an actual causal core.

## Exact sufficient statistics

Fixed actual source head A_source includes final gamma and LM multiplier. Current
Falcon source logits are A_source u,where u is a normalized2048-feature vector;
there is no added vocabulary bias in this scoped family. Teacher q is the same
actual-source F64 operator reference,not new generation or cached BF16 teacher.
Compile m_source=A_source^T q,H(q). For ANY source-feature output u:

`KL(q||softmax(A_source u))=logsumexp(A_source u)-m_source^T u-H(q)`.

`grad_u L=A_source^T p-m_source`.

Decoder u may change or be conditional:for y=mu+Bz and
r=sqrt(||y||²/2048+epsilon_source),u=y/r,
`grad_y L=grad_u L/r-y*(y^T grad_u L)/(2048*r^3)`.
Thus decoder and causal-code parameters can move without replaying source q or
recompiling a fixed257-column target moment every step. Conditional reader
C_e=A_source B_e is within this fixed source-head column space;new readers
outside that space are NOT covered by these moments.

Native F32 quantization of C_e generally introduces components outside the exact
F64 source-head factor product. This compiler does not make those machine logits
exactly source-derived;actual native export/probability/precision audit remains
required. Likewise source bias or changed gamma/head in another family require a
separate statistic/variant,not silent applicability claims.

## Finite first implementation and meaningful controls

All existing4422 FIT/4386 DEV raw source labels,original case/domain weights;
same source weights/epsilon/tokenizer records,full65537 probabilities. No model
forward/new data/T4/RESERVED/longalign. m_source F64 output
8808*2048*8=144,310,272B plus entropy/metadata,versus4,617,999,168B full F64 logits.
Storage scales with source feature width and label count,not experts n;still
report that cost for larger source dimensions rather than claiming100B admission.

Compile first,then independently qualify full source-feature moments with
different reductions/direct full q. Reproduce full categorical losses/feature
gradients for existing source-derived decoder points:original broad factors,
chosen coupled-gradient factors,and an audited oracle source reconstruction.
Use full direct source logits/probabilities,not arbitrary synthetic matching
probes. Keep current native F32 head/probabilities separate from F64 source-
derived controls. All outputs/binding/hashes/resource limits frozen before run.
GPUF64 source-width computation must be qualified against CPU for the changed
shape;prior257-column timing is not its cost. No optimization in compilation.

## What this enables next; what remains missing

Use source moments for a history-to-code model with a moving source-derived
decoder,not an encoder that receives the donor's final hidden state at inference.
Select a compact pretrained-initialized SSM/SWA/conditional-function geometry and
measure its real native active cost before a long distillation. Reuse trained
source generator/state projection evidence only within its verified assumptions;
full features retained by exact lifts remain costly until measured/changed.

Potential conditional low-rank readout corrections need explicit norm transport,
selected-rank memory/DRAM and CPU routing IDs+masses. They are not implemented
here. Static expert capacity and dynamic recurrent/SWA state are separate:causal
closure/future information/own assistant histories require dedicated checks.
Dense conversion masters/gradients O(n) remain unresolved.

After this compiler,implement a concrete causal forward/backward/native control
and donor-relative pilot;do not continue only emission-oracle work. Learned-model
held-out/own-history quality and same useful artifact>=50 across declared hardware/
contexts/families/scales remain the goal. Any hours-long required run follows the
user's announce/launch/pause instruction;no delegation/Luna.
