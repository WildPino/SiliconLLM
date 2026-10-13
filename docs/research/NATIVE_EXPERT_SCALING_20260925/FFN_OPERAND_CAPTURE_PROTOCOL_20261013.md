# Exact qualified-model FFN operand collection

13 October2026. Freeze before allocating source-model run. Question:what is
the actual nonlinear source FFN function distribution encountered by the
qualified24 native-state chatbot? This data enables full-size affine expert
coverage/remainder/routing/precision tests. No affine candidate is fitted here.
Before-SSM H from old traces is not the actual pre-FFN input.

Reuse EXACT model/SSM packets/F0/compiler/source parameters/source_method,
batch1/no padding/serial source operators/attention cache and actual22547 known
history IDs. FIT24/DEV24 only. OWN zero-initialized native states per new
sequence,persistent throughout,no teacher states/fields/answers injected.
Observer hooks only,original source FFN and decoder formulas unchanged.

Collect each of24 sites per consumed token:pre-FFN normalized x2048,up4608,
raw gate4608,scaled gate4608,actual product feature4608,FFN output2048,
all BF16. Scaled gate observer literally repeats Python alpha multiplication
on actual gate output;raw gate separates matrix and scalar errors. Feature
is original down_proj input,output is source FFN return after beta. No proxy
features or premature BF16 scalar conversion.6 field groups,6912 raw files,
12,190,531,584 BF words/24,381,063,168B. Plus1152 native snapshots and8808
full label score rows F32. Expected total~28.1GB,cap32GiB with>=1GiB metadata
margin checked before launch. Unique E ffn_operand_capture_20261013 family.

Every actual native stream write must match the admitted parent:initial
native output,later23 actual SSMin H/preconv MuP row/native output. Stream
validator reads parent only to check newly computed bytes,never feeds the
model. Byte/word comparisons and actual incremental stream hashes,all final
state/ring/seen/last,attention cache digests and every requested label logit
are exact controls. Save fresh snapshots/label logits for stored auditing.
Native streams already stored by parent are not duplicated. Dataset mismatch
is a mechanical collector validity failure,not a new model-quality verdict.
Keep partial bytes/terminal for diagnosis. No FAILED-EARLY quality shortcut.

All411 canonical BF16 coefficients before/after,actual MuP/scalars/config,
identities/_versions unchanged. New selected module bodies and numerical/
framework package trees/extensions pinned;their scope does not retroactively
repair the parent's historical transitive runtime pin gap. No exhaustive
standard-library/OS-runtime custody claim.

Separate CPU-only audit:all1152 case/site sets,every BF word finite,canonical
F64 up/gate/down matrices followed by declared BF nodes. Literal Torch CPU
Python alpha multiply on captured raw gate must be BF word-exact. SiLU and
product from actual captured up/scaled gate relative RMS<=1e-4 per case/site.
Canonical matrix/reduction plus scalar control<=1% per case/site inherited
source-equation accuracy policy. Record maxima and failures through complete
audit. Check fresh native snapshots/rings and label logits independently;
stream/attention digests are producer comparison witnesses. Shared Torch CPU
node kernels are declared;no independent GPU attention reproduction.

Local RTX3060,worker7200s/reserve180s,OS8GiB,GPU allocated6GiB/reserved10GiB,
family32GiB. Holder7200s/exact owned PID/create time and resource watchdog,
DETACHED8 worker. Separate CPU audit7200s/OS8GiB,no CUDA initialization.
At least34GiB free before allocation. Announce hours-long purpose/budget and
pause goal after verified live startup under explicit human instruction.
No delegated monitor/Luna/T4/RESERVED/optimizer/new generation/model speed.
No restarts from observation timeout. Require real terminal0 before full audit.

Next decisions only after full data audit:measure source smooth proxy and
nonlinear remainder,fit centers/operators/routing on FIT,hold DEV unchanged,
evaluate actual own-model expert substitution under frozen whole-model gates
and real CPU/DRAM cost. Gathering operands does not admit useful expert-n or
final engine>=50 useful accepted token/s.
