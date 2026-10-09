# Common-operand source-group sum versus selected mixture

9 October2026. Prepared before new values. Run only after the whole boundary286
evaluation is terminal and if its quality does not support extending the recipe.
Question:can amplitude alone reconcile the initialized selected mixture with
the complete projected group sum, or is directional information omitted?

## Reuse and isolated variable

Reuse all3132 stored C ff_input operands from six old pilot prefixes,261
positions/12 sites. Four FIT prefixes170 positions,2 DEV prefixes91 positions.
Reuse all12 actual source-informed INITIAL master tensors from their retained
component files,not the trained/ternary packed outputs of the old bank diagnostic.
New variable is aggregation of these projected full-precision initial groups.
No old native/source/learner forward or generation repeated.

At common x512 compute F64 outputs of all72 projected SwiGLU groups from actual
F32 initial gate/up/down masters. Each36-group half came from one source FFN.
Both halves see the same operand here:their sum is a diagnostic proxy,not the
source's actual two-block composition. These old student operands are not a
sample of source hidden states or the new checkpoint286 distribution.

Let f_j(x) be these F64 group outputs and F=sum_j f_j. Use the exact saved
initial projected router in F64,stable top8 ties by lowerID,selected-softmax mass.
Compare selected sum S,normalized mixture M,72*M,and c_FIT*M. One c_FIT per
site minimizes summed squared error on FIT only; report its actual sign/value
and DEV error without choosing another scalar after DEV. Additionally compute
optimistic per-operand unconstrained oracle c(x)=<F,M>/||M||².

If any DEV oracle response has relativeRMS>1%,amplitude alone cannot meet the
fixed1% component criterion on all these operands. Record count/distribution;
decision AMPLITUDE_ONLY_REPAIR_INSUFFICIENT_ON_TESTED_OPERANDS. If oracle has no
failures and c_FIT also no failures,mark a fixed-scalar candidate; otherwise
scalar requires input dependence in this scope. This does not change the old
1e-4 native logit gate or establish a chatbot quality ceiling.

## Algebra and retained evidence

For each site/split retain72x72 group Gram C_ij=sum_t<f_i(x_t),f_j(x_t)>;
sum_ij C_ij must match total||F||² within relative1e-10 in F64. Report
||F||²/sum_j||f_j||² to quantify alignment/cancellation. All values finite,
reference/mixture energy strictlypositive; otherwise preserve a fault without
epsilon-filled divisions. Retain all3132 F64 vectors F/S/M,all row/scalar/error
metrics/IDs/masses and Gram arrays,hashes/extents. No exact-real elementary
function certificate or independent CUDA implementation is claimed.

Missing groups,core/width reduction,projection and ternary arithmetic are
different uncertainties. This isolates aggregation at fixed projected masters/
common inputs; it cannot measure donor-preservation loss from the other changes.
An amplitude failure motivates shared-plus-private/overlapping covering experts
or trained source-function approximation rather than blind multiplication.

## Resources and identity

ONE local GPU family<=300s/4GiB OS/2GiB allocated GPU/3GiB reserved/128MiB output,
worker30s reserve. Shape deduction44.3B F64 matrix products for all groups;
roughly38.5MB retained responses plus~1MB Gram/metadata. This is bounded operator
work,not an accepted-token benchmark. Actual cost required before any future fit.
Python3.12.10/Torch2.6.0+cu124/NumPy2.4.6/psutil7.2.2;no optional model kernels,
no source/model/optimizer,full library certificate or T4.

Bind trace hash/native receipt/query positions,all12 initial component extents/
manifest SHA,worker/launcher/Python/protocol and selected runtime files. Extend
the launcher only AFTER the live boundary286 evaluation exits;freeze before
binding/run. Preserve faults/partials,never automatically replay completed rows.

Whole pretrained-to-original-LUT/ternary/SSM chatbot quality plus>=50 accepted
batch1 IDs/s on the same artifact,useful n/normalized CPU routing mass/physical
DRAM and actual family/scale variants remain required. No promotion from this
local controlled comparison alone.
