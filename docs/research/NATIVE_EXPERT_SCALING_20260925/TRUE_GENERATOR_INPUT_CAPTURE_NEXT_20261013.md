# True generator-input calibration after the temporal counterexample

13 October2026. **CONDITIONAL NEXT / PROPOSED / UNIMPLEMENTED / UNBOUND.**
Choose after the full temporal audit and fixed error-decomposition audit.
If the state failure is reproduced and explained by regenerated forcing,
capture true module inputs before function compression. Numerical faults must
be adjudicated first. Do not relax the fixed1% final-state gate or optimize the
inverse on the failed DEV history. The response-rank proposal remains conditional.

## Question and expected decision

Existing gate-derived inputs passed all local operand consistency gates but
produced a paired final-state counterexample at site23. A true in_proj input
capture supplies the missing actual BF16 input boundary and lets us measure
the inverse's real input error,rounding effects and replay fidelity directly.
This changes calibration data,not the architecture or goal. It prepares
source-function fitting/conditional ternary transfer with observed supervision.

Use the SAME pinned Falcon-H1-1.5B-Instruct weights/config,actual source
constructor dtypeBF16/MuP buffer,tiled reference SSM helper and original48
complete forced histories. No LM-head,new answer/generation,optimizer or
RESERVED. One full base-model prefill per history,zero source state/use_cacheFalse.
All24 layers execute;only sites0/12/23 are captured. H is args[0] at in_proj:
AFTER ssm_in_multiplier .625 and BEFORE projection. Assert actual BF16 dtype,
shape(1,history,2048),finite values;save original bits,not casted inputs.

Capture true H and all7 existing recurrent fields x/B/C/delta/gate/y/output
at these sites,using the original observation boundaries. All144 pairs must
match corpus histories/IDs/positions. Fixed per-case/per-field relative RMS
continuity budget1e-4 against the adopted original capture,with exact bit
mismatch counts and centered/label/domain/worst retained. This distinguishes
a true-input observation of the same source computation from a changed
runtime/model history. Freeze exact protocol/gates before execution.

After capture:full source parameter identities/dtypes/runtime/hash/resource/
terminal and data finite/shape/extents audit,all continuity fields. Then
replay actual H through the already qualified promoted/rounded generator
and fixed native recurrence,with independent full audits and unchanged
field1%/paired own-output1%/final-state1% controls. Report true-H versus
H64/H_BF16 errors without treating H64 as an alternative selected on DEV.
Capture alone does not prove CPU generator/CUDA reduction equivalence.

If continuity fails,retain all completed evidence and identify the changed
source equation,dtype/helper/history/runtime before adopting the new data.
If true-input replay still fails,its original CUDA reduction/rounding or
observer semantics become the isolated defect. Neither outcome supports
more inverse fitting on DEV.

## Budget evidence and launch prerequisites

The old all24-site capture logged45 completed cases at1267.25s,then disappeared
without a known terminal;this is partial elapsed evidence,not full48 runtime.
The separate3-case completion terminal reports301.75s including family work.
Do not claim a precise new runtime from either. Reducing captured sites lowers
data traffic but the whole pretrained base model still executes.

Proposed local RTX3060 job family<=1800s/reserve90s,OS12GiB,GPU allocated10GiB/
reserved11GiB,new ordered E family<=3GiB. Raw payload is
22547*3*(29792+4096)=2,292,218,208B,including277,057,536B true H. Charge actual
metadata/logs/workspace/launcher,not just tensor payload. No T4 allocation.
Audit CPU producer900s/audit1200s/OS6GiB proposed for subsequent replay;exact
implementation/runtime/compiler/data/limits remain UNBOUND.

Before launching,announce purpose/resource budget/stops to the user;freeze
source,source-metadata/corpus/method/old capture/runtime resolved paths (including
CUDA/CUBLAS/driver witnesses),storage and exact launch. Use a bounded owned
worker/launcher recording PID+creation time,terminal/resources/partial faults.
Do not interfere with unrelated Python daemons. Check actual GPU availability
and memory now,not from old screenshots or old PID inventories. A timeout is
not termination. If later design requires hours,follow user's announce/bind-
launch/PAUSE rule;no hours-long work is launched by this proposal.

## Route onward

With qualified actual input data,implement [FIT source response-rank envelope](SOURCE_FUNCTION_RANK_NEXT_20261013.md),
then cheaper shared input/output plus conditional ternary residual functions.
Keep trained shared recurrence and explicit causal information boundaries.
Joint layers,own emitted histories,source categorical/chatbot quality,CPU
IDs+masses,real expert DRAM,useful n,conversionO(n) master/gradients,multiple
donor families/scales and same useful artifact>=50 batch1 token/s remain required.
