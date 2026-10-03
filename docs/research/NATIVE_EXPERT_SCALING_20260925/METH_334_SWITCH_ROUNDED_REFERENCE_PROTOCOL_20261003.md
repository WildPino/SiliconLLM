# METH-334: explicitly rounded independent norm reference

Prospective after preserved333 primitive failure (freeze d28241f/raw777d97b).
333 matrix/attention NumPy oracles exact, softmax error1.86e-9, norm7.15e-7
>2e-7 FAIL before C/Tiny/source execution. No333 gates/data relaxed or hidden.

Change ONLY independent reference norm primitive: F32 squares/F64 mean/F32
mean and F32 epsilon addition preserved; evaluate sqrt in F64 then round F32;
evaluate reciprocal of that rounded root in F64 then round F32. This realizes
the declared correctly rounded F32 primitive boundary without relying on Torch
standard F32 primitive approximation. Check SAME NumPy F32 norm oracle2e-7;
record separate standard-versus-rounded sqrt/reciprocal differences on the
same arrays. They identify implementation differences only on fixed operands,
not a universal primitive quality statement. Original333 helper immutable.

Reuse immutable333 native C/audit wrapper/engine macro EXACTLY. Controller334
copies333 with new reference/protocol/output names and pinned333 failure only.
Same333 full specified arithmetic policy and numeric estimand; no C precision/
architecture/weights/inputs/gates changed. Original329/330/332 original-backend
probability guards remain FAIL. This does not establish original donor quality.

Same primitive seed333, exact NumPy matrix/attention/F32 output and softmax/norm
<=2e-7. Same exact328 Tiny weights, IDs, capacities1/64, nine semantic faults;
stop before source if primitive or Tiny fail. Complete6392 actual source byte
audit, fresh original sixarchive/sidefile hashes, same two consumed source
engineering inputs/forced decoder IDs. Match independent prescribed arithmetic
for every-state/logit<=1e-4, selected probability<=1e-6, exact route/capacity/
greedy and zero-head fault>1e-4. Save full matched NPZ arrays/hashes/op counters.
Original unmodified official4.57.6 PRIMARY donor must later score NEW untouched
reconstruction/prediction/generation/task sources. No quality substitution.

Freeze BEFORE observation. MAIN20min AFTER imports, checked combined32GiB,
compiler120s, CPU1, no GPU/download/training/overlapping timing; preserve all
failures/resources/commands/binary/runtime/code/source hashes as333. Installed
project/official source unchanged; restore instance norm methods after calls.
PASS only qualifies prescribed implementation apparatus for separately frozen
compact integer/scaling reference and cost preflight. FAIL closes unchanged
candidate. Neither proves final LUT/RAM-useful n or SAMEartifact accepted>=50
or multiple families/~100B. See333 protocol for complete recipe and boundaries.

Command:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth334_switch_full_source.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth334_switch_full_source_result.json
```
