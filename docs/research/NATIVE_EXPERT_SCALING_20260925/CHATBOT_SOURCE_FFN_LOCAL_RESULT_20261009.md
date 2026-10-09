# True source FFN operands, calibration and fixed function recovery

9 October 2026. COMPLETE; local absolute fidelity FAIL. This is an offline
conversion stage toward original engine.c ternary/AQ63/LUT functions. Original
source width and active work are retained to isolate arithmetic before selection
and compact-core changes. It is not a final affordable donor runtime.

## Source and common operands

Falcon-H1-1.5B-Instruct revision80ebc50d7799a440b96c93bb6686a3924a09b0cb;
source weight SHA1fb788513f2c58e3abb91af8730afe19e247bce7a7f01a46e452a084e98b2abd.
Original D2048/L24/FFN4608, SiLU/MUP positions, BF16 casts and source tensors.
Sites0 and23; four metadata-selected short/long FIT/DEV cached forced trajectories:
rewriting008/magpie022/rewriting035/magpie036,18/256/12/256 labels.
Actual post-residual/pre-FFN-norm x and FFN y:542 positions,8,880,128 raw BF16 bytes.
All35,521,054 newly obtained source logits coordinates equal retained BF16 bits.
No source replies generated or RESERVED queried. Source SSD helper unchanged.

At the same floating x, F32-only output error is .0035-.0045; local original BF16
batching has .0022-.0037 error versus cached incremental GEMM, explicitly measured.
AQ63 with original F32 weights: .1940-.2435; ternary without AQ: .5173-.5624.
Combined package:site0 .5576-.5799/site23 .6473-.6581. These are normalized function
L2 errors, not weight norms/knowledge fractions; component losses are not additive.
All baseline code and row-scale bits match original packed sector9aa6f319.

## Exact real-algebra calibration and retained proposal error

For F(x)=b D[(Ux)*SiLU(aGx)], x'=Sx,G'=G/S,U'=R U/S,D'=D/R preserve F in real
arithmetic. F32 numerical identity was checked; maximum FIT relative defects
4.8691e-7/site0 and1.6919e-7/site23, before final BF16 cast.
Nine alpha/beta{0,.5,1} candidates/site, FIT-only case-balanced selection, DEV
checking:site0 picks identity/no improvement;site23 alpha0/beta1 changes DEV
errors .647276/.647260 -> .488125/.501664, case-mean ratio .764590 (~23.54% gain).
Overall required two-site improvement fails.

The original proposal reversed input balancing direction: for x'=Sx,W'=W/S,
equal RMS implies S^2=B_weight/A_input. Preserve original positive-grid failure;
the corrected negative alpha grid{-.5,-1}, beta{0,.5,1} adds12 new FIT trials and
reuses six alpha0 trials. Same selected functions; overall FAIL. This closes
this finite diagonal grid, not ternary conversion in general. Selected S=I at
both sites adds no input multiplication; hidden R folds into coefficients.

## Actual fixed local learning

256 effective updates/site from selected functions; three F32 master matrices
plus positive row scales (28,322,816 trainable elements/site). Exact deployable
nearest-even trits/AQ63/int-dot/F32 scales/MUP/SiLU/final BF16 forward, declared STE
surrogate backward. All initial code/scale/inference/STE BF16 bits equal retained
selected responses. Same4608 active rows; no new deployed operator or reference.
AdamW5e-5/.9/.999/eps1e-8/wd0/foreachFalse/gradclip1; scales>=1e-8. Alternate FIT
short/all18 and long/cyclic64 rows. All six gradients and masters/moments finite;
no DEV fitting or checkpoint selection. Final packed fields roundtrip all pairs.

| Site / case | Before error | Final error | Final cosine |
|---|---:|---:|---:|
| 0 / FIT short | .579868 | .117989 | .993015 |
| 0 / FIT long | .557603 | .355843 | .935934 |
| 0 / DEV short | .560921 | .539483 | .846220 |
| 0 / DEV long | .558897 | .538734 | .845314 |
| 23 / FIT short | .485921 | .093480 | .995686 |
| 23 / FIT long | .484984 | .121257 | .992918 |
| 23 / DEV short | .488125 | .191030 | .984689 |
| 23 / DEV long | .501664 | .224401 | .977498 |

DEV ratio .962850/site0 (~3.72% improvement), .419716/site23 (~58.03%). Both retain
each DEV case within1.05; only site23 meets relative .90 criterion. Neither meets
all-case absolute error<=.10/cosine>=.99. SOURCE_FFN_LOCAL_RECOVERY_FAIL. Real
learning is feasible and site23 partially generalizes in this consumed scope;
this is not a proof of sufficient conversion, local capacity ceiling or broad quality.

## Failure, resume, resources and artifacts

Original worker f45c1d0/binding7172fd54 failed after32 actual site0 updates because
logging supplied seconds twice. Durable state2 only;30 unpersisted steps lost.
Repair24c8635/bindingeb53c727 restores model/Adam2/RNG/history;510 new updates.
Final effective512, combined actual542 including30 discarded/re-executed updates.
Original family24.516s/exit1;repair67.515s/exit0, combined92.031s. Logging repair
and best-effort fault snapshots do not change algorithm or gates. No live job.
Final states:site0 SHA634a2db60615f4cf760524fc50bc8d5cc20eea4f4e5ae64a20ba5c72f0deb816,
339,920,840B;site23 SHAc9a10967d7f358ed304251a2de5283015bab6d292a4f44afd10ed063b60198c5,
339,920,520B. All model/moment/scale/RNG/history/provenance tensors retained off-repo.

| Family | Freeze / binding prefix | Wall seconds | Held OS bytes | GPU allocated / reserved bytes | Output bytes |
|---|---|---:|---:|---:|---:|
| Capture |34c600a /ea78e341|100.047|3,723,501,568|5,580,204,032 /6,908,018,688|8,938,023|
| Positive grid |34c600a /6581468f|34.375|1,482,698,752|592,355,328 /744,488,960|126,825,780|
| Corrected input grid |998e7eb /1dd6b4cc|45.219|1,212,190,720|574,214,144 /704,643,072|68,928,300|
| Recovery repair |24c8635 /eb53c727|67.515|2,028,957,696|893,831,680 /912,261,120|1,052,850,277|

All completed family input/resource checks PASS; exit0 denotes completed
measurement, not scientific quality PASS. Worker timings differ from family
wall time; family includes import/hash/verification and terminal retention.
Worker JSON elapsed91.391/25.641/35.062/57.156s respectively. Event completion
timestamps include work after result construction; do not substitute them.

Tools:[capture/calibration](../../../benchmarks/native_expert_scaling/chatbot_source_ffn_local.py),
[mirrored grid](../../../benchmarks/native_expert_scaling/chatbot_source_ffn_input_balance.py),
[local recovery](../../../benchmarks/native_expert_scaling/chatbot_source_ffn_local_recovery.py).
Bindings/results/terminal JSONs and logs beside this record provide exact commands,
input/output sizes/hashes/PIDs/timestamps. Respective result SHA prefixes:
2b6e7153/b09061a9/cecd7007/1bf8b912. Protocols and prelaunch/resume fault records
remain immutable. Large sectors, x/y, responses and states are off-repo under
results/native_expert_scaling/chatbot_source_ffn_*_20261009.

## Decision toward engine.c

Completed [saved geometry/centering](CHATBOT_SOURCE_FFN_RECOVERY_GEOMETRY_RESULT_20261009.md)
qualifies the interpretation:91-94% of site23 DEV squared-error gain is mean
correction. A constant fitted only on FIT nearly matches that learned total
error; centered DEV error still .467/.492. This is partial response recovery,
not evidence of broad input-conditioned capacity preservation. At site0,
amplitude-only scaling of the learned correction cannot fix its poor direction.

Do not replay grids or extend256 updates on these same two FIT prefixes by default.
The saved-only diagnostics are COMPLETE and reusable. Broader independently
sourced input coverage is the next
candidate variable before all-site/whole-source recovery, selection and compact
SSM-core transfer. Changes must stay inside a measured engine active-cost budget.
Fresh useful same-artifact chatbot quality+>=50 accepted IDs/s, useful n/CPU LUT
IDs+mass/DRAM and other families/scales remain open. Month-plus T4 feasibility
cannot be inferred from this two-site local price alone.
