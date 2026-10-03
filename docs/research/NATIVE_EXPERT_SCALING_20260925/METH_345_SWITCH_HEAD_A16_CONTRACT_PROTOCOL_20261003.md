# METH-345: actual HEAD A16, core/experts A8, unchanged serialized target

Freeze before any numerical/model output or timing. New variable: final head
activation only becomes symmetric signed16 with F32(absmax/32767), nearest-even
F32 division, clipped[-32767,32767], zero vector scale1. All upstream operations,
batched341 prefill, serialized I8 weights/F32 row scales, router/control/lookup,
actual256 learned experts/bank and 338 payload/manifest remain identical.
Selected after343 original-primary95% top1 FAIL and344 fixed-state attribution;
343 failure remains. Consumed-state counterfactual35 versus55 differences is
motivation only. No NEW source, original source reload, download, fit or GPU.

## Numerical contract and artifacts

New opt-in `SILICON_SWITCH_HEAD_A16` in engine.c; original default tail unchanged.
`meth345_switch_head_a16.c`, `_entry.c`, `_reference.py`, `_contract.py` frozen.
AVX2 extends16 I8 weights to16 I16, pairs with I16 activations using VPMADDWD,
I32 lane accumulation. At cols<=4096 each lane has<=512 products, bound
512*127*32767=2,130,641,408<INT32_MAX. Horizontal reduction and tail are I64;
global bound4096*127*32767=17,045,131,264. Return
F32((F64(dot)*F64(rowScale))*F64(activationScale)); no saturation/FMA/reassociation.
Loader, core A8, cache, route/capacity and logical read counters unchanged.
Head codes are ephemeral16, weight reads remain I8 plus original F32 scales.

Independent reference uses NumPy I64 matrix sums and prescribed scaling on
actual serialized codes, with qualified336 rounded upstream arithmetic. Separate
head forward override restores prior methods even on exception. Original
donor is not relabeled as this numerical estimand. Full teacher is unnecessary.

## Fixed gates and order

1. Thirteen primitive cases, seed345345: cols8/15/16/17/31/32/33/768/3072/4096,
   nearest-even positive/negative half ties, random/cancellation/tails; additional
   all-positive/all-negative/zero4096 extreme vectors. Codes, F32 scale/output,
   I64 dots and Python scalar sums EXACT; max dot17,045,131,264 observed. Primitive
   SWI8D001 input; SW16R001 output carries F32y/I16codes/F32scale/I64dots.
2. Same328 Tiny seed/config/weight SHA, capacities1/64. Complete encoder,
   decoder cache/states/logits/routes BYTE/element EXACT independent reference;
   all nine semantic faults detected by logit relative error>1e-4. Require A16
   head hook four calls and nonzero A8/attention/softmax operations.
3. Fresh whole14,818,015,744B target338 SHA and actual manifest SHA, unchanged
   file size/mtime throughout. Prior336 all6392 native code/scale audit reused;
   no claim of a new full source archive-envelope verification.
4. Both consumed336 engineering cases vs full independent A8-core/A16-head
   official architecture with actual F32 controls: all outputs EXACT, upstream
   states/routes identical old336, threads1/6 with profile0/1, repeat SHA exact.
5. Both old341 long forced32 fixtures, source9/64, threads1/6: complete core/
   routes EXACT341 and head EXACT independent I64 projection of saved normalized
   states. Not timing gates in this experiment. Save expected output SHA for
   subsequent same-binary actual CPU cost.
6. ALL96 consumed343 cases, actual native thread6 full forward: encoder/decoder/
   route/probability exact old343, logits exact retained344 native_a16 oracle.
   Bind every old binary/counterfactual array SHA. Any mismatch fails; no rerun
   or tolerance relaxation. No original-quality gate applied to consumed data.

## Resources, reproduction and decision

Isolated324 Python/official4.57.6/Torch2.6.0+cu124, original model-source SHA;
CPU only, reference1/native1or6, passiveOMP/KMPnone, OMP_PROC_BIND absent.
Compiler/runtime SHA reused336, O3 C11 x86-64-v3/no-fast-math/ffp-contract-off/
OpenMP. No other model or timing job. MAIN<=20min after imports, combined
checked RSS<=16GiB, child failures terminate and preserve partial/failure JSON.
Working directories immutable after run; never rerun into same path.

```powershell
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks\native_expert_scaling\meth345_switch_head_a16_contract.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth345_switch_head_a16_contract_result.json
```

ALL gates PASS licenses separate actual complete CPU cost at unchanged341
20ms decode/full per32 forced positions and1.10 repeat ratio thresholds. Only
then freeze NEW original-primary quality excluding all342 consumed books.
Failure preserves raw/partial before separately named repair. Neither success
nor counterfactuals prove accepted generation, useful larger-n/LUT/physical
DRAM/cross-family/~100B or goal completion.
