# METH-218: hard calibration passes fit, fails unchanged reserved layers

Protocol/runner frozen at `0de484b`. On the hash-bound METH-217 fit
capture, NumPy FP32-add/argmax reproduces original Torch counts exactly
for all five offending parents. Direct coordinate calibration converges
in one or two sweeps. Their corrected maximum shares are 11.368%,
12.250%, 12.602%, 13.840%, 13.628%; maximum bias change 0.000447631.
Exactly five raw vectors (45 FP32 values) change. All keys, projection,
ChatML biases and other raw vectors remain byte-identical. Export readback
is exact. Four changed raw layers are recomputed on all captured fit
selections; unchanged-layer and ChatML fit metrics reuse original identity.

| Cell | All gates pass? | Load-failing layers | Worst max-load ratio | Worst hot-parent share |
| --- | --- | ---: | ---: | ---: |
| Raw fit | yes | 0 | 1.003802 | 23.329% |
| ChatML fit | yes | 0 | 1.005573 | 11.811% |
| Old raw reserve | no | 3 | 1.338872 | 21.875% |
| Old ChatML reserve | no | 2 | 1.272918 | 23.875% |

Eight-prompt teacher/control logits match exactly before reserved
inference. Old raw fails only max-load at zero-based layers 8/15/19;
old ChatML fails only max-load at 7/23. Every share, coverage and content
gate passes. **All failing reserved layers have unchanged bias vectors**;
the targeted correction resolves its fit problem without causing these
failures. It does not solve the full local-key route's generalization.

Frozen decision: stop before new 512-draw reserves and source screens.
No B training or native/full-model promotion. Do not select further bias
corrections from these reserved outcomes. This diagnoses and repairs the
specific fit soft-to-hard gap, not a validated larger-n transfer route.
The targeted static calibration sequence is stopped. A changed
route/function-learning mechanism remains necessary; do not indefinitely
perfect load proxies without transferred conditional function and quality.

Local RTX3060: 112.562 s after imports, 2.951 GB peak allocated GPU,
4.333 GB end RSS; hard fit/calibration/screen completed in 3.250 s before
model load. Router: 40,354,576 bytes, SHA256
`42cb36188c998cbd20562cb76bcee733ddb4fb6509832f6f8260cc72b6ea287d`.
Byte ledger remains unchanged and hypothetical; native feature reuse,
floating-point fidelity, traffic and accepted rate are unverified.
[Raw result](meth218_hard_parent_bias_result.json) SHA256
`e64e5227e8de459d49f08d5efb971a162a1dba8223e965c3887c6f740162c4de`.
Session 85910 completed with exit 0; no project inference process remains.

Command:
```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth218_hard_parent_bias.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth218_hard_parent_bias_result.json --router results/native_expert_scaling/meth218_hard_parent_bias_router.npz
```

Next pipeline uncertainty: the saved compact core's failed BF16-framework
quality does not establish the behavior of the existing native FP32
arithmetic path. Freeze a consumed-source development comparison that
loads all saved tensors, explicitly reconstructs Q8 weights in FP32 and
uses the exact BF16 effective factor-bank values, with matched BF16 and
FP32 controls. Preserve existing quality limits; this is a changed
arithmetic representation, not permission to relabel METH-214 as passing.
Only a development pass permits separately frozen fresh adjudication;
native kernel parity/rate must subsequently match that representation.
This connects compact-core work to eventual useful conditional capacity,
while full larger-n, multi-family and 10B/100B requirements remain intact.
