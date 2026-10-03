# METH-331: actual router error decomposition before another repair

Prospective after complete330 preserves probability FAIL1.63913e-6 versus1e-6;
stable RMS lowers pooled logit error but does not close original guard. Do not
infer naive variance sum was the sole cause. Resolve actual upstream-state
versus same-input classifier/softmax contributions, not blind numeric tuning.

New source COPY330 changes NO arithmetic: add trace of actual F32 normalized
router input and all256 raw classifier logits after every native sparse token.
Source exact one trace-function/insertion difference checked before run; ALL
native encoder/decoder/logit/route output bytes must EXACT original330 percase.
Same original weights/config/spec/tokenizer/cases/forced IDs/thread/compiler,
original reference source and officialCPU1thread. No gates or candidate change.
New opt-in engine SILICON_SWITCH_ROUTER_TRACE, old bodies/binaries retained.

All acquired source files freshly hashed; official full META+weights_only/mmap
source loaded, confirmed ties restored. Capture original normalized router inputs,
raw logits/probabilities/choices from same full source engineering calls. Exact
row order/count and original choice required. Same original330 max probability
error must be reproduced EXACT. Preserve complete reference states/logits/routes/
router input/logit arrays for later numerical apparatus reuse, not quality data.

For EACH actual row, replay official F32 linear with native normalized inputs
at SAME original call shape(batch1/full encoder call or1decoder token), then
official softmax. Signed observed error decomposes exactly into same-input
classifier+softmax and upstream-input error. Require identity equality, otherwise
invalid diagnostic. Also compute independent F64 dots+softmax on native and
original F32 input/source weights, and F64 softmax on original native raw logits.
Record max raw-logit errors to exact-input F64, softmax rounding, norminputL2,
bank/token/ordinal/actualselected probabilities. Counterfactual F64 router error
is diagnostic: cannot assume it composes into whole-model repair or matches
official F32 backend everywhere. No numerical promotion threshold here.

20min main AFTER imports, checked combined32GiB, compiler120s, native250ms
sampling. No GPU/timing benchmark, no source overwrite or data selection. All
partial failures preserved. Driver source/trace/wrapper/protocol frozen before
observations. Preserve329/330 guards; select next specific variable only from
measured decomposition, new prospective full composition validation mandatory.

Command:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth331_switch_router_diagnostic.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth331_switch_router_diagnostic_result.json
```

No final quality/precision/LUT/accepted>=50 or larger-n usefulness evidence.
