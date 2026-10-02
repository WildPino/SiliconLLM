# METH-259: complete saved core and unique bank conserve original execution

The [frozen unique-storage export](METH_259_UNIQUE_BANK_CORE_PROTOCOL_20261002.md)
executes once afterfe1fedf, session72563 exits0. All apparatus gates pass:

- All361 qualified source FFN segments and220 non-FFN/head-proposal fields
  unchanged;725 finite tensor fields readback exactly.
- Every original1280 leaf maps to its original effective BF16 B. Dictionary
  counts/signatures match258; mappings in range and surjective. No copied
  conditional padding, new training or altered routing/scores.
- All6144 original stored source-state vectors preserve FP32/BF16 source
  FFN outputs and original routed/gated conditional BF16 outputs bitwise.
- Complete loader reads config/weights/banks/maps from this one archive,
  removes72 random constructor FFNs before execution, supplies218 other
  model parameters, consumes every725 field and preserves tied head pointer.
  No original donor weight or training-checkpoint fallback by the loader.

Actual artifact:
`results/native_expert_scaling/meth259_qwen05b_unique_source_core.safetensors`,
**1,321,032,556bytes**,1,320,959,748byte tensor payload,
SHA `3c0949fb2b7f99dc887c6aec1041aa162b35801939349966d18e52b8807a6f71`.
It includes shared source core and real already learned conditional banks:
1243..1280 unique composite parameter functions per layer,30,556 total,
plus1280 original route labels/layer with164 total aliases. Alias maps add
122,880bytes; no claim of1280 independent functions in every layer. Effective
BF16 behavior preserved, not unrounded FP32 training coefficient storage.

[Raw export/loader record](meth259_unique_bank_core_export_result.json) SHA
`39f796651522a2af907f4123756b9c7ab1e838057debbb142db5c1e02765b663`.
Local RTX3060/six threads,82.750s after imports,3,947,663,360byte end RSS,
3,282,810,368byte peak allocated GPU. No active job after exit. This changes
storage and core execution apparatus; no target/logit/document/generation/
task score was read. METH-257's all10-distinct export remains stopped.

## Next full-model gates

Freeze METH-260 on the original consumed24-source development manifest.
Compare full-head document BPB and donor-top1 agreement against original
BF16 donor and BF16 centered E1280; replay old control scores exactly.
Keep pooled/category BPB+.01/+.02, top1 loss1/2 percentage points versus
BF16 E1280 and finite K64 prompt inclusion/exact-rerank gates. Failure
closes this fixed complete candidate before new-source/native promotion.
Pass only licenses genuinely new independent source selection, prediction,
generation/tasks and native full-model/alias lookup/LUT/DRAM/accepted-rate.

The core still stores all source-derived dense FFN rows; source preservation
and learned residual bank size do not demonstrate new semantic capacity
extracted from a100B donor. Qualified2539.318ms is a component, not a
same-artifact full model rate. RAM-only arbitrary n, preserved large-n
quality,>=50 accepted batch1tok/s and second-family/10B/100B stay open.
