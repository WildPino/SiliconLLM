# Useful expert count, RAM and actual CPU routing: current boundary

## What the existing complete artifact contains

The user's main scaling requirement remains useful conditional capacity
growing with available RAM, approximately tenfold n for a tenfold donor
size when conversion is applicable, while active cost and routing complexity
remain affordable and quality is retained. This is a hypothesis to validate.

Original276 preserves learned centered1280 route labels/layer,24 layers,
four selected parents with one child each,rank8 parent features. The
[258 audit](METH_258_EFFECTIVE_BANK_DIVERSITY_RESULT_20261002.md) identifies
30,556 distinct effective parameter functions and164 aliases in30,720
addresses. These counts describe stored functions, not independent useful
capabilities or transferred capacity from a10B/100B donor. Full dense source
FFNs still execute all4864 features/layer. Neither the1.329GB archive nor
GPU quality passes establish constant-active-cost large-donor conversion.

The original C loader/operator in `meth285_model_operator.h` fixes parent
shape128, child shape10, feature width32, parent projection64 and factor
rank8. Axis selection arrays admit at most16 entries. Merely allocating
more RAM does not make this executable support arbitrary n. A future
configurable schema/loader/router is required, with new shape/range/control
checks and whole-artifact quality and cost. METH-295 qualifies current
native usefulness first, not that larger schema.

## Source-derived operand accounting, not a bandwidth benchmark

For this fixed geometry, the actual `cc_conditional` loops read per layer:
router235,520bytes, child projection114,688bytes, selected child keys
4*C*32*4bytes, selected parent A57,344bytes, selected B57,344bytes and
four int32 aliases16bytes. These are logical operand bytes; cache lines,
reuse, overfetch, DRAM residency and physical traffic require measurement.
They exclude source FFN, attention/head/KV and all runtime glue.

Holding128 parents, four selections and every projection/rank dimension
fixed while increasing only children C gives the following arithmetic:

| Children/parent C | Route labels/layer | Maximum24-layer BF16 B storage | All stored child keys | Logical selected bank operand bytes/token | Child score dot terms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1,280 | 440,401,920 | 3,932,160 | 11,280,768 | 30,720 |
| 100 | 12,800 | 4,404,019,200 | 39,321,600 | 12,386,688 | 307,200 |
| 1,000 | 128,000 | 44,040,192,000 | 393,216,000 | 23,445,888 | 3,072,000 |

Storage formulas: B=24*128*C*896*8*2, keys=24*128*C*32*4.
Actual current unique B storage is438,050,816bytes; shared parent A is
44,040,192bytes. The larger rows are unimplemented hypothetical geometries,
not trained pools, achievable capacity claims or performance predictions.
The chosen child search is linear in C for each selected parent. Fixed top-k
alone does not keep routing cost fixed. Changing parent-axis width or using
a deeper child index is another design variable requiring separate learning
and correctness evidence; no logarithmic-time/cache assumption is made.

## Prior decisive evidence and what not to repeat

- [176 large-bank independent prediction](METH_176_LONG_FRESH_PREDICTION_RESULT_20260930.md):
  trained E12,800 pooled gain .00011440BPB, but bootstrap lower gain
  -.00004190 fails the frozen positive-gain gate. Extra useful specialists
  were not demonstrated. Do not reinterpret nominal count as success.
- [179 child-route intervention](METH_179_CONTENT_ROUTE_FUNCTION_RESULT_20260930.md):
  exact route does not robustly outperform child rotations; mechanism
  gates fail. Extending the same fixed-route/shared-base training duration
  has no supported next-run justification. New specialist signal or learned
  child identity must be chosen on training/development alone.
- [198](METH_198_PAIRED_LUT_SCALING_RESULT_20260930.md),
  [199](METH_199_BOUND_LUT_SCALING_RESULT_20260930.md) and
  [200](METH_200_SINGLE_THREAD_LUT_SCALING_RESULT_20260930.md) synthetic selected
  LUT-pool timings fail repeatability. They establish neither a reliable
  tenfold scaling ratio nor useful learned capacity/full-model>=50tok/s.
  No retiming without a concrete apparatus change explaining the drift.

## Required future comparison

Once a complete pretrained-to-native path retains actual quality, add new
learned conditional capacity with a changed specialist-learning signal or
router. Fix/report core, active selections, representation, training-data
budget and per-expert exposure; train a matched smaller-pool control.
Use untouched sources for whole-model prediction/generation/tasks and
route displacement/rotation controls to establish usefulness of identity.
Measure actual parent/child selection, LUT, chosen expert reads, cache/DRAM
and total accepted decode at both n values on the same declared CPU.
Report resident storage and actual active/read traffic separately. A larger
pool that adds negligible useful signal or unbounded search is not the goal.
Reuse GigaChat donor-adaptation source/fidelity assets only for a specified
compact transformation; its generic port remains paused. Second-family and
real10B/100B transfer still need their own retained-capacity evidence.
