# METH-255: isolated source fidelity does not qualify learned composition

Frozen f28780f, session68832 exits0. All source/native/capture/route,
private row/feature byte,16 original learned-parent coefficient and exact
fit-score controls pass. Same source-only32 feature rows, no new fit/unit
selection/validation/native timing. Four actual functions on all512 fit
windows show:

| Readout | Shared Q8 features, SSE/energy | Fixed private32 features, SSE/energy |
|---|---:|---:|
| Learned METH-247 E16 | .00004587007460894334 | .00005872514958692855 |
| Original source245 mixed+BF16rank32 | .0002270084576060162 | .00020905379678066324 |

Learned private composition loses **28.02%** relative to learned shared.
Coherent source private gains only **7.91%**, below10%, and remains4.56x
the learned-shared fit error. Absolute1% passes; both prospective source
coherence gates fail. Do not construct a source-reset unit pair from this
failed diagnosis or repeat254's unchanged learned-readout promotion.

METH-252/253's81.03% local fidelity recovery remains true on its pinned
source-prefix vectors/reference FP32 FFN. It does not transfer to these
captured BF16 target windows or adapted readouts. The protocol prevents
that unsupported composition claim. This comparison establishes a fit
composition gap, not a complete causal attribution, independent quality
result, or global rejection of private source features.

Runtime4.922s, RSS2,067,894,272bytes, GPU peak2,255,497,216;
local RTX3060/six threads, no T4/download.
[Raw result](meth255_feature_readout_composition_result.json) SHA256
`243f748287a50b88c4aeaeecaa362f7f8e7f742271c692a495d5497ac17691a2`.
No new model/bank was saved.

Change the private response/learning coupling. A proposed nonlinear
source-difference dictionary learns bounded amplitudes while preserving
the complete actual adapted parent. This differs from forcing full BF16
feature replacements or refitting an independent high-dimensional output
matrix. Continuous pilot first, then a separately frozen native blending
operator/bank only if useful count qualifies. Full end-to-end transfer
quality/rate and donor family/scale remain required.
