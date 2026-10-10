# Source readout margins: positive-gap drift, not exact teacher ties

10 October2026. Stored diagnosis COMPLETE/exit0. Original
SOURCE_STATE_LABEL_ALIGNMENT_FAIL retained. No changed quality criterion,
codec fit, source history, full-head contraction, optimizer or native call.

## Observations

The full source-readout audit confirmed the sole failed case is
broad_dev_everyday_conversations_036:3/53 predicted IDs differ in both readouts.
[Prospective diagnostic protocol](SOURCE_READOUT_MARGIN_PROTOCOL_20261010.md).
The bounded diagnosis reads that case's saved teacher/BF16/F64 score streams.
It recomputes all53 KL values, top sets and argmaxes; label indices below are
zero-based and positions refer to the retained full-prefill history.

| Label / position | Teacher ID -> predicted ID | Teacher gap to contender | BF16 pairwise perturbation | F64 pairwise perturbation |
|---|---|---:|---:|---:|
|21 /168|2632 ->3942|.125|.171875|.161864704161|
|23 /170|4496 ->15155|.015625|.0625|.052653367697|
|33 /180|9330 ->228|.140625|.1796875|.186659718907|

All three teacher argmaxes are unique. Both readouts choose the same three
different contenders, each with a unique argmax too. Therefore the verdict is
**POSITIVE_MARGIN_DRIFT_PRESENT**: exact cached teacher top-tie breaking does
not explain these three mismatches. Source-head F64 accumulation on the same
savedBF16 features also does not fix them. This does not isolate whether hidden
states, normalization rounding, remaining finite precision or an implementation
difference produced the drift.

Independent logaddexp KL versus retained primary values differs by at most
2.21703e-13. Previous complete source audit already independently verified all
48 cases' metrics/counts and source scalar-head witnesses. This new diagnosis
has input/output sealing and algebraic self-checks; it does not claim a separate
full independent audit of its derived margins.

## Algebra and temporal control

For q cached teacher logits, p reconstructed source logits and delta=p-q,
each wrong contender b versus teacher winner a obeys exactly

    p_b-p_a = (delta_b-delta_a) - (q_a-q_b).

The measured perturbations exceed the teacher gaps. The minimal constant-shift
invariant Linf error epsilon=(max(delta)-min(delta))/2 gives the sufficient
unique-argmax condition teacher gap>2epsilon. It certifies30/53BF16 and32/53F64
positions; none is a mismatch. Non-certification is not failure: the bound is
sufficient, conservative and sensitive to full-vocabulary tails. Its inability
to certify the three disagreements is a consistency witness, not an explanation
by itself or a new loss estimate.

Off-by-one target control,52 comparisons per direction: BF16 meanKL14.2531781
for previous-label pairing/11.4382955 for next-label pairing; F6414.2521542/
11.4407049. Correct pairing caseKL is only.000725/.000694. This rejects the
simple uniform +/-1 shift as a better explanation for this case; it does not
certify every possible capture/mapping mechanism or all cases.

Let f be the actual final norm, s the head scalar and W the source head.
The full-prefill versus cached output discrepancy decomposes in real arithmetic as

    delta = s W [f(h_prefill)-f(h_cached)] + numerical/operator discrepancy.

Only h_prefill is retained here. Cached h is not observed by this diagnostic;
the first term cannot be quantified or blamed by inference alone. Inspect the
two actual execution/capture paths, then freeze a bounded cached final-state
capture paired with its own logits if necessary. This is a new intermediate
observable, not an unchanged history-training dose.

## Custody and measured cost

Freeze`008979a787ffad96ae09923ec25360b27636d457`.
[Binding](source_readout_margin_binding_20261010.json) SHA256
`1d0dafb5b2d7545e33a2f0227c0e1c20c790f4857a91f5be1d015486375e4ac0`;
30 inputs77,864,838B, all checked before/after. Single-case source teacher and
two score streams total41,681,532B. Qualified NumPy/psutil/Python/helper
identities inherited, no GPU/model runtime used.

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_readout_margin.py --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_readout_margin_binding_20261010.json --binding-sha 1d0dafb5b2d7545e33a2f0227c0e1c20c790f4857a91f5be1d015486375e4ac0 --freeze 008979a787ffad96ae09923ec25360b27636d457 --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_readout_margin_result_20261010.json
```

Held launcher2400/worker25764 exit0 and gone; worker created11:35:19.281763+02:00.
Held2.813s, OS80,109,568+28,880,896=108,990,464B<=512MiB/60s. First faults none.
[Result](source_readout_margin_result_20261010.json),68,631B, SHA256
`8adba9f611881bbc44cd202a49b7b32f87250baefbc00d8a6ed0d88fa13eb69f`;
receipt/log retained. Output/log caps met. Source history/head/optimizer/native/
RESERVED calls0. No owned process remains.

## Next decision

[Cached-state investigation](SOURCE_CACHED_FINAL_STATE_NEXT_20261010.md): inspect
exact generation/full-prefill capture semantics before selecting a new call.
No threshold relaxation, new oracle scale, width/rank ladder or codec fit.
If a new cached-state observation is required, first specify calls, input/label
custody, gates, budget/stops and preserved prior outcomes. Final compact own-history
chatbot, accepted same-artifact50/useful-n/DRAM/family requirements remain open.
