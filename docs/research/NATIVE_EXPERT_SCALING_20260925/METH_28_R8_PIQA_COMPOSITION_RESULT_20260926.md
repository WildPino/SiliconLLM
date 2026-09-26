# METH-28: stored R8 composition retains the PIQA task gate

Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth28_r8_piqa_composition.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth28_r8_piqa_composition_result.json`.
The [prospective protocol](METH_28_R8_PIQA_COMPOSITION_PROTOCOL_20260926.md)
fixed all 1,838 PIQA validation items via the prior METH-21 token
manifest, SHA-256
`58acfd5882ae5d7e155bdd0ae826728aba510dc6ef4f599b7b807161a498531d`.
The same model instance reproduced the old BF16 donor's 1,298
correct answers before loading the 496,122,224-byte core from its
stored int8 codes/scales. All 169 matrices, 121 control vectors,
metadata and tied head passed load checks. The separate E128 adapter
was then enabled without changing the core or task scorer.

| Mean-suffix-NLL PIQA choice | Correct / 1,838 | Accuracy | Δ vs BF16 donor | Paired lower CI95 |
|---|---:|---:|---:|---:|
| Original BF16 donor | 1,298 | 70.620% | — | — |
| Stored-R8 donor | 1,290 | 70.185% | −0.435 points | −0.925 points |
| Stored-R8 + E128 adapter | 1,281 | 69.695% | **−0.925 points** | **−1.687 points** |

The adapter is −0.490 points relative to the R8 donor. Against the
original donor, R8 donor has 18 lost and 10 gained correct items;
R8+adapter has 43 lost and 26 gained. The total-NLL sensitivity
accuracies are 70.348%, 70.076% and 69.859%, respectively. Both
new arms clear the frozen ≥−2-point difference and ≥−5-point lower
confidence gates. The RTX 3060 run took 570.4 s, with 2.283 GB peak
allocated GPU memory and 3.215 GB final RSS. All item-level choices
and NLLs are in the [result JSON](meth28_r8_piqa_composition_result.json).

This is task retention for one already inspected English multiple
choice set. It does not erase [METH-27's](METH_27_R8_FRESH_GENERATION_RESULT_20260926.md)
failed relative and absolute generation gates. PyTorch expanded the
stored int8 codes to BF16 during scoring; native C parity, CPU speed,
large-E routing and cross-family transfer remain unverified.

**Decision:** retain the compact artifact as a task/document-quality
research candidate, but hold native promotion until generation is
repaired on new data and new fixed prompts.
