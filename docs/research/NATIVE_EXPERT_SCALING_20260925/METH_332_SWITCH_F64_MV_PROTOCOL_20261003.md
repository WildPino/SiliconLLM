# METH-332: stable matrix-vector accumulation, same complete source guards

Prospective after331 actual unchanged330 diagnostic. Worst source row first
sparse encoder has -1.63913e-6 probability error = -7.45058e-7 same-input
classifier/softmax plus -8.94070e-7 upstream input contribution. F64 router
alone at current native input counterfactual still1.50269e-6 FAIL. Choose
one shared matrix-vector precision variable addressing classifier AND prior
upstream projections; no isolated-router tweak or tolerance waiver.

New C copy330, exact structural difference checked: add AVX2 F64 dot from
original F32 operands, use it for all matrix-vector rows, then output rounded
F32. Prior330 norm retained, attention QK/value operations unchanged F32.
No weights/bank/data/gating/capacity/config change, no all-F64 activations.
Original329/330 failures and331 diagnostic remain immutable. New opt-in
SILICON_SWITCH_F64_MV_SOURCE; default engine unchanged. F64 is an accuracy
reference variant, not promised fast representation or compact LUT method.

Freeze code/protocol BEFORE observations. Same330 complete original source
files freshly hashed, all6392 new native mapped tensor hashes EXACT327,
same official original4.57.6 reference/META+weights_only/mmap CPU full model,
original tokenizer, original two engineering sources/forced IDs; ALL original
1e-4 pooled ANDevery-state errors, exact choices/capacity/greedy, probability
<=1e-6 unchanged, zero-head fault>1e-4. Both exact328 Tiny weights/source IDs/
four cached steps/caps1/64 recreated, all original Tiny contracts required.
Matrix output F32 conversion/per-layer nonlinear composition must be measured,
not inferred from independent dot accuracy. Source whole quality still missing.

Same20min main AFTER imports, combined checked32GiB/120s compiler, CPU1thread,
no fast math/FPcontractoff, no GPU or concurrent timing. Same complete record
and exception preservation as330. No gate rescue or candidate reordering.
If PASS: eligible separately compact precision/LUT whole quality and actual
cost; if FAIL: preserve, close unchanged variant and use diagnostics before
another action. Final source-relative heldout/generation/tasks/>=50 accepted
SAMEartifact/useful n and cross-family/~100B are unchanged missing obligations.

Command:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth332_switch_full_source.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth332_switch_full_source_result.json
```
