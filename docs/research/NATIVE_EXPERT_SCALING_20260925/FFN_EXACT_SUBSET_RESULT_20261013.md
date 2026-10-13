# FIT-selected exact nonlinear FFN remainder: qualified local failure

13 October2026. **COMPLETE/NUMERIC AUDIT PASS/NO ADMITTED SELECTED CANDIDATE.**
Goal ACTIVE/INCOMPLETE. No physical ternary bank or whole-model substitution.

## Experiment and evidence

[Protocol](FFN_EXACT_SUBSET_PROTOCOL_20261013.md),source18e7da7/bindinge9a9fa5.
[Producer](ffn_exact_subset_result_20261013.json) uses actual own pre-FFN inputs,
all48 histories/sites0,12,23/existing64 FIT cells and full2048 outputs.
FIT-only per-cell energy ranks32/128/512/2048 source neurons;both exact omitted
terms retained together. All4608 is the identity control,not a reduced-cost
candidate. No source model/GPU/RESERVED/optimizer or new centers.

[Full audit V2](ffn_exact_subset_adjudication_v2_20261013.json),sourcea99c170/
binding44902e9:all720 predictions/routing IDs/masses/case/domain/selection
metadata and all144 source-proxy controls PASS. Original FIT scores/ranks
replayed EXACTLY;independent score algebra max relative3.861e-15/down column
energy4.365e-15/F32 prediction2.663e-8. All-neuron identity max6.890e-16,
proxy-versus-actual-BF max.004774,within the frozen1% numerical budget.

## Complete local output curves

Case-weighted relative output errors against actual source BF,not whole-model
quality. All counts,including failures and the full identity control,retained.

| Site | Neurons | FIT mean | FIT worst | DEV mean | DEV worst | FIT eligible | DEV eligible |
|---|---:|---:|---:|---:|---:|---|---|
|0|32|0.529070|0.607525|0.575733|0.646370|no|no|
|0|128|0.425134|0.468290|0.518641|0.613602|no|no|
|0|512|0.260582|0.331113|0.400908|0.541574|no|no|
|0|2048|0.108113|0.130057|0.243356|0.398866|no|no|
|0|4608|0.004141|0.004290|0.004123|0.004256|yes|yes|
|12|32|0.754475|0.812403|0.883059|0.997096|no|no|
|12|128|0.663045|0.725387|0.820818|0.940590|no|no|
|12|512|0.504582|0.561074|0.688920|0.817862|no|no|
|12|2048|0.244910|0.282628|0.404537|0.506711|no|no|
|12|4608|0.004672|0.004773|0.004669|0.004754|yes|yes|
|23|32|0.100716|0.184887|0.113219|0.181237|no|no|
|23|128|0.085335|0.164889|0.101091|0.170463|yes|no|
|23|512|0.059268|0.123678|0.077901|0.144107|yes|yes|
|23|2048|0.020574|0.044203|0.036098|0.080473|yes|yes|
|23|4608|0.003113|0.003207|0.003107|0.003210|yes|yes|

The frozen rule selects the smallest FIT-passing noncontrol count. Sites0/12
select NONE. Site23 selects128,then fails DEV mean.101090835>.10. Counts512/
2048 at23 pass local FIT/DEV metrics,but are NOT substituted for128 by looking
at DEV. No threshold rounding or relaxed gate. This experiment does not yield
an admitted candidate across the three sites,let alone all24/model quality.

## Audit fault and identifiable selection weakness

Original [audit terminal](ffn_exact_subset_adjudication_20261013.terminal.json)
exit1/106.875s/OS960,466,944B is preserved. [Diagnostic](ffn_exact_subset_rank_discrepancy_20261013.json)
repeats the failed frame and saves every score/rank/count (7,080,504B on E).
Site0 counts agree and independent score difference is3.291e-15 relative.
The11 differing cells have ten singleton FIT supports plus one24-row support.
Producer scores are EXACTLY ZERO there;its declared stable neuron-index
spareggio is correct. Equivalent independent floating-point algebra produces
scores<=5.774e-29 and a different ordering. The original audit incorrectly
required that ordering to be bit-identical across real-algebra associations.

[Repair](FFN_EXACT_SUBSET_AUDIT_REPAIR_20261013.md) adds separate literal score/
rank replay alongside independent numeric score/prediction verification.
All candidate values/IDs and quality gates remain unchanged;no cell skipped.
Every value in the diagnostic snapshot is replayed exactly at site0. No such
rank discrepancy at12/23. This also identifies a real selection weakness:
FIT cannot distinguish neuron subsets in a null-remainder cell. The source
index fallback carries no learned importance for off-anchor inputs.

These11 cells cover34/11148 FIT and26/11399 DEV tokens at0 (all24 cases have
at least one such token). This count does not bound their output energy or
causal importance;it is not a complete explanation of the large quality loss.
The diagnostic CLI closed0 via tool92924 and preserved its code/report/array
snapshot;it did not persist a separate PID/create-time receipt. V2 replay and
full hash custody qualify the snapshot without inventing retrospective PID data.

## Costs and closure

Preparation12.828s/OS42,618,880B,producer988.953s/OS1,198,706,688B,
full V2 auditor424.859s/OS1,418,113,024B. All three receipts exit0/owned
PID-create pairs absent;original failed audit also absent and preserved.
[Closure](ffn_exact_subset_session_closure_v2_20261013.json) full source/input/
output/protected hashes PASS. E family723 files/2,785,911,606B<3GiB,plus
separate7.08MB rank diagnostic. Closure command exit0/16.610s/OS30,208,000B.
Original engine.c and foreign tracked hashes unchanged.

These CPU costs are experiment/conversion diagnostics,not batch1 inference
latency. Candidate evaluation still uses implicit source operators;no physical
full-J bank or reduced active-operator timing. Logical coefficient saving is
not measured DRAM or TPS. General source-family/scale transfer remains unproven.

## Exact resumption toward the goal

Implement/freeze [per-token remainder importance oracle](FFN_REMAINDER_SELECTION_ORACLE_NEXT_20261013.md)
and its complete auditor/binding. Same64 cells/full outputs/32,128,512,2048
budgets/control,but selected neuron IDs can vary at each token. This separates
fixed-cell aggregation loss from observed per-token sparse coverage for the
specified energy heuristic. It is neither optimal subset search nor a universal
lower bound. Computing its IDs uses ALL source projections/nonlinearity and
is explicitly not a cheap deployable router or candidate.

A strong improvement would justify source-informed partitions or a learned
cheap CPU predictor with fixed active width/more meaningful choices. Otherwise
inspect projected covariance/sign changes/joint selection before asserting
that more experts cannot help. Compact shared core,ternary export,actual CPU
IDs AND masses/DRAM,own-model quality,multiple donors and same useful50 remain
open. RESERVED remains unopened. No GPU/T4/owned job remains active.
