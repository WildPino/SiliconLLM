# METH-270: prospective fixed private-input coverage/cost test

## Changed variable and decision

269 rejects a simple BF16-node-only remedy: energy improves but mean-state
error worsens. Exact-source BF16 features lower mean-state error80.4% at
unchanged readout,so test a bounded increase in exact input-row coverage.
This is a source-core fidelity/cost question,not a new expert-n result or
a reopened local child/readout fit. No270 observation or artifact exists
at this freeze. Implement and freeze the apparatus before executing.

Change only private source input rows32->128 per layer. Keep the existing
source-only FP64 gate/up coefficient-error ranking,ties by unit ID,and
sorted physical IDs. Require old32 IDs to be an exact subset of the128.
Use exact original BF16 gate/up rows at the additional slots. SharedQ8
fields,LUT513/down escapes/residual rank32/biases,arithmetic and all existing
learned unique-bank A/B/router/key/alias fields stay fixed. Do not replace
this with post-observation64/256 counts or an activation-fit selection.

Bind269 raw SHA
`95f905587e74de27b8d7661f9b4a0160af824cdc331c9cf34c8aef3df5b5cc1f`,
original259/export/helpers/source hashes,252 fixture/source segments,
253 native controls and125256states/layer. All inputs are consumed controls.
The new private rows add96*896*2weights*2bytes*24layers=8,257,536bytes
plus4,608 ID bytes. Account these actual extra reads/work; this is
additional source precision,not independently learned conditional capacity.

## Preregistered component gates

Require exact unchanged-field hashes and source row identity,all old private
IDs retained,all6144 outputs finite. Relative to the actual source BF16
MLP,mean per-state relative squared error across6144states must be at most
0.85 of the frozen269 existing-FFN value .0001972897928984215. Maximum
layer energy-normalized squared error must not exceed existing269 maximum
.00030542892636731267. Both criteria are prospective,not inferred quality.
Failure closes this fixed128-row coverage recipe before complete candidate
assembly or any new-source scoring.

If fidelity passes, build the same253 split-workshare kernel with private
count128 and the exact new fixture. No other operator optimization.
Use all original384native output vectors for reference/C comparison with
the original252/253 relative-error limits,then the original three24-layer
component timing sweeps,six threads. Require median <=10ms. No GPU/model
work overlaps timing; explicitly synchronize/finish component scoring.
Stop and preserve native fidelity/cost failure; no speed-threshold widening.

A component pass licenses assembly of a separately versioned full archive
and consumed full-model regression; it cannot promote259 or satisfy the
goal. Freeze further generation/task/anonymous-semantic and excluded-source
checks before that changed complete candidate is scored. Native whole-model
quality/cache/route-LUT/real DRAM/accepted>=50tok/s remain mandatory. Larger
dense/sparse donors and useful RAM-scale n need separate applicable variants.

## Budget and implementation resumption

Local RTX3060/six threads,10min for source-only row selection/scoring/export;
20GiB RSS/10.5GiB GPU,3GiB output budget. Native compiler/check/timing up to
5min separately,no T4/download. Preserve partial/failure evidence. First
implement source-row fixture export and BF16-referenced component checks;
then freeze code before any270 observation. Do not immediately build a
new full model without the fidelity/native-cost prerequisite pass.
