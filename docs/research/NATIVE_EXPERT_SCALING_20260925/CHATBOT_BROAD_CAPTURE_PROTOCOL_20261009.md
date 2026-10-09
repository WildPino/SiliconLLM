# New broad donor supervision: fixed observation protocol

9 October2026. Preregistered before source execution. Separate versioned worker
and schemas; old160 source observations remain unchanged and are not repeated.
This resolves available supervision/response-length/capture-cost uncertainty
for a controlled data-coverage transfer pilot into the original engine target.
It does not test a new model architecture or claim source-answer correctness.

## Reused inputs and changed variable

Pinned Falcon-H1-1.5B-Instruct revision80ebc50d7799a440b96c93bb6686a3924a09b0cb,
original source package/tokenizer/config/BOS17/V65537/BOTH EOS11+228;
[624-prefix adoption](CHATBOT_BROAD_DATA_RESULT_20261009.md). Fixed post-adoption
cohort SHA25614050c2ce63c39b205bc007abc8b41933747ffb594b530ff3bf7375fa30b34e4:
48 cases,12 sources times2 FIT/2 DEV,25th/75th eligible length ranks,ID tie break.
Actual input34-1252/max256 new IDs/2304 total. No model-loss/answer filtering.
Public prior assistant turns/external references retain public provenance;
source outputs supply new labels.104 new/64 old RESERVED stay unqueried.
Longalign8186-20874 remains intact/deferred to a separately costed context stage.

## Computation, budget and stop

Original source BF16/eager/greedy/cache/TF32off/no optional kernels,local3060,
six CPU threads,seed0. Verify all canonical source template/IDs against adopted
cases before loading weights. Charge source loading/prefill/generation/full
V65537 logits/serialization/guard/input hashes and held worker through exit.
No learner forward/update/native call or T4 allocation in this family.

ONE family<=2700s/8GiB OS/10GiB GPU allocated/11GiB reserved/2GiB output,
30s worker reserve. This replaces the earlier proposed1800s BEFORE observations:
the upper bound12288 labels is2.138 times the old5746-label/861.8s family,about
1843s at that historical aggregate cost,with longer new prefills and input/hash
overhead requiring additional allowance. This is a budget inference,not measured
new throughput or a guarantee of completion. Full packets are bounded by
48*256*65537*2=1,610,637,312B plus metadata. Stop on any input/precision/shape/
finite/argmax/EOS/position/resource fault; do not raise caps after observing it.

## Durable results and adoption

Persist every completed generation's BF16 packet and full case JSON BEFORE
transport/resource assertions. Preserve all EOS/length-cap partials,without
bad-answer filtering; length256 partials are prefix supervision,not complete
responses. Save firstfault/active case/durable completion IDs/costs. No automatic
restart or regeneration of completed cases. Interrupted packets are retained
with the original failed family identity and need separately declared repair.

Admission requires all48 completed,exact cases/IDs/template,finite full65537
rows,lossless BF16 producer flag,greedy argmax matches IDs,BOTH-EOS stop/no earlier
EOS,correct teacher-forcing positions,all source/foreign input hashes and caps.
All result/packet extents,commands/PIDs/exits/peaks are retained. A separately
frozen saved-only adoption must verify all BF16 bits/argmax/IDs/EOS/positions
without new source calls before fitting. Shared Rust tokenizer is not an
independent BPE implementation; selected runtime hashes are not a full DLL tree.

## Decision after observations

Use actual FIT/DEV labels/lengths/EOS fraction and memory/timing to price ONE
whole-output pilot from audited checkpoint286/Adam294. Keep compact SSM/SWA/
original packed ternary LUT geometry as the data-coverage control. Freeze
optimization/retention/absolute-domain/own-history/native/export stops separately.
Two cases per split/source are calibration and consumed development,not a
representative capability benchmark. Final fresh donor-relative quality and
accepted50 on the same artefact,large useful n/CPU routing mass/physical DRAM/
other families/scales remain missing. Long offline T4 must have measured
feasibility and communicated hours/checkpoint/plateau stops before starting.
