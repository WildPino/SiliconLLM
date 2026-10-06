# PQT-005-R1: original expert functions, strong fitted-set gains without preservation

5 October 2026. **No ternary arm promoted. Research goal incomplete.**
[Protocol](PQT_005_PROTOCOL.md), [R1 repair](PQT_005_R1_PROTOCOL.md),
[adjudication](PQT_005_R1_ADJUDICATION.json),
[retention](PQT_005_R1_RETENTION.json),
[repair reproducibility](PQT_005_R1_REPAIR_VERIFICATION.json).

## Decision and scope

Actual original F32 Wi and Wo of Switch decoder block 11 experts 8 and 105
were converted together, including their ReLU function. Source references
were recomputed from original pretrained values, never from inherited I8
function outputs. Each arm retains the same artifact for fidelity, storage
and future native execution. None meets the prospectively fixed <=1% RMS
function-preservation gates on both experts. I8 passes every function gate
for both experts but uses 50.1628% of FP16 bytes, exceeding the 35% capacity
budget. I4 uses 28.125% but fails all nonzero function-fidelity gates.

This is a scoped negative finding about six fixed methods, two selected
experts and these inputs. It is not a proof against every ternary method,
complete-model conversion or a verdict on semantic task quality. The input
distribution is inherited from the parent's I8 core. All 95 role-1 observations
are consumed development evidence, not new natural validation or states from
the original float model. Gaussian and perturbed probes are fresh conditional
function probes; Gaussian coordinates deliberately lack natural correlations.

## All arms and experts

Relative RMS in percent; lower is better, prospective threshold **<=1% on each
raw-development/Gaussian/perturbed group for each expert**. Calibration is fitted
evidence, not a promotion gate. Zero response is separately gated.

| Expert | Arm | Calibration | Consumed raw | Gaussian | Perturbed | Zero/calibration norm | Payload bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | D | 37.7404% | 38.9692% | 42.7703% | 37.4063% | 0 | 1,474,560 |
| 8 | DR | 1.1636% | 42.1891% | 53.5319% | 3.1142% | 0 | 1,474,560 |
| 8 | PR-B | 0.8138% | 40.1150% | 51.9376% | 2.6348% | 0.029939 | 1,489,920 |
| 8 | FF | 13.2063% | 41.3898% | 52.5793% | 14.6416% | 0.030486 | 1,489,920 |
| 8 | I4 | 8.3026% | 8.9403% | 10.9851% | 8.2894% | 0 | 2,654,208 |
| 8 | I8 | 0.7033% | 0.7472% | 0.9372% | 0.7057% | 0 | 4,733,952 |
| 105 | D | 38.0035% | 38.8403% | 42.1127% | 37.6120% | 0 | 1,474,560 |
| 105 | DR | 2.0843% | 50.8314% | 57.7629% | 3.7620% | 0 | 1,474,560 |
| 105 | PR-B | 1.7598% | 47.3899% | 54.5719% | 3.4004% | 0.062286 | 1,489,920 |
| 105 | FF | 13.2387% | 50.5722% | 56.5476% | 11.9996% | 0.065414 | 1,489,920 |
| 105 | I4 | 9.1051% | 8.3775% | 11.4742% | 8.8953% | 0 | 2,654,208 |
| 105 | I8 | 0.7410% | 0.6848% | 0.9180% | 0.7252% | 0 | 4,733,952 |

Pooled error/reference-energy RMS on consumed raw inputs: D 38.8966%,
DR 47.2550%, PR-B 44.3624%, FF 46.7875%, I4 8.6276%, I8 0.7127%.
This weights rows by original output energy and is not an equal-expert average.
Raw versus effective-input results differ by <=0.00012 percentage points
across arms on these rows; detailed effective and cosine/row metrics are
retained in the adjudication and raw points. No effective-input selection
or fitting occurred.

Original zero-input output is exactly zero. D/DR/I4/I8 preserve it; offsets
in PR-B/FF produce 2.994%-6.541% of calibration output norm and fail the
fixed 0.01% zero gate. Zero-source RMS is undefined, explicitly stored as null;
the separate output/calibration norm handles this case without division by zero.

## What changed, and what the evidence does not explain

Layerwise behavior-aware conversion fits calibration strongly, but development
RMS is worse than direct ternary for both experts. Nearby perturbations transfer
much better than raw development or independent-coordinate Gaussian probes,
yet still fail 1%. This supports a fitted-set/generalization distinction for
the current procedure. Calibration has only 87/122 rows against 768 input and
3072 hidden dimensions; insufficient coverage/regularization is a plausible
explanation, not a causal result established by this experiment.

The fixed hard-forward STE schedule does not improve the already accurate
PR-B calibration function: final RMS becomes ~13.2% for both experts. All six
parameter tensors verify 256 applied Adam updates per expert. Wi/Wo hard codes
change by 26,965/36,756 (expert 8) and 59,784/100,538 (105). A finite,
correctly applied identity-surrogate gradient is not a guarantee of decreasing
the discontinuous hard objective. Only the prospectively fixed final checkpoint
was evaluated; no best-step selection or learning-rate tuning on these outcomes.

A next method study should address full-function acceptance and transfer,
rather than infer usefulness from fitted error or rewrite thresholds. A
prospective bounded discrete-update experiment with explicit behavior checks
and controlled calibration coverage can distinguish these hypotheses. Its
training/evaluation probes must receive new fixed identities and seeds; current
role-1/probe outcomes are now consumed evidence. Whole-model language capacity
and native execution still require their separate validations.

## Counted deployment and cost

Both matrices together: 4,718,592 weights, 9,437,184 FP16 reference bytes;
original admitted F32 pair uses 18,874,368 bytes. Each native pair adds exactly
64 header bytes. Ternary D/DR payload ratio is 15.625%; PR-B/FF 15.7878%,
including 15,360 bytes of offsets. I4 28.125%; I8 50.1628% includes row scales.
No optimizer or continuous latent matrices are counted as deployed state,
because they are absent from the exported pair files.

Total fitting: 21.2515s; FF 2.7178s/2.4091s. Bootstrap install 20.1234s,
experiment 48.0107s, audit 6.6360s. Experiment-internal elapsed 34.7828s,
peak cuda:0 allocation 529,676,800 bytes, final process RSS 1,785,880,576 bytes.
These are phase/process observations, not end-to-end native latency or summed
device RAM; launch/queue/retrieval costs are outside those phase measurements.
Post-run acct2 quota reports 0.0433467 GPU hours used including the failed attempt,
29.9566533 unreserved; acct1 29.5884739, acct3 30, all zero reservations.
Snapshots do not prove account/resource idleness. No local native benchmark ran.

## Verification, first failures and resumption

Scientific source `3215ca3`, bundle `01e53cd`, private acct2 kernel
137234639/version 1, terminal COMPLETE. The unchanged original input dataset
is private ID 12391078/version 1, nine files / 41,059,932 bytes, exact mounted
hashes verified. The separate cuda:1 audit imports no fitting code and independently
decodes every native/code/scale/bias representation, verifies segment bytes,
regenerates probe and optimizer-schedule bytes, replays all conditions in FP64,
checks row/aggregate metrics and reconstructs all gates. Maximum main-F32 versus
audit-FP64 RMS is 3.6339962e-7, below fixed 1e-5. Source gradients are absent.

R1 changes only the duplicate calibration-input save. The first failed run is
retained intact (103 files / 92,691,349 bytes), including its original trace;
all 81 first-run numerical NPY/native artifacts reproduce byte-for-byte in R1.
An additional post-retrieval Git-path verification error, caused by Windows
separators, is retained in `PQT_005_R1_VERIFICATION_FAILURE_001.json`; only the
Git lookup path was normalized, without altering raw evidence or tolerances.
All 379 files across both attempts were rehashed and all 21 first-run small
raw files checked against their committed Git blobs. R1: 276 files /
108,634,286 bytes, thirteen actual server sources exact against frozen blobs;
small raw evidence under `pqt_005_r1_evidence/`, full arrays/native files under
`results/progressive_ternary/PQT-005-R1/remote_001/PQT-005-R1/`.

Monitoring and terminal retrieval were delegated to Luna, with the coordinator
waiting as requested. Monitoring is finished, all jobs terminal. No merge,
push to Git, source-checkout edits, local GPU job or native timing. Do not
repeat dataset creation, kernel dispatch or retrieval. Resume from
[TERNARY_INDEX](TERNARY_INDEX.md). Native implementation should consume these
same files and establish correctness before a separately coordinated timing
slot; quality failure forbids presenting a faster packed artifact as preserved
useful capacity. The complete-model research goal remains active.
