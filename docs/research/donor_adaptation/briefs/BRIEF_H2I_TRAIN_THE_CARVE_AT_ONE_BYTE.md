# BRIEF H2I — train the carve at one byte, not the byte conversion

**Status:** Phase A pre-registered 2026-09-15, before its runner exists and before any H2I
number is measured. Phase B is contingent on Phase A and receives a separate addendum before
GPU execution.

**Cost now:** CPU only. No rate. No T4 request in Phase A.

## 0. The question

Can the selection/routing damage measured by E64 be trained away while the FFN stays in the
one-byte `R8` representation that E60/E62/E66 measured as faithful?

This is not `H2T`. That proposal trained the ternary body, paying GPU to heal a format loss
which E66 shows can be avoided. H2I keeps the byte conversion and trains the part still broken:
the carve and its router.

## 1. Why this experiment, from measured facts only

| fact already banked | consequence here |
|---|---|
| E66: 7 B `R8`, no fold, is +0.000378 BPB from fp32 and reproduces 4/5 complete trajectories | one byte is a faithful starting representation; do not train ternary-format damage |
| E64: at 1.5 B the `R8` carve costs +1.827–1.945 BPB; at `k=16`, +1.940853 and the full model is above chance | the selection hole survives, and is larger on int8 than on ternary |
| H1: training eight ternary-carved layers improves 1.096636 → 0.962593; the trained router becomes helpful | applied and trained selection are different objects; the training mechanism is live |
| E40: `R128` buys 50 tok/s at about 6% FFN activation and >100 tok/s near 1.7% | `k=16/256 = 6.25%` is the good-target structural rung; `k≈4` is the later excellent rung |

No scale transfer is made. H2I uses the 1.5 B donor because it fits one T4 and because E64 and
H1 were measured there. It is a mechanism falsification before any larger claim.

## 2. Phase A object — the matched applied baseline

Start from the exact H1 start state:

1. Qwen2.5-1.5B revision `8faed761d45a263340a0528343f099c05c9a4323`;
2. H0 run-3 trained ternary low-rank `q/o` factors installed;
3. the same eight layers `[3,6,9,12,15,18,21,24]`;
4. the same D0c `E=256` partition and E37 fitted routers;
5. hard engine-equivalent `{0,1}` gate, `k=16`;
6. **only the three FFN quantizers change:** H1's activation-aware ternary `R3` becomes
   per-output-row int8 RTN `R8`, exactly `amax/127`, round, clamp `[-127,127]`.

The comparison is therefore within one shape and one start state:

| row | FFN representation in 8 layers | selection |
|---|---|---|
| `h0-run3` | fp32 | none |
| `int8-8L` | `R8` STE forward | `k=E`, no carve |
| `applied-int8-8L` | same `R8` values | E37 router, hard `k=16` |

The published H1 `ternary-8L=0.947851138055396` and
`applied-8L=1.096636132580995` are hash-pinned and **not remeasured**. They are descriptive
cross-format partners, not thresholds imported into the new gate.

## 3. Frozen inputs

| input | bytes | sha256 |
|---|---:|---|
| H0 run-3 factors | 352,967,686 | `dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8` |
| D0c `labels_E256.npz` | 859,537 | `c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c` |
| H1 activation statistics | 1,189,630 | `49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe` |
| E37 routers | 44,046,858 | `42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a` |
| published H1 applied result | 1,605 | `6ebc333f74f0f337567a99e527cdfa7750d092161cc5ca2d1836a510557f1b17` |
| H1 held-out slice file | 98,564 | `110a90ab358efcacdc090fdf0fce62bb74d6d06df02d1083cf4cf7e752529d35` |

The runtime slice must still report 24×512, 51,870 scored bytes and IDs sha256
`a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`.
Hashes are identities, not evidence that the scientific gate fires.

## 4. Phase A controls and gates

All controls fire before `applied-int8-8L` is read.

### `G-H2Ia` — provenance

The runner must be the exact committed HEAD blob. Every input in §3 must match byte count and
sha256. The published H1 JSON must contain the exact four registered BPBs, layers, `k`, `E`,
model revision and slice identity. Any mismatch refuses before model loading.

### `G-H2Ib` — the R8 implementation is the shipped rule

On deterministic toy tensors, H2I's STE forward value must be bit-identical to
`t2_rules.r8_int8_rtn`: one scale per output row, codes in `[-127,127]`. A planted
non-ternary tensor must use at least five distinct code values; otherwise a ternary rule has
been mislabeled int8. Under backward, every fp32 master must receive a finite non-zero gradient.

### `G-H2Ic` — carve wiring

On the toy and one real layer:

- hard `k=E` must be bit-identical to the uncarved int8 FFN for **any router**, zero tolerance;
- hard `k=16` must activate exactly `16 × 35 = 560` neurons in the 1.5 B shape and must differ
  from `k=E`;
- the installed labels and router shapes must match exactly.

### `G-H2Id` — inherited instrument controls

On the frozen CPU fp32 instrument:

- intact BPB reproduces `0.767594964119663` within `1e-5`;
- H0 run-3 reproduces `0.810022487699936` within `1e-5`.

Failure voids all H2I cells. These are necessary same-instrument controls, not new findings.

### `G-H2Ie` — one byte is inert enough on the matched eight layers

`abs(int8-8L − h0-run3) <= 0.01 BPB`.

This is deliberately much looser than E62/E66's observed one-byte errors and is a launch gate,
not a claim of exact equivalence. If it fails, H2I is not selection-only and Phase B may not be
described that way.

### `G-H2If` — the applied selection hole is live

`applied-int8-8L > int8-8L`, ordinal, and the difference must exceed `0.05 BPB` before GPU
training is worth launching. If it is at most 0.05, the matched object is already close enough
that training this eight-layer proxy has little information value.

`applied-int8-8L` becomes Phase B's frozen matched threshold. It is written once to a new result
file and then hash-pinned in the Phase B addendum; it is never remeasured after training.

## 5. Rank partner — measured now, never inferred from BPB

Using the five frozen E6 prompts, Phase A first records H0 run-3's own 32-token greedy
continuations. It then measures `applied-int8-8L` against those exact trajectories:

- free-running positional agreement out of 160;
- first divergence and per-prompt counts;
- teacher-forced top-1 agreement and mean target rank on the frozen H0 trajectories.

This is descriptive in Phase A: no rank threshold is invented after seeing it. Phase B will
freeze its rank gate against this exact baseline before GPU execution. The purpose is to prevent
a BPB-only improvement from being promoted as a functioning model, the failure E16/E20 exposed.

## 6. Predictions, registered before the runner

1. `G-H2Ia–d` all fire.
2. `G-H2Ie` fires and `int8-8L` is within **0.002 BPB** of H0 run-3. The formal bar remains 0.01.
3. `G-H2If` fires by more than 0.05 BPB.
4. Despite the better uncarved base, `applied-int8-8L` is **worse** than H1's ternary
   `applied-8L=1.096636`, matching E64's sign. Descriptive magnitude prediction:
   **1.15–1.50 BPB**.
5. Free-running agreement is below **150/160**. This predicts that one byte preserves weights
   but does not make the severe selection rank-neutral.

## 7. What Phase A decides

| reading | action |
|---|---|
| controls fail | repair apparatus; no scientific number |
| one-byte delta >0.01 | H2I is not selection-only; diagnose quantizer mismatch before GPU |
| carve delta <=0.05 | do not train this proxy; selection is already cheap on the matched scope |
| one-byte delta <=0.01 and carve delta >0.05 | write Phase B addendum, adapt H1's trainer to R8 and run CPU toy/real smoke |

Phase B's intended object is one continuous 11-hour T4 session, same eight layers and `k=16`,
starting from H0 run 3 rather than from H1's already-trained ternary checkpoint. The quantizer
change should reduce per-step cost because R8 has no ten-point threshold search, but no throughput
or step-count prediction is registered until the actual trainer smoke measures it.

## 8. What H2I cannot establish

- No rate: training computes the dense FFN before masking. Engine rate remains E40/E63's domain.
- No 10 B quality transfer and no width/depth scale law.
- No `R128` rank-axis healing: H0 factors are a fixed inherited base here.
- No excellent-target claim: `k=16` prices the ~50 tok/s rung, not `k≈4` for ~100 tok/s.
- No activation-int8/LUT claim. That is a separate forward quantizer and must remain a separate
  partner so weight healing cannot hide activation damage.
- No result from the T4 until CPU fp32 BPB **and** rank partners adjudicate it.

---

# ADDENDUM A — Phase A result, 2026-09-15

**Written after the write-once Phase A run.** It records the registered gates without changing
them. Runner commit: `c4a156f`; result:
`s1/results/h2i/h2i_applied_8L.json`, 12,239 bytes, sha256
`c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc`.

## A.1 Controls

All preconditions fire. The runner and all imported helpers were exact committed HEAD blobs;
all six frozen inputs matched their registered byte counts and hashes. R8 was bit-identical to
`t2_rules.r8_int8_rtn`, used 13 distinct planted codes, and passed finite non-zero STE gradients
through `gate/up/down`. On the real layer-3 shape, hard `k=E` was bit-identical (`max|d|=0`),
hard `k=16` activated exactly 560 neurons at every tested token, and the carve was live
(`max|d|=11.7700`). The frozen instrument reproduced intact `0.767594964119663` exactly and H0
run 3 `0.810022487699936` to floating-point round-off.

## A.2 The matched baseline

| row | BPB | delta |
|---|---:|---:|
| H0 run 3, fp32 FFN | 0.810022488 | — |
| R8 on the eight H1 FFNs, `k=E` | **0.810005595** | **−0.000016893** vs H0 |
| R8 + E37 router, hard `k=16` | **1.019076465** | **+0.209070870** carve cost |

`G-H2Ie` fires by a factor of about 592 against its `0.01` maximum; `G-H2If` fires by a factor
of about 4.18 against its `>0.05` launch threshold. **Phase B is eligible.** The one-byte
conversion is not the object to heal on this matched scope; the selection hole is.

The registered prediction that applied R8 would be worse than H1's applied ternary
`1.096636133` is **falsified in the favourable direction**: R8 is better by `0.077559667`.
E64's all-28-layer ordering does not transfer to this eight-layer restriction. This is not a
contradiction: the scope and total accumulated perturbation differ, and neither result may be
multiplied by layer count.

## A.3 Rank partner

Against H0 run 3's own trajectories, captured before installing R8:

| metric | result |
|---|---:|
| free-running positional agreement | **9/160** |
| per prompt | **[6, 0, 0, 0, 3]** |
| first divergence, zero-based | **[1, 0, 0, 0, 3]** |
| teacher-forced top-1 | **99/160** |
| teacher-forced per prompt | **[19, 26, 18, 19, 17]** |
| mean target rank | **3.675** |
| target rank <=5 | **143/160** |

Prediction 5 (`free <150`) is taken, but that loose prediction is not the result. The result is
the same autoregressive cliff seen earlier: H0's target remains close under fixed context while
free-running fails immediately on four prompts. Phase B must beat the exact score and rank
baseline; a BPB-only win is insufficient.

## A.4 Predictions scored

1. controls fire — **TAKEN**;
2. R8 within 0.002 BPB — **TAKEN**, absolute error `0.000016893`;
3. carve cost above 0.05 — **TAKEN**, `0.209070870`;
4. applied R8 worse than applied ternary, predicted 1.15–1.50 — **FALSIFIED**, favourable,
   actual `1.019076465`;
5. free-running below 150/160 — **TAKEN**, actual `9/160`.

No rate, 10 B, scale, rank-compression or activation-int8 claim is added by this result.

---

# ADDENDUM B — Phase B training protocol, before its trainer exists

**Pre-registered 2026-09-15 after Phase A and before `h2i_qat.py`, `h2i_eval.py`, any training
bundle, CPU real-donor update or T4 result exists.** Phase A's raw result is now an input, fixed
at 12,239 bytes and sha256
`c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc`.

## B.1 The object and the one deliberate initialization change

Phase B trains the fp32 masters behind R8 and the router jointly, while every forward keeps the
three selected FFNs in exact R8. It inherits H1's model revision, H0 run-3 frozen q/o, layers
`[3,6,9,12,15,18,21,24]`, D0c labels, `E=256`, `k=16`, renormalized soft training gate,
hard engine-equivalent evaluation gate, AdamW betas `(0.9,0.95)`, no weight decay, gradient
clip 1.0, batch 2, accumulation 8, expert LR `2e-4`, router LR `3e-4`, aux `0.01`, and SDPA.

The router starts from **E37's fitted router**, not zero. This is registered now because Phase A's
matched baseline is that exact router and because H1 already paid to show that a useful fitted
router exists. Relearning it from zero would spend the new run on an axis already measured. The
soft-gate step-0 value will differ from the hard applied baseline by construction; it is recorded,
not used as a threshold. CPU adjudication is hard-gated.

No H1 ternary FFN checkpoint is loaded. This is a fresh optimizer and a fresh R8 master state
starting from H0, so `--resume` is forbidden in the first Phase B session.

## B.2 Frozen training inputs

In addition to Phase A's six inputs:

| input | bytes | sha256 |
|---|---:|---|
| H1 train stream | 64,000,260 | `0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9` |
| H1 calibration stream | 131,332 | `b8d4184db0988f4d86ae3089e99ba167b8e05e6dcdbbc504f7679729fe7aa019` |

The held-out stream is already pinned in §3. A bundle packer must hash all payloads after writing
its run instructions, re-open the manifest and independently verify every entry. The training
runner must validate the manifest and refuse unlisted or changed scientific inputs before loading
the model.

## B.3 Pre-GPU controls

The T4 may not start until one committed runner passes:

1. exact R8 forward parity to `t2_rules.r8_int8_rtn`, at least five planted codes and finite
   non-zero STE gradients through all three FFN masters;
2. hard `k=E` identity for any router, soft `k=E` identity at router zero, exact real-shape
   `k=16` cardinality of 560 neurons and a live carve;
3. E37 router shapes and values installed exactly before step 0;
4. on a real-donor CPU one-update smoke, an optimizer update is actually applied and moves the
   layer-3 gate master, layer-24 down master and layer-3 router; no non-finite microbatch;
5. save/reload reproduces every master, router and label exactly and invalidates quantization
   caches.

The CPU smoke is an apparatus test, not a quality result; it may use a short stream and one or
two layers to control cost, but must include both depth extremes for the moved-master check.

## B.4 T4 session

One continuous session, maximum **11.0 hours**, `steps=4000`, checkpoint every 250 updates, new
seed `4242`. The trainer stops itself before the platform limit, saves both periodic and final
bundles, and records `steps_completed`, `stop_reason`, seconds/step, non-finite microbatches,
declined scaler updates, soft and hard progress BPB, occupancy and router-vs-STATIC. The exact
expected step count and bundle command are filled only after the CPU smoke measures this R8
runner; no H1 ternary seconds/step is transferred as a prediction.

GPU progress numbers are diagnostics. They never decide Phase B.

## B.5 CPU fp32 adjudication and frozen gates

The evaluator validates the Phase A result hash and the trained bundle/labels before model load,
reproduces intact and H0 within `1e-5`, then reports these hard-gated arms:

- trained R8 experts at `k=E`;
- trained experts + original E37 router at hard `k=16`;
- trained experts + trained router at hard `k=16` — the deployable arm;
- trained router against STATIC, both hard, on held-out tokens.

`G-H2I-score` fires iff deployable trained BPB is strictly below Phase A's
`1.0190764652622473`. The bands are:

| trained hard BPB | reading |
|---|---|
| `>= 1.0190764652622473` | `TRAINING-DOES-NOT-HELP` |
| `(0.8200055950172889, 1.0190764652622473)` | `TRAINING-HELPS` |
| `[0.8000055950172889, 0.8200055950172889]` | `CARVE-IS-NEAR-FREE` (within 0.01 of uncarved R8) |
| `< 0.8000055950172889` | `TRAINING-OVERSHOOTS-BASE` |

Boundary ownership is explicit. These labels describe the measured proxy; none means 10 B.

The rank reference is Phase A's stored H0 target IDs. `G-H2I-rank` fires only if **all three**
strict improvements hold: free-running `>9/160`, teacher-forced top-1 `>99/160`, and mean target
rank `<3.675`. Per-prompt counts and first divergence remain mandatory. The conjunction is
deliberately conservative: a loss win that leaves the autoregressive model at Phase A's floor is
not a functioning selection win.

`G-H2I` fires only if both score and rank gates fire. Router-vs-STATIC is separately required to
claim learned per-token routing; if it fails, an overall H2I pass is credited to trained experts,
not to the router.

## B.6 Predictions

1. Every pre-GPU and inherited instrument control fires.
2. `G-H2I-score` fires; descriptive expected hard BPB **0.86–0.96**. The range is not the gate.
3. `G-H2I-rank` fires on all three clauses. This is the highest-risk prediction.
4. The trained router beats STATIC under the same hard gate, continuing H1's mechanism result.
5. The trained router improves over the original E37 router with experts fixed. If not, the
   expert update may still win H2I but no new router claim is made.
6. The run reaches at least 250 applied updates and writes a loadable periodic checkpoint. The
   actual step-count prediction is deferred to the post-smoke execution addendum.

## B.7 Stop rules and exclusions

- Failed pre-GPU control: repair apparatus; spend zero T4 hours.
- Non-finite run or no applied update within 40 attempts: numerical failure, not H2I's null.
- Score fails: do not run rank as a rescue claim; retain any already-produced diagnostic only.
- Score passes but rank fails: `SCORE-ONLY`, do not export or scale.
- Both pass: next object is an engine-export parity check, still not a rate result.
- Do not tune gates, bands, prompts, target IDs, layers, `k`, learning rates or aux after T4.
- Do not mix activation-int8/LUT into this session. That partner starts only after weight-only R8
  is adjudicated.

---

# ADDENDUM C — pre-GPU readiness and correction to the deferred step-count estimate

**Recorded 2026-09-15 after the committed apparatus and real-donor CPU smoke, and before any
H2I T4 run.** This addendum does not change the Phase B object, command, gates, bands or
predictions.

## C.1 The pre-GPU gate fires

The committed `h2i_qat.py` self-test and a real-donor CPU smoke both pass. The write-once smoke
artifact is `s1/results/h2i/h2i_qat_cpu_smoke.json`, 5,259 bytes, sha256
`33ae14478a457e140b5de034fa2b9c31af448ac400308058d573fce675c9456e`. It validates the
bundle manifest and all inputs before model load, then establishes:

- exact R8 parity to `t2_rules` with 13 distinct planted codes;
- finite non-zero gate/up/down STE gradients;
- exact hard `k=E` identity for arbitrary routing and exact soft `k=E` identity at zero router;
- on real layer 3, exactly 560 active neurons at hard `k=16`, a live carve, and exact E37 router
  initialization;
- one applied optimizer update, no non-finite microbatch, and movement of `L03.gate`
  (`2.0000339e-4`), `L24.down` (`2.0000339e-4`) and `L03.router` (`3.0004978e-4`);
- exact save/reload of masters, routers and labels.

This is apparatus evidence only. The two-layer, one-update CPU smoke took 39.386 s and is not a
quality, throughput or expected-T4-step measurement.

## C.2 Frozen execution bundle

`h2i_pack.py` built `s1/_h2i_bundle` once from committed apparatus and hash-pinned inputs. Its
manifest is 3,387 bytes, sha256
`a37d65fdc3e7833bb2b539abd43a07d958a98bc9dd65d091240716a5606abf20`; it lists 15 payloads
totalling 463,452,303 bytes. A second independent pass re-opened and rehashed every payload with
zero mismatches. Status is **`READY_NOT_RUN`**. The only permitted first-session command is the
one in `_h2i_bundle/RUN.md`; it remains seed 4242, 4,000 requested updates, checkpoint every 250
and a hard 11.0-hour limit, with no resume.

## C.3 Correction to B.4

B.4 said the exact expected step count would be filled after the CPU smoke. That promise was too
strong: a two-layer CPU apparatus smoke cannot honestly predict the throughput of an eight-layer
T4 training session. No step-rate or expected final count is therefore invented here. The
already-preregistered directional prediction remains the only progress threshold: at least 250
applied updates and one loadable periodic checkpoint. Actual T4 seconds/update and completed
steps will be reported, not back-filled as a prediction.

The apparatus has now exhausted the useful local checks. Phase B requires one continuous
11-hour T4 session; CPU reruns would duplicate controls without adjudicating H2I.

---

# ADDENDUM D — external dispatch contract, before upload or launch

**Recorded 2026-09-15 before creating the H2I Kaggle dataset or kernel.** This is an operational
contract only: all scientific choices remain exactly those in addenda B–C.

The shared dispatcher is `s1/h_training_kaggle.py`. It must be committed before launch and must
pass `check h2i` plus `preflight h2i acct2`. The preflight observed the server identity
`giggio253`, 30.00 GPU-h remaining, and no active kernel at the reserved ref. H2I is assigned:

- account alias `acct2`, server identity `giggio253`;
- private dataset `giggio253/h2i-one-byte-phase-b-bundle`;
- private kernel `giggio253/h2i-one-byte-phase-b`;
- Kaggle machine enum `NvidiaTeslaT4`, platform cap 12 h, trainer cap 11.0 h.

Before upload, the dispatcher rehashes the frozen manifest and all 15 payloads. It stages through
hardlinks or copies in a temporary directory, never mutating `_h2i_bundle`. It waits until the
dataset is server-side READY before pushing. The generated kernel locates the mounted manifest by
its exact sha256 rather than by a guessed mount path, rehashes every mounted payload before model
load, runs only addendum B's command, and refuses success unless both final outputs exist.

The dispatcher also refuses a credential/server identity mismatch, less than 11 GPU-h remaining,
or an already queued/running kernel at the same ref. Kaggle progress BPB remains diagnostic and no
checkpoint selection is introduced. Status monitoring must be sparse/event-driven; do not poll at
two-second cadence.

---

# ADDENDUM E — Kaggle run 1 is operationally void; writable-copy repair before run 2

**Recorded 2026-09-15 after kernel version 1 terminated and before version 2 is pushed.** Run 1
is **`VOID_OPERATIONAL`** and contributes no H2I evidence. The mounted-manifest gate passed all
15 payloads at 5.633 s, proving the uploaded scientific bundle was exact. At 10.387 s, before
model load or training, importing `t2_rules.py` raised:

```
OSError: [Errno 30] Read-only file system:
'/kaggle/input/datasets/giggio253/h2i-one-byte-phase-b-bundle/results'
```

The module creates its historical `results/t2_arms` directory at import time. `_h2i_bundle/RUN.md`
explicitly said to copy a read-only mount into the working directory, but dispatcher version 1
ran directly from the mount. This is a launcher defect, not a failed pre-GPU control or numerical
failure. Kaggle still reported 0.00 GPU-h used after termination.

The only repair is in the generated kernel shim: after validating the mounted bundle, copy it to
`/kaggle/working/h2i_bundle`, repeat the manifest and every payload size/hash check on that copy,
and run the unchanged addendum-B command there. The dataset, manifest, trainer, inputs, seed,
hyperparameters, gates and output names do not change. Because run 1 never loaded the model or
attempted an update, version 2 is the first scientific H2I session and may reuse seed 4242.

The downloaded run-1 log is 4,169 bytes, sha256
`60e8d2fe4cbefa8e585fcab86ffeda582032073dc9a34189ba2425fab5fc4704`; its normalized record is
`s1/results/h2i/h2i_kaggle_run1_VOID_OPERATIONAL.json`.

---

# ADDENDUM F — Kaggle run 2 is operationally void; flat-bundle path repair before run 3

**Recorded 2026-09-15 after kernel version 2 terminated and before version 3 is pushed.** Run 2
is **`VOID_OPERATIONAL`** and carries no H2I evidence. It advances the apparatus audit beyond run
1: the mounted-manifest gate passed at 9.680 s, the copied-working-bundle gate passed at 11.467 s,
and `h2i_qat.py` entered its preflight at 23.565 s. At 26.449 s manifest validation attempted to
stat `/kaggle/working/engine/e6_generate.py` and raised `FileNotFoundError`, before model load or
any optimizer attempt.

The transport bundle is flat, but `h2i_qat.apparatus_files()` preserves the source-tree path
`HERE/../engine/e6_generate.py`. The identical file is already present and hash-pinned at the
bundle root; only its expected sibling path is absent. Version 3 therefore creates
`/kaggle/working/engine/e6_generate.py` as a copy of that payload and asserts its sha256 against
the unchanged manifest before starting the trainer. No scientific file, input, command, seed,
hyperparameter or gate changes, and no dataset upload is needed.

The shim also removes the temporary 463 MB working copy and compatibility directory in a
`finally` block. This prevents a normal startup error from being published and downloaded as
kernel output; final trainer artifacts remain outside that directory. Run 2 used 0.01–0.02
rounded GPU-h in total with run 1, and neither loaded the donor. Version 3 remains the first
scientific session and retains seed 4242.

The run-2 log is 4,257 bytes, sha256
`d171e348e41291e62e28fb972aa4bf5b6b07c72299678a33f161ef74e6738efd`; normalized record:
`s1/results/h2i/h2i_kaggle_run2_VOID_OPERATIONAL.json`.

---

# ADDENDUM G — Phase B scientific session launched

**Recorded 2026-09-15 after kernel version 3 remained RUNNING beyond both prior failure
windows.** Version 3 was pushed from dispatcher commit `7e1c5ce` against the unchanged dataset
and manifest. At 13:45:08 Europe/Rome, the authoritative Kaggle state was
`KernelWorkerStatus.RUNNING`; account `acct2` reported 0.03 GPU-h used / 29.97 remaining.

This is the first scientific Phase B session: versions 1 and 2 are operationally void and both
ended before model load. No tuning or seed change occurred between them. The live run is private
kernel `giggio253/h2i-one-byte-phase-b`, version 3. Do not infer startup-control success from
RUNNING alone; the final log must contain both mount/copy gates, trainer controls and an applied
update before the training artifact can be adjudicated.

---

# ADDENDUM H — Kaggle run 3 is operationally void; device-local RNG repair before run 4

**Recorded 2026-09-15 after kernel version 3 terminated and before editing the runner or pushing
version 4.** Run 3 is **`VOID_OPERATIONAL`** and contributes no H2I treatment evidence. It passed
the mounted-manifest gate at 6.727 s, the independently rehashed writable-copy gate at 8.299 s,
all pre-GPU controls and pinned-input checks, and completed loading the exact donor. At 59.765 s,
before the training loop or any optimizer update, the real-shape wiring control raised:

```
RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
```

`_real_controls()` asked `torch.randn` to allocate directly on `mod.gate.device` while supplying
the default CPU `torch.Generator`. The earlier real-donor smoke ran on CPU by contract, so it could
not expose this cross-device API incompatibility. This is a deterministic probe-construction
defect, not a failed wiring, loss, optimization or quality gate. Aggregate account usage after all
three short operational attempts is 0.04 GPU-h.

The only scientific-runner repair permitted before version 4 is to generate the seeded probe on
CPU with an explicit CPU generator and then transfer/cast it to the module device, matching the
already-safe construction in `h1_qat.py`. The probe remains seed 90210, shape `(1, 6, hidden)`,
and the same destination dtype/device; it consumes no global RNG state and is used only by the
pre-update control. Add a deterministic helper self-test, but do not alter data, labels, routers,
layers, `k`, loss, optimizer, learning rates, schedule, training seed, step count, gates or bands.

Because `h2i_qat.py` is hash-pinned apparatus, rebuild the bundle and publish a new dataset
version; every unchanged scientific payload must retain its exact hash. The dispatcher must be
made code-page-safe as well: output download succeeded only after forcing UTF-8 because the
Kaggle progress glyph could not be written to a cp1252 console. That transport fix has no bearing
on the kernel. Version 4 may launch only after committed self-tests, a full local bundle audit,
exact remote inventory and the standard account/quota/no-active-kernel preflight. Since version 3
made zero optimizer updates, version 4 remains the first H2I training session and retains seed
4242 without resumption.

The run-3 log is 104,042 bytes, sha256
`160b50b586e65e482efdb888c0b4f81cecfcd1b2f71423cdd283cc555802ad72`; normalized record:
`s1/results/h2i/h2i_kaggle_run3_VOID_OPERATIONAL.json`.

---

# ADDENDUM I — version-4 apparatus is ready, before upload or launch

**Recorded 2026-09-15 after implementing and testing addendum H, and before uploading the new
dataset version or pushing kernel version 4.** Commit `0de313c` implements only the registered
probe construction and output-console repairs. `h2i_qat.py --self-test` fires the inherited gates
and the new helper's repeat-exact/global-RNG-isolation checks. A new real-donor CPU smoke against
the rebuilt manifest fires the real k=16/cardinality/live-carve/router-init controls, applies one
update with zero non-finite microbatches, moves `L03.gate`, `L24.down` and `L03.router`, and reloads
every saved master/router/label exactly. The smoke record is 5,486 bytes, sha256
`315d033f3b9b1e1833162c09cfe57b5126bddcf8d8ba98b08df4380c277ca8ca`.

The immutable replacement bundle is `s1/_h2i_bundle_v4`: 15 payloads, 463,453,554 bytes;
manifest 3,387 bytes, sha256
`330fa237ebe3ea85c48447f52628a3f77f1f4cf0e7275ef7e122e9eb8b93ba56`. A second verification
pass rehashed every payload. Comparing manifests finds all eight scientific payload records
bit-identical and exactly one changed payload, `h2i_qat.py` (30,716 bytes, sha256
`f8df56204e564ca0ced074059d0418fcb16682b5d200cc18a1785d46290d6c02`). The old bundle remains
untouched as v1–v3 provenance. Status is **`READY_NOT_RUN`**; v4 still requires the standard remote
identity, quota, inactive-kernel, uploaded-inventory and mounted-hash gates.

---

# ADDENDUM J — Phase B version 4 launched

**Recorded 2026-09-15 after the repaired bundle passed every remote dispatch gate.** Account
`acct2` authenticated server-side as `giggio253` and reported 0.04 GPU-h used / 29.96 remaining
before launch. Dataset `giggio253/h2i-one-byte-phase-b-bundle` was versioned, then withheld until
its remote filename/size inventory exactly matched all 15 manifest payloads plus the manifest.
The kernel was inactive before push.

Private kernel `giggio253/h2i-one-byte-phase-b`, version 4, was pushed from dispatcher/bundle
commit `dd8ca8e` and reported `KernelWorkerStatus.RUNNING` at 17:59:22 Europe/Rome. This is the
first H2I session capable of entering the training loop; versions 1–3 remain operationally void
and are never resumed. RUNNING is not evidence that the startup control or first update fired.
Monitoring remains sparse/event-driven, and only terminal output may advance the protocol.

---

# ADDENDUM K — CPU adjudication frozen before terminal output is visible

**Recorded 2026-09-15 while version 4 still reported `KernelWorkerStatus.RUNNING`, before any
terminal artifact or final training metric was available.** This addendum changes no evaluator,
gate, band, model, prompt, target, training choice or checkpoint rule. It closes an operational
reproducibility gap: `h2i_eval.py` checks its dependencies against the then-current Git `HEAD`, so
the exact already-committed blobs that are allowed to adjudicate v4 are now frozen externally.

The machine-readable record is
`s1/results/h2i/h2i_phase_b_v4_adjudication_freeze.json`. It pins byte sizes, SHA-256 and Git blob
IDs for `h2i_eval.py` and all six dependencies it validates: `common.py`, `h1_qat.py`,
`h2i_applied.py`, `t2_rules.py`, `e6_generate.py` and the E6 engine descriptor. All seven matched
commit `9f2f7dc` at freeze time. The evaluator itself remains byte-for-byte the preregistered code;
no post-launch repair is introduced.

Terminal handling is fixed before the outcome:

1. While the kernel is non-terminal, do nothing except sparse status monitoring.
2. On `COMPLETE`, download once into a fresh directory, preserve the raw log, require the final
   NPZ+JSON pair, rehash the seven frozen evaluator dependencies, then run the write-once CPU fp32
   evaluator exactly once.
3. On `ERROR`, download the log once and classify the last passed gate; do not run the scientific
   evaluator against a missing or partial final pair.
4. Evaluator exit `2` is the registered `FAIL-SCORE` outcome and intentionally runs no rank. Exit
   `3` is `SCORE-ONLY`, meaning score passed and at least one rank clause failed. Neither is an
   operational error to be repaired or rerun.
5. Only exit `0`, with the recorded combined gate true, licenses the conditional engine-export
   seam. No checkpoint may be selected from progress diagnostics.

---

# ADDENDUM L — the v4 adjudication freeze is executable, before terminal output

**Recorded 2026-09-15 while version 4 still reported `KernelWorkerStatus.RUNNING`, before any
terminal artifact or final training metric was visible.** Addendum K froze the seven allowed
evaluator blobs, but its prose/JSON protocol still depended on a person correctly retyping and
checking every condition. `s1/h2i_v4_adjudicate.py` now makes that freeze executable without
changing `h2i_eval.py`, any threshold, input, prompt, target or exit-code meaning.

The guard refuses an uncommitted guard, any of the seven evaluator dependencies whose byte size,
SHA-256 or Git blob differs from addendum K, and any frozen scientific input whose identity
differs. It accepts only the final `h2i_trained_s1.npz` plus same-basename JSON and the preserved
raw `h2i-one-byte-phase-b.log`; periodic checkpoints are not candidates. The sidecar must identify
the exact v4 manifest `330fa237…`, all eight training inputs and all six Kaggle apparatus hashes,
the registered model/revision/object/hyperparameters, at least 250 completed and applied updates,
zero non-finite microbatches, a final stop reason, movement of every watched tensor, and every
pre-training real/toy/RNG control firing.

Only after that preflight does the guard invoke the unchanged write-once CPU evaluator with every
path explicit. It accepts evaluator return `0`, `2` or `3` only when status, score, rank execution
and combined-gate fields agree exactly, then writes a separate write-once audit containing hashes
of the raw log, trained pair, dependencies, inputs and result. Return `2` remains the scientific
`FAIL-SCORE` verdict and return `3` remains `SCORE-ONLY`; neither is repaired or rerun. A planted
self-test must pass before this guard is committed, and the committed guard identity is recorded
separately while v4 remains non-terminal.

---

# ADDENDUM M — committed H2I v4 terminal guard is ready

**Recorded 2026-09-15 after addendum L's implementation commit and while the last authoritative
status read still reported v4 `RUNNING`; no terminal output had been downloaded or inspected.**
Commit `365f811` adds only the external guard and documentation. `h2i_eval.py` and all seven
dependencies frozen by addendum K remain byte-identical.

The committed guard is 21,172 bytes, SHA-256
`cac0c8188d3f3df0f458e74317ad2b5fabbbf12c2d78250ec015f03df0bfe124`, Git blob
`25c948b6f5accce3d3cd8ce73f404201b2558cdd`. Its planted metadata/exit-semantic self-test passes;
its worktree blob equals the committed blob; and a post-commit pass rehashes all seven frozen
evaluator dependencies and seven local evaluator inputs successfully. The machine-readable
readiness record is `s1/results/h2i/h2i_phase_b_v4_adjudication_readiness.json`.

The terminal commands are now fixed: download once to the fresh
`D:\_ktmp\h2i_kaggle_v4_output`, then invoke the guard once with the final NPZ and preserved raw
log. Direct invocation of `h2i_eval.py` is superseded operationally for v4, but the evaluator
itself remains the unchanged scientific authority inside the guard.
