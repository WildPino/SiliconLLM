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
