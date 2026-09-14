# E64 — the carve on an int8 FFN: run 1 VOID; run 2 says CARVE-IS-DEARER-ON-INT8

**Brief:** `briefs/BRIEF_E64_WHAT_DOES_THE_CARVE_COST_ON_AN_INT8_FFN.md` (+ addenda A/B)
**Runner:** `benchmarks/donor_adaptation/engine/e64_carve_on_int8.py`, 20 self-tests
**Run-1 result:** `benchmarks/donor_adaptation/engine/results/e64_carve_on_int8.json`
**Run-2 result:** `benchmarks/donor_adaptation/engine/results/e64_carve_on_int8_run2.json`
**Run-2 audit:** `benchmarks/donor_adaptation/engine/results/e64_carve_on_int8_run2_audit.json`
**Run-2 log / contention note:** `benchmarks/donor_adaptation/engine/e64_run2.log` ·
`benchmarks/donor_adaptation/engine/results/e64_run2_CONTENTION.txt`
**Engine:** `donor_engine_e63.exe` · donor Qwen2.5-1.5B · frozen slice 24×512, 51,870 scored
bytes, 12,264 predicted positions · chance BPB 4.069819
**Run-1 wall:** 4,956 s · **Run-2 wall:** 4,875 s, timing inadmissible due to contention ·
**QUALITY ONLY, no rate measured. `G-E63d` is untouched and still `VOID`.**

**Run-1 verdict: `E64-VOID-BY-ITS-OWN-CONTROL`.** `G-E64a` did not fire. §4 of the brief and
prediction 4 both registered that in this case **no cell in E64 may be read**, and no cell is
read here. The band the runner computed is not published as a result.

---

## 1. What the control did

`G-E64a` is a replication: re-run E37's ternary carve ladder on today's binary and reproduce
E37's published BPB to **|d| ≤ 1e-06**. It is the programme's standing `feedback_planted_controls`
requirement — *an instrument must FIRE on a known positive before its nulls count.*

| `k` | E64 re-run (e63 binary) | E37 published | \|d\| | agrees at 1e-06 |
|---|---|---|---|---|
| **256** (uncarved) | 3.475706063144 | 3.475706652030 | **5.889e-07** | **yes** |
| 64 | 3.927120144932 | 3.927383733540 | 2.636e-04 | no |
| 32 | 4.073547190543 | 4.074431016419 | **8.838e-04** | no |
| 16 | 3.987074577961 | 3.986800953001 | 2.736e-04 | no |
| 8 | 3.996743524570 | 3.996693360510 | 5.016e-05 | no |
| 4 | 4.023469372936 | 4.023439039328 | 3.033e-05 | no |
| 3 | 4.029419786438 | 4.029398350227 | 2.144e-05 | no |
| 2 | 4.014875101074 | 4.014866077843 | 9.023e-06 | no |
| 1 | 3.989826112802 | 3.989838501155 | 1.239e-05 | no |

`max|d| = 8.838e-04` against `tol = 1e-06`. **DOES NOT FIRE.**

## 2. The failure has a shape, and the shape names the cause

This is not a uniform offset. Two cells of the same run bound it from the other side:

* **`k = 256`, the uncarved cell, agrees to 5.889e-07** — the same weights, the same engine, the
  same slice, with the selection switched off.
* **`G-E64b` FIRES**: carved-at-`k=E` against the separately-exported uncarved file reads
  |d| = **2.609e-07** against a 1e-04 tolerance.

So the engine's arithmetic reproduces E37 to ~1e-07 wherever nothing is being selected, and
disagrees by up to 8.8e-04 wherever something is. The discrepancy is **confined to the carve
path** and is largest at intermediate `k` (peak at 32, decaying towards both 256 and 1).

**The mechanism that fits that shape is top-`k` boundary flips.** Selection is a *step function*
of the router scores. A difference of ~1e-07 in a score — a different reduction order
between the kernel E37 ran and the one `e63` defaults to (**section 7 names it**) — is invisible
while it stays inside a continuous path, but when it lands on two groups that are nearly tied at
the `k`-th position it **flips which group is kept**, and a whole group entering or leaving the
computation is worth ~1e-03 BPB. The number of flippable boundaries is zero at `k = E` (nothing
is excluded), small at `k = 1` (only the argmax, rarely contested), and largest in between —
which is the column above.

**The law, and it generalises past E64** -- **the AMPLIFICATION below is confirmed and the
ATTRIBUTION is CORRECTED in section 7: the perturbation is not build noise, it is a named
kernel change, and it was measurable rather than inferable:**

> **A selection path amplifies a perturbation of the scores from ~1e-07 to ~1e-03, because
> top-`k` is discontinuous in the scores. A replication tolerance derived from a DENSE path is
> MALFORMED on a CARVED one.**

`1e-06` was not too strict in general — E37's own `G-E37A` used 1e-04 and the uncarved cell here
clears 1e-06 comfortably. It was strict **on the wrong axis**: a dense-path tolerance applied to
a discontinuous quantity, with no measured cross-build dispersion behind it. That is
`feedback_gate_vs_measured_dispersion` — *a gate's tolerance must not be tighter than the
dispersion measured on the axis it watches* — and no such dispersion had ever been measured for
a cross-build carve replication, because nobody had run one.

## 3. A second lesson: the control was wired to void more than it validates

`G-E64a` is a **cross-binary** control: E37's binary against `e63`. `G-E64d`, the verdict, is a
**within-run** comparison: two artefacts measured minutes apart **on the same `e63` binary**.
Cross-build drift is common-mode to both arms of that comparison and cannot mechanically produce
a difference between them.

**The registered linkage is therefore stronger than the failure logically requires** — and it is
still honoured, because it was registered. But it is a defect in how the control was written, of
the same family as `feedback_instrument_must_not_measure_itself` and
`feedback_gate_blind_to_what_it_gates`:

> **A planted control must be scoped to what it actually validates.** A control that proves
> *"this binary agrees with that binary"* should gate claims that cross binaries. Wiring it to
> void a same-binary comparison discards evidence it does not bear on.

Writing a control that voids too much is the safe direction to err, and it cost 4,956 s rather
than a retraction. It is recorded so the re-run's control is scoped correctly.

## 4. What run 1 does NOT say

* **No band.** `CARVE-IS-DEARER-ON-INT8` is what the runner printed; it is **not published as a
  result** and does not enter the ledger, the INDEX or any decision. The runner printing it while
  refusing it is `feedback_runner_verdict_outlives_spec` working as designed: *the script prints
  the numbers and not the conclusion.*
* **No carve cost on int8**, in either direction. The question the brief asked is **unanswered**.
* **Nothing about 10 B**, nothing about rate, and nothing that reopens post-hoc conversion. E38
  stands.
* **`G-E63d` is still `VOID` and OWED** and E64 did not touch it.

## 5. What the re-run needs, pre-registered here

1. **SUPERSEDED BY SECTION 7 — the repair is not a tolerance, it is an assertion.** This item
   registered two options: run on E37's binary, or widen the tolerance from measured cross-build
   dispersion. Section 7 measured the cause and **neither is needed**: the re-run asserts
   `attn=serial`, the arm E37 ran, and the control then reproduces E37's ladder **exactly**.
   There is no dispersion to widen for, because there was no noise — there was a flag.
2. **The verdict's control must be same-binary**, matching the comparison it gates: both arms
   exported and read on one build, with a discrimination control (`G-E64b`-style) per arm.
3. **CORRECTED BY SECTION 7 — the binary was never the unit, and it was never missing.**
   `results/e37_sparsity_cost.json` does not record the engine, which is a real defect; but
   `e37_sparsity_cost.py:44` hard-codes `donor_engine_e26.exe`, so the provenance **was** in the
   repo. And the binary is the wrong unit anyway: `e63` reproduces E37 exactly once
   `--attn serial` is passed. **Owed fix, restated:** every runner's output records the engine
   filename, its sha256, **and the engine's own `CONFIG` line** — which has named the kernel arm
   since E50.
4. **E36's run-2 rule applies in spirit**: today's run is void, so a corrected run is a *new*
   registered measurement, not a promotion of this one, and today's numbers may not be quoted as
   corroboration of whatever it returns.

## 6. Cost

4,956 s of CPU, one exported artefact (`D:/_ktmp/e64/e64_carved_i8.bin`, 2,292,609,912 B, Gate
E26-L matched at zero tolerance), and two laws. The export is **kept** — it is correct, its
sidecar confirms `ffn_bytes_per_weight=1.0` and `kinds=['MK_I8','MK_I8_T']`, and the re-run does
not need to rebuild it.

---

## 7. CORRECTION, same day — the perturbation has a NAME, and I could have read it instead of inferring it

**What section 2 published:** *"a selection path amplifies **build-level floating-point noise**
from ~1e-07 to ~1e-03"*, with the perturbation attributed to *"a different reduction order, a
different instruction selection between the E37-era binary and `e63`"* — i.e. to incidental
variation between builds. **That attribution is wrong, and it was inferred from the shape of a
residual when it could have been read off three files and then measured.**

### 7.1 What was already written down, before E64 was published

| where | what it says |
|---|---|
| `engine/e37_sparsity_cost.py:44` | `ENGINE = os.path.join(HERE, "donor_engine_e26.exe")` — E37's runner **hard-codes its binary** |
| `engine/donor_engine.c:207-208` | *"DEFAULT avx4, adopted by E50. It was `ATTN_SERIAL` until 2026-09-13 and **NO runner from E26 onward passed `--attn`**"* |
| `engine/e64_carve_on_int8.py:121` | my own control's docstring: *"E37's published ladder, **which was taken on `donor_engine_e26.exe`**"* |

E50 is commit `61d1c29`, **2026-09-13**, *"the fast kernel is the default"*; its diff is
`-static int g_attn=ATTN_SERIAL;` / `+static int g_attn=ATTN_AVX4;`. E37's result was committed
`14455ec`, **2026-09-12** — the day before. **`G-E64a` was comparing two different attention
kernels and calling the difference a replication failure.**

### 7.2 Two measurements, both registered before they were run

**Prediction 1** (registered before the run): *`donor_engine_e26.exe`, the pre-E50 build,
reproduces E37's `k=256` and `k=32` to ~1e-9.*

**Prediction 2** (registered before the run, after prediction 1 returned): *`donor_engine_e63.exe
--attn serial` — today's binary forced onto E37's arm — reproduces the same two values; if it
does, the cause is **one flag** and the four other commits between `e26` and `e63` are exonerated
for these cells.*

Same frozen slice (`N_PREDICTED 12264`), same artefact `D:\_ktmp\e37\e37_carved_nf.bin`,
`--threads 6 --seqlen 512`:

| `k` | E37 published | `e26` (pre-E50) | `e63 --attn serial` | `e63` **default** (`avx4`, what E64 ran) |
|---|---|---|---|---|
| **256** (selection OFF) | 3.4757066520304316 | **same, \|d\| = 0** | **same, \|d\| = 0** | 3.4757060631435763 — \|d\| **5.889e-07** |
| **32** (selection ON) | 4.074431016419263 | **same, \|d\| = 0** | **same, \|d\| = 0** | 4.073547190542747 — \|d\| **8.838e-04** |

Both predictions **TAKEN**. The agreement is exact to the resolution of the engine's own log
(`NATS_PER_TOKEN` prints 10 decimals = **3.4e-11 BPB**); it is not claimed tighter than that.
`e63 --attn serial` printed `CONFIG  attn=serial  attnr=none  fexp=libm  mvacc=4  threads=6
quant=ternary` on both cells, so the arm is asserted from the engine's own output, not assumed.

### 7.3 What survives, and what changes

**The amplification SURVIVES and is now better founded than when it was published.** Section 2
inferred it from the shape of nine deltas across two binaries. It is now a **paired,
single-variable measurement**: one binary, one process, one artefact, one slice, **`--attn` the
only difference** — 5.889e-07 with selection off, 8.838e-04 with selection on, a factor of
**~1500x**. That is a cleaner demonstration than the one that produced the claim.

**The attribution CHANGES**, and with it the prescription:

> **§63.1 (corrected) — a selection path amplifies a perturbation of the scores by ~1500x: the
> same difference reads 5.889e-07 with selection off and 8.838e-04 with selection on. The source
> of the perturbation is irrelevant; its size in the CONTINUOUS path is what gets multiplied.
> Therefore (a) a replication tolerance taken from a dense path is MALFORMED on a carved one, and
> (b) a carve replication must assert the KERNEL ARM, not the binary — the engine has printed it
> in its `CONFIG` line since E50, and asserting it costs nothing and needs no tolerance at all.**

**Scope of the 1500x:** one perturbation source, one donor, one slice, two cells. It is an
order-of-magnitude statement about what selection does to a small difference, **not a constant**,
and nothing should be divided by it.

### 7.4 `G-E64a` is MALFORMED, not failed — and what that does and does not permit

`G-E64a` asked *"does today's binary reproduce E37?"* when the quantity that decides replication
is the **kernel arm**, which the gate never asserted and the engine was already printing. Under
the **E4 precedent** — *a gate that cannot answer the question it was written for is MALFORMED,
not failed, and may be re-specified* — `G-E64a` is re-specifiable. It is **not** the
`E40 addendum A` case of a gate that fired correctly and is being re-run to a pass: the repair
makes the control **stricter** (it asserts something the original omitted), and it was forced by
a measurement rather than chosen after browsing the deltas for a way through.

**What this does NOT do:** it does not un-void E64's int8 cells. Those were computed under
`attn=avx4`, and **E36's run-2 rule** stands — a corrected run is a *new* registered
measurement, and today's int8 numbers may not be quoted as corroborating whatever it returns.
The re-run is cheap: the export `D:/_ktmp/e64/e64_carved_i8.bin` is kept and correct, so only
the BPB sweeps repeat, with `--attn serial` asserted from `CONFIG`.

### 7.5 The lesson, and it is not the one section 2 recorded

Section 2 recorded *"a replication tolerance must match its axis"*, which is true and was already
`feedback_gate_vs_measured_dispersion`. The lesson that actually cost something is different:

> **I inferred a mechanism from the shape of a residual while the answer was sitting in three
> files I had already written or read — including the docstring of the control that failed.**
> `feedback_verify_public_claims` says a mechanism claim must be derived or measured, never
> asserted because it fits. *"It fits the shape of the deltas"* is exactly the kind of fit that
> feels like a derivation and is not one. **Before explaining a discrepancy, read the provenance
> of both sides** — and when the instrument prints its own configuration, the provenance is one
> line of output away.

The irony is worth keeping: E50's stated purpose was *"the thing that hid for twenty-two
experiments can no longer hide, because it appears in every log"*. It added the `CONFIG` line
for precisely this. **E64 walked into a cross-kernel comparison anyway, because E37's artefact
predates that line and I compared artefacts instead of configurations.**

---

## 8. RUN 2 — the repaired control fires, and the carve is dearer on int8

**Run 2 is a new measurement; it does not rehabilitate run 1.** Addendum B was committed at
`ccb80e2` before the runner was touched and before the run began. It fixed the attention arm to
`serial`, required every engine `CONFIG` line to confirm that arm, tightened `G-E64a2` from
`1e-06` to `1e-09`, and wrote to a new result path.

The re-run completed on the frozen 24×512 slice: 51,870 scored bytes and 12,264 predicted
positions. The runner's 20 self-tests pass, and a post-run independent audit recomputed every
BPB from `NATS_PER_TOKEN`, every carve cost from its own `k=E` baseline, and every inter-format
difference with zero discrepancies.

### 8.1 Provenance audit

| object | evidence |
|---|---|
| runner | current file is byte-for-byte Git blob `cf99c351c963cba2fe8314b4408ef34f417703e1`, committed in `f72e87e` |
| engine | `donor_engine_e63.exe`, sha256 `56272fdbe615d61739094605cb026aa308fd74604ba5598fab501c09188ae687`, mtime 2026-09-14 03:28:51 UTC — before the 08:01:36 UTC run start |
| ternary carved | 1,714,582,392 bytes, sha256 `fae666d2a4d4de7e69903f0c713181ed5214b416f77bbdafa7480490231b7150`, matches its sidecar |
| ternary dense | 1,709,047,348 bytes, sha256 `5d50e3778917a46f329c89334274be47941897f05b969b92bb4409fa3b0337eb`, matches its sidecar |
| int8 carved | 2,292,609,912 bytes, sha256 `e3233042eb733ca04dc2b3b51009ec67dd039712d2858ed38a6a4538b1f9fc9d`, sidecar confirms `ffn_rule=R8`, 1 B/weight and `MK_I8`/`MK_I8_T` |
| result | sha256 `7df759a547bdc3040d1bd9c3f6dbc886276d950c0fa86e5cde823c171a591821` |
| readable log | sha256 `cf73ab4b4a68dc62fbce34aa1febcae05e07d211917f7c668edd34523a1efe3b` before this promotion |

One sequence defect is recorded rather than hidden: the run-2 source was saved at 08:01:33 UTC
and the log opened at 08:01:36 UTC, but the exact source blob was committed only at 08:24:57 UTC,
about 23 minutes after launch. Thus `f72e87e`'s subject saying the apparatus was pushed before the
run is inaccurate. This does **not** alter the pre-registration — addendum B was committed at
08:00:54 UTC, before the source edit and run — and the committed blob exactly matches the current
runner. It is nevertheless a provenance limitation: Git proves the apparatus after launch, while
the result and log prove that the registered assertions actually executed.

The raw result's `brief` string says `+ addendum A` and omits the governing addendum B. That is
a metadata defect, not permission to rewrite the raw result after the fact; the separate audit
JSON records the correct commit and timeline while preserving the measured file byte-for-byte.

The contention sidecar also contains a malformed attempted shell timestamp rather than a clean
open/close interval. Therefore **none of the per-cell seconds is admissible as timing**. This is
already outside E64's estimand: the run is quality-only, and every accepted quantity below is a
deterministic BPB or a difference of BPBs.

### 8.2 Gates

| gate | run-2 reading | status |
|---|---|---|
| `G-E64a2.1` | all 19 scored invocations report `CONFIG attn=serial` | **FIRES** |
| `G-E64a2.2` | all nine E37 ladder cells reproduce exactly; max \|d\| = **0.0** vs `1e-09` | **FIRES** |
| `G-E64b` | ternary carved `k=E` vs same-format dense \|d\| = **2.801852e-07** vs `1e-04` | **FIRES**, ternary scope only as registered |
| `G-E64c` | each ladder is differenced against its own `k=256` baseline | **FIRES** |
| `G-E64d` | all eight non-baseline differences exceed `sigma_seed=0.250` with positive sign | **`CARVE-IS-DEARER-ON-INT8`** |

### 8.3 The measurement

The two baselines are **3.475706652030** for ternary and **2.216492625964** for int8. These are
not dense-fp32 baselines: all non-FFN organs remain in the artifact's registered ternary format.

| `k` | ternary carve cost | int8 carve cost | int8 − ternary |
|---:|---:|---:|---:|
| 64 | +0.451677 | +1.585027 | **+1.133350** |
| 32 | +0.598724 | +1.945073 | **+1.346348** |
| 16 | +0.511094 | +1.940853 | **+1.429759** |
| 8 | +0.520987 | +1.910935 | **+1.389949** |
| 4 | +0.547732 | +1.895515 | **+1.347783** |
| 3 | +0.553692 | +1.912017 | **+1.358325** |
| 2 | +0.539159 | +1.890057 | **+1.350897** |
| 1 | +0.514132 | +1.827039 | **+1.312907** |

At the target `k=3`, the int8-carved arm reads **4.128509790 BPB**, which is **0.058691 above
chance** (`4.069818973`). The one-byte FFN removes the half-byte format damage before selection,
but it exposes substantially more damage when the same post-hoc mask is applied. This is a
post-hoc floor on one donor, not a trained-format verdict.

### 8.4 Predictions scored

Addendum B prediction 1 **passes** (control and `CONFIG` assertions). Prediction 2 **passes** at
both named rungs and, more strongly, all eight rungs. Prediction 3 **misses**: `k=32` costs
`+1.945073`, above the registered `+1.20` ceiling. Prediction 4 **misses**: the int8 baseline
beats ternary by `1.259214`, not more than `2.0` BPB; the prediction incorrectly treated the
FFN-only format change like a whole-model format change. Prediction 5 is **not measured**: this
runner has no free-running arm, so silence in the BPB log is not generation evidence.

The original §5 predictions also stay visible. Precision-blindness and the predicted
`1.30–1.35` BPB `i8-k3` cell are **falsified** (`4.128510`). The no-10B prohibition is honoured,
the repaired planted control fires, and the prediction that the format gap is smaller at `k=16`
than at `k=3` **misses** (`1.429759 > 1.358325`).

### 8.5 What changes, and what does not

* The missing composition is now measured at 1.5 B: the post-hoc carve costs about **1.83–1.95
  BPB on int8**, versus **0.45–0.60 on ternary**, across the registered ladder.
* The result argues **against** assuming that a one-byte FFN makes the post-hoc carve easy. It
  does not argue against training in that format: H1 already showed applied and trained are
  different objects.
* No 7 B or 10 B quality extrapolation is licensed; E62's non-monotonic scale result still
  requires E66's direct 7 B measurement.
* No rate was measured. `G-E63d` remains `VOID` and OWED.
