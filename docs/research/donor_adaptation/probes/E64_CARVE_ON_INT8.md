# E64 — the carve on an int8 FFN: THE PLANTED CONTROL DID NOT FIRE, AND NO CELL MAY BE READ

**Brief:** `briefs/BRIEF_E64_WHAT_DOES_THE_CARVE_COST_ON_AN_INT8_FFN.md` (+ addendum A)
**Runner:** `benchmarks/donor_adaptation/engine/e64_carve_on_int8.py`, 20 self-tests
**Result:** `benchmarks/donor_adaptation/engine/results/e64_carve_on_int8.json`
**Engine:** `donor_engine_e63.exe` · donor Qwen2.5-1.5B · frozen slice 24×512, 51,870 scored
bytes, 12,264 predicted positions · chance BPB 4.069819
**Wall:** 4,956 s · **QUALITY ONLY, no rate measured. `G-E63d` is untouched and still `VOID`.**

**Verdict: `E64-VOID-BY-ITS-OWN-CONTROL`.** `G-E64a` did not fire. §4 of the brief and
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
of the router scores. A build-level difference of ~1e-07 in a score — a different reduction
order, a different instruction selection between the E37-era binary and `e63` — is invisible
while it stays inside a continuous path, but when it lands on two groups that are nearly tied at
the `k`-th position it **flips which group is kept**, and a whole group entering or leaving the
computation is worth ~1e-03 BPB. The number of flippable boundaries is zero at `k = E` (nothing
is excluded), small at `k = 1` (only the argmax, rarely contested), and largest in between —
which is the column above.

**The law, and it generalises past E64:**

> **A selection path amplifies build-level floating-point noise from ~1e-07 to ~1e-03, because
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

## 4. What E64 does NOT say

* **No band.** `CARVE-IS-DEARER-ON-INT8` is what the runner printed; it is **not published as a
  result** and does not enter the ledger, the INDEX or any decision. The runner printing it while
  refusing it is `feedback_runner_verdict_outlives_spec` working as designed: *the script prints
  the numbers and not the conclusion.*
* **No carve cost on int8**, in either direction. The question the brief asked is **unanswered**.
* **Nothing about 10 B**, nothing about rate, and nothing that reopens post-hoc conversion. E38
  stands.
* **`G-E63d` is still `VOID` and OWED** and E64 did not touch it.

## 5. What the re-run needs, pre-registered here

1. **A replication control scoped to the axis it watches.** Either compare against E37 **on
   E37's own binary**, or register the tolerance from *measured* cross-build dispersion on a
   carved ladder — measure it first, then register it. It may not be chosen after seeing today's
   deltas.
2. **The verdict's control must be same-binary**, matching the comparison it gates: both arms
   exported and read on one build, with a discrimination control (`G-E64b`-style) per arm.
3. **Record the binary in every result JSON.** `results/e37_sparsity_cost.json` does not say
   which engine produced it — that is why §2's diagnosis had to be inferred from the shape of
   the deltas instead of read off the artefacts. **Owed fix:** the engine filename and its
   sha256 go in every runner's output.
4. **E36's run-2 rule applies in spirit**: today's run is void, so a corrected run is a *new*
   registered measurement, not a promotion of this one, and today's numbers may not be quoted as
   corroboration of whatever it returns.

## 6. Cost

4,956 s of CPU, one exported artefact (`D:/_ktmp/e64/e64_carved_i8.bin`, 2,292,609,912 B, Gate
E26-L matched at zero tolerance), and two laws. The export is **kept** — it is correct, its
sidecar confirms `ffn_bytes_per_weight=1.0` and `kinds=['MK_I8','MK_I8_T']`, and the re-run does
not need to rebuild it.
