# BRIEF E50 — TURN THE FAST KERNEL ON BY DEFAULT, AND MAKE THE ENGINE SAY WHICH ONE IT RAN

**Registered before any E50 cell exists.** E49 §6 said explicitly that changing `g_attn`'s
initialiser is *a separate, gated change, and E49's job is to establish first whether it deserves
to be made*. E49 established it. This is that change, and its gates are about the **claims already
published under the old default**, not about the speed — the speed is E49's and is not re-measured
here.

## 1. The two changes, and why they are one experiment

**Change 1 — `donor_engine.c:202`:** `static int g_attn=ATTN_SERIAL;` → `ATTN_AVX4`.

**Change 2 — the engine prints its own attention arm.** Today the `BENCH` line prints threads and
quantisation and the FFN witness, and **not** the attention arm. Nor does any other mode.

They are one experiment because change 2 is what makes change 1 auditable, and because **change 1
is the thing that can break a published claim** — so both are gated together or neither ships.

## 2. The part of this the source already knew, which is worse than E49 said

`donor_engine.c:76`, on why `g_mvacc` defaults to 4 rather than living behind a flag:

    // Defaulted rather than left as a flag because a flag a runner must remember to pass is exactly
    // the defect of s9, where --attn stayed on the slow kernel for every E3 and E7 number.

and again at line 85, on why the contention witness `g_wit` is on by default:

    // ON BY DEFAULT: a witness a runner has to remember to pass is the
    // same defect that left --attn on the slow kernel for every E3 and E7 number (s9).

**The file names this exact defect twice, cites `--attn` by name both times, and uses it as the
justification for defaulting two *other* things — while leaving `--attn` itself on the slow
default.** E49 reported that the flag was never passed. It is worse than that: the lesson had
already been drawn, written down in the source, and applied everywhere except to the flag it was
drawn from. E49's B.7 called this a class defect on the strength of the output line alone; the
source shows the class was already identified and the instance left standing.

## 3. What is actually at risk, and it is not the speed

E49 gated `avx4` against `serial` on **BPB** and got `|ΔBPB| ≤ 8.47e-07`. That is a scalar
average over 12,264 positions. **It is not the same claim as "the greedy trajectory is
unchanged"**, and this programme has a law about exactly that gap: Phase 60's — kernel-bit-exact
does not compose to system-correctness, so parity is end-to-end or it is nothing.

The end-to-end claim at risk is **E6's**, and it is the strongest claim this programme has:

> `results/e6/summary.json` — `A1` (fp32 Qwen2.5-0.5B): `agree 1.0`, `matched 160`, `counted 160`.
> **160 of 160 generated tokens identical to PyTorch's greedy trajectory** over five frozen
> prompts.

`e6_generate.py:88` builds `[ENG, --weights, wp, --threads, 6]` and **passes no `--attn`**. So
E6's 160/160 was measured on `serial`. A greedy argmax is a **discrete** function of the logits:
an `8.47e-07` average BPB change says nothing about whether some step's top-2 gap was smaller than
the perturbation. E6's own `ref.json` records `top2_gap` per step precisely because the brief
anticipated tie-breaks.

**If A1 falls below 160/160 under the new default, the default does not change.** That is the
whole point of gating it.

## 4. The gates

### `G-E50a` — E6's greedy claim must survive the new default

Re-run E6's **engine stage only**, unmodified in every other respect, with the new default and no
`--attn` flag, into `results/e50/` (E6's own results are **not** overwritten — they are the
record). Score the generated ids against E6's **frozen** `results/e6/ref.json`.

* **PASS**: `A1` matches **160/160**, exactly as `results/e6/summary.json` records.
* **FAIL**: anything less. The change is reverted and E50 reports that the fast kernel cannot be
  the default without re-opening E6.
* **PLANTED CONTROL, and the pass does not count without it.** The same scorer, unchanged, must
  **FIRE** on the two arms E6 already measured as disagreeing: `A2` (ternary+head 0.5B) at
  **3/160** and `A3` (1.5B) at **10/160**. A scorer that reports agreement for everything has not
  shown it can see disagreement. Both controls must stay far below 160 and both must remain
  **within ±2 tokens of E6's recorded counts**, since the ternary arms are affected by the kernel
  too and a large move there is itself news.

### `G-E50b` — the new default must reproduce E49's published numbers BIT FOR BIT

The engine is deterministic (E49 addendum A.1 reproduced the pair to the last digit). So:

* `--bpb` on S15 `--carve-k 256` with **no flag** must print `NATS_TOTAL` **124963.9517608703** —
  E49's `avx4` value, to every digit.
* the same with **`--attn serial`** must print **124963.9729339122** — E49's `serial` value, to
  every digit.

The first says the default really is `avx4`; the second says **the old numbers remain
reproducible**, which is the condition on which twenty-two experiments' worth of `serial` readings
stay in the ledger rather than becoming unrepeatable.

### `G-E50c` — the witness must be shown to WITNESS

A line that prints a constant is not a witness. The engine's own output must read `attn=avx4` with
no flag and `attn=serial` under `--attn serial`, **and** the same for a mode that is not `--bench`
(`--bpb`), since `--bpb` is where parity is decided and it prints no `BENCH` line at all.

### `G-E50d` — no existing runner may be broken by the new output

Collect every `BENCH`-parsing regex in `benchmarks/donor_adaptation/` (there are eight, in
`e3/e25/e26/e28/e30/e44/e48/e5`) and run all of them against the **new** `BENCH` line. Every one
must still match and must still return the same three fields (tokens, seconds, tok/s). This is
cheap and it is the difference between an observability fix and an outage.

### Not a gate — the drift line

The default arm's tok/s should now read E49's `avx4` cells. **The machine is not certified idle,
so this is reported with its occupancy and carries no verdict** (E43: a non-contended timing is
not a point either; E44 owns the interval). It is written down so that a wild disagreement is
visible, not so that an agreement can be claimed.

## 5. My prediction, registered

| quantity | prediction |
|---|---|
| `G-E50a` A1 | **160/160 survives.** The fp32 0.5B's top-2 gaps in `ref.json` are order 0.1–1.5 nats and the perturbation is order 1e-06 |
| `G-E50a` controls | A2 stays at **3/160** and A3 at **10/160**, unmoved — their disagreement is caused by ternarisation, which the kernel does not touch |
| `G-E50b` | both values reproduce **to every digit** |
| `G-E50c` | fires both ways |
| `G-E50d` | all eight regexes still match — they all stop at `tok/s`, before the parenthesis I am editing |
| drift line | the default reads E49's `avx4` cells within the dispersion E49 measured (2.3–5.9% at the large windows) |

E49 scored four right, one wrong, one void. This brief's predictions are weaker claims than E49's
and should be scored as such: the only one with real uncertainty is `G-E50a`.

## 6. What E50 does not claim

It does not re-measure speed — E49 owns those numbers and E44 owns their interval. It does not
touch quality. It does not revisit `--attnr`, whose default (`none`) is correct. And it does not
retro-fit the arm into any published log: **everything measured between E26 and E48 stays a
`serial` number**, and `G-E50b`'s second half is what keeps that statement checkable.

---

# ADDENDUM A — `G-E50a` AS WRITTEN CANNOT ATTRIBUTE A FAILURE, AND `G-E50b` CANNOT TELL MY CHANGE FROM THE BUILD

**Written before any E50 cell exists, after reading what the runners actually pin.** Both
additions are *arms*, not tolerances: no bar moves, no gate is relaxed, and every PASS condition
in §4 stands exactly as pushed.

## A.1 What §3 missed: E6 did not run the binary E49 ran

`e6_generate.py:33` pins `ENG = donor_engine.exe`, and E6 ran on **2026-09-07**. Every experiment
from E26 onward instead pins the frozen **`donor_engine_e26.exe`** (built 2026-09-11 18:34) —
`e26`, `e28`, `e30`, `e33`, `e34`, `e35`, `e36`, `e37`, `e39`, … and `e49`. Between those two
dates `donor_engine.c` took **E13** (`076381a`, blocked tile-major), **E25** (`e506c7b`, the
factored matvec) and **E26** (`1674e0c`, the carved FFN).

So re-running E6's engine stage on a fresh build tests **my one-line change PLUS six days of
accumulated engine drift**, and a fallen `A1` would not say which. An instrument that cannot
attribute its own failure is not finished.

`git log` confirms the other half of it: `1674e0c` is the **last commit to touch
`donor_engine.c`**, and the only working-tree diff is mine. So HEAD's source **is** the source of
`donor_engine_e26.exe`, and the drift above is entirely between E6's binary and E26's.

## A.2 The added arm — `G-E50a` gets a `serial` twin

`G-E50a` runs E6's engine stage **twice** on the new binary: once with no flag (the new default,
`avx4`) and once with **`--attn serial`** (the old default restored exactly). Both are scored
against E6's frozen `ref.json`. The four-way reading:

| `A1` avx4 | `A1` serial | reading |
|---|---|---|
| 160/160 | 160/160 | **PASS.** The default may change; nothing else moved either. |
| **< 160** | 160/160 | **FAIL, attributed to the kernel.** The default does not change, and E6 would have to be re-opened before it could. |
| 160/160 | **< 160** | **PASS on the gate, but an unrelated finding is now open**: something between E6 and E26 moved the `serial` trajectory. Report it, do not fold it in. |
| **< 160** | **< 160** | **VOID, not FAIL.** The change is not what broke it; E13/E25/E26 drift is, and that is a separate experiment. |

The last row is the one this addendum exists for. Under §4 as pushed it would have been recorded
as my change failing, which would have been **false**.

## A.3 The same hole in `G-E50b`, and the same shape of fix

`G-E50b` asks the new build to reproduce E49's two `NATS_TOTAL` values bit for bit — but E49
produced them with `donor_engine_e26.exe`, and I am running a **freshly compiled** binary. A
mismatch would therefore have two possible causes: my change, or the build not being reproducible
from the same source and flags.

The `--attn serial` half already discriminates them, and that is why it was registered: **if
`--attn serial` reproduces `124963.9729339122` exactly, the build is reproducible and my change is
the only delta.** So the addition here is only a *diagnostic*, and it is conditional:

* **If `G-E50b` passes, nothing extra runs.**
* **If it fails**, and only then, HEAD's **unmodified** source is compiled to a third binary and
  the `serial` arm is re-run on it. If that also misses, the defect is the toolchain and not this
  change, and `G-E50b` is recorded **VOID** rather than FAILED — `feedback_runner_verdict_outlives_spec`,
  same as `G-E48d`.

Registering the diagnostic *before* the run is the point: deciding after a failure which extra run
would exonerate me is how a gate gets talked out of firing.

## A.4 One more thing the run must not silently do

E6's stage writes `results/e6/engine.json`. E50 must **not** overwrite it — §4 already says so,
and the runner enforces it by pointing E6's `RES` at `results/e50/` before calling the stage, and
by refusing to start if `results/e6/engine.json` is missing or unreadable (it is the record being
compared against, and a run that silently regenerated it would be comparing the new binary to
itself).

## A.5 Prediction for the added arm

`A1` reads **160/160 on both kernels**. The accumulated E13/E25/E26 changes are all behind flags
(`--lutblk`, `--rank`, `--carve-k`) that E6's command line does not pass, so the fp32 0.5B path
should be untouched by them. **If that is wrong, the third row of A.2 is the interesting one and I
would rather find it here than have it found for me later.**

---

# ADDENDUM B — ALL FOUR GATES PASS, AND THE DEFAULT IS CHANGED

**Results: `results/e50_default_kernel.json`, log `e50_run1.log`, engine `donor_engine_e50.exe`
(frozen).** 17 decision-function self-tests fired in both directions before any cell was measured.

## B.1 `G-E50a` — E6's greedy claim survives, and the controls reproduced E6 EXACTLY

| arm | E6 (serial, 2026-09-07) | E50 `avx4` | E50 `serial` |
|---|---|---|---|
| **A1** — fp32 0.5B, the known-positive | **160/160** | **160/160** | **160/160** |
| A2 — ternary+head 0.5B, planted control | 3/160 | **3/160** | **3/160** |
| A3 — 1.5B, planted control | 10/160 | **10/160** | **10/160** |

**`G-E50a`: PASS.** A1 is 160/160 on both kernels, so the default may change.

Three things this says that the gate did not have to give:

1. **The controls did not merely stay inside the ±2 addendum A allows — they reproduced E6's
   counts to the token.** The scorer is demonstrably seeing the same disagreements E6 saw, on
   artifacts whose disagreement is caused by ternarisation and not by the kernel.
2. **The `serial` twin came back at 160/160 too**, which rules out A.2's third row: the six days
   of E13/E25/E26 drift between E6's binary and E26's left the fp32 greedy trajectory untouched.
   That was the row I most wanted not to find, and the arm existed so that I could not have
   confused it with my own change.
3. **`G-P` and `G-D` passed 30 times out of 30** — prefill logits byte-identical to `--logits`,
   and the same arm run twice byte-identical — across both kernels and all three arms.

So the discrete claim now stands where E49 only had the scalar one: `avx4` does not move a single
greedy token of the fp32 donor over five prompts and 160 steps.

## B.2 `G-E50b` — both values to every digit

| | measured | E49 |
|---|---|---|
| no flag | **124963.9517608703** | 124963.9517608703 (`avx4`) |
| `--attn serial` | **124963.9729339122** | 124963.9729339122 (`serial`) |

**`G-E50b`: PASS**, and the addendum A.3 diagnostic never ran because it did not have to. The
first line says the default really is `avx4`. The second says **every pre-E50 `serial` reading
stays reproducible on demand** — which is the condition on which E26–E48's numbers remain part
of the ledger rather than becoming orphans. It also settles A.3's other question: the build is
reproducible from the same source and flags.

## B.3 `G-E50c` — the witness witnesses

    CONFIG  attn=avx4  attnr=none  mvacc=4  threads=6  quant=fp32
    CONFIG  attn=serial  attnr=none  mvacc=4  threads=6  quant=fp32

Reported in `--bench` **and** in `--logits`, which prints no `BENCH` line at all, and it changes
with the flag. **PASS.** Under `--sweep*` it prints `attn=sweep`, because there the arm rotates
per token and a single name would be a lie.

## B.4 `G-E50d` — nobody is broken

    BENCH  20 tokens  1.181 s  16.93 tok/s  (threads=6, fp32, attn=avx4)  ffn~ 39.381 ms/tok

**PASS.** Eight call sites — `e3`, `e5`, `e25`, `e26`, `e28`, `e30` (which imports `e28`'s),
`e44`, `e48` — seven distinct regexes, all still matching and all returning the same
`(tokens, seconds, tok/s)`. The arm was placed **after** `tok/s` for exactly this reason; the
self-test `D2` runs the version that puts it before, and that version breaks every parser.

## B.5 The drift line — not a gate

| window | measured | E49 `avx4` | | occupancy | arm reported |
|---|---|---|---|---|---|
| n=40 | 124.67 tok/s | 121.65 | +2.5% | 25.9% | `avx4` |
| n=1280 | 74.47 tok/s | 74.11 | +0.5% | 54.9% | `avx4` |

Inside the dispersion E49 measured, and **with no flag passed** — which is the whole point. It
carries no verdict: the machine is not certified idle and E44 owns the interval.

## B.6 Prediction scorecard — 6 of 6, and they were easy

Every row of §5 came in as registered, including the two that could have gone otherwise (`A1`
surviving, and the controls not moving). §5 said in advance that these were weaker claims than
E49's and that only `G-E50a` carried real uncertainty; scoring them 6/6 is worth less than E49's
4-of-6, and is recorded that way rather than banked.

## B.7 What changed on disk, and what it means for everything else

* `donor_engine.c:202` is now `static int g_attn=ATTN_AVX4;`, with the E49/E50 provenance written
  above it and `--attn serial` named as the way to restore the old default exactly.
* Every mode prints a `CONFIG` line naming the arm. **The class defect is closed**: the thing
  that hid for twenty-two experiments can no longer hide, because it now appears in every log.
* `donor_engine_e50.exe` is frozen next to `donor_engine_e25/e26.exe`, following the convention
  that a published number names the binary that produced it.

**Three consequences that are now true and were not before:**

1. **`e44_interval.py` defaults to `donor_engine.exe`** (line 188), so when the idle-machine
   window arrives E44 measures the **restated** headline on `avx4` without any change to it — and
   its log will say `attn=avx4` rather than leaving it to be inferred.
2. **Every runner still pinned to `donor_engine_e26.exe` keeps measuring `serial`**, deliberately:
   those are the binaries their numbers were taken on. Re-pointing them is a separate decision per
   runner, not a sweep.
3. **`--sweep6`'s silent `ATTN_AVX4` is no longer silent, and no longer a discrepancy** — the
   sweep and the default now agree, which is what made E48 and E46 look like they contradicted
   each other.

## B.8 A defect of my own, recorded because it nearly cost run 1

The drift phase crashed on `Occupancy.start()`, a method that does not exist — `e44_interval.py`'s
`Occupancy` exposes `sample()`, which returns the busy fraction **since the previous read**. It
crashed **after every gate had returned**, and because the runner wrote its JSON only at the very
end, a clean four-for-four run produced **no results file at all**. Both are fixed: the call is
correct, and the runner now writes `results/e50_default_kernel.json` **after every phase**, so a
later crash cannot take the phases that already ran with it.

No gate was re-run. `results/e50_default_kernel.json` is rebuilt from run 1's own artifacts —
the greedy ids in `results/e50/engine_{avx4,serial}.json` are **re-scored** by the same decision
function, and the two `NATS_TOTAL` values are transcribed from `e50_run1.log`. Only the drift
line, which is not a gate, was executed a second time. **Run 1 remains the measurement**
(E36 run-2 rule).
