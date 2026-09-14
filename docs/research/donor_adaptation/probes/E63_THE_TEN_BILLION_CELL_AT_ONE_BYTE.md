# E63 — the 10 B cell at one byte per weight

**Verdict (Part A): `THE-PATH-EXISTS-AND-IT-IS-EXACT`.**
**`G-E63d` is `VOID` and Part B is OWED, and is stated as owed.** §12 adds an INTERIM
paired ratio, registered in addendum C before it ran and carrying an explicit
non-promotion clause: **no rate in this document is a measurement of the 10 B cell.**

Brief: `briefs/BRIEF_E63_THE_TEN_BILLION_CELL_AT_ONE_BYTE.md`, pre-registered and pushed before
a line of engine code existed. **Addendum A (A.1, A.2, A.4) pushed before the run**, re-specifying
two gates that could not be answered as written. Apparatus pushed before the run. Results:
`benchmarks/donor_adaptation/engine/results/e63_part_a.json`. **7.5 min**, one binary.

---

## 1. What was blocking

E62 removed the escape route: dBPB does not rank with N, so no fit licenses a 10 B claim and the
cell must be measured. But it could not be measured, because **the engine refused to load one**:

> `donor_engine.c:477` — *"row-selected matvec is implemented for the plain packed kind only"*
> `donor_engine.c:709` — *"matvec_colacc needs a transposed packed matrix"*

The one-byte rung (E60) and the carve (E26/E36) had never been combined, in either direction.
E63 Part A builds that path and proves it computes the right thing. **Part A is the precondition,
not the answer** — the answer is a rate, and a rate needs a quiet machine.

## 2. What was built

Two new self-describing matrix kinds in the tagged format, so every existing `quant=3`/`quant=4`
file loads byte for byte and no existing branch is edited:

| kind | layout | consumer |
|---|---|---|
| `MK_I8 = 4` | `[out, in]` int8 codes + `out` fp32 scales | `matvec_sel`, with the row list threaded through |
| `MK_I8_T = 5` | block-major `[out/blk][in][blk]`, one byte per weight along OUT, + `out` fp32 scales | new int8 branch of `matvec_colacc` |

Both leave `m->packed` at 0, which is the discriminant every consumer already branches on. The
`matvec_sel` guard relaxed from *plain packed only* to *plain packed or plain int8*; the int8
fallback loop threads the row list (`o = rows ? rows[t] : t`) with a compacted write, keeping
E61's four-accumulator structure untouched. `matvec_colacc_i8` is **simpler** than the packed
kernel, not harder: no trit decode, no even/odd split, no `pshufb` — `cvtepi8_epi32 →
cvtepi32_ps → fmadd`, 64 weights per `CA_BLK` block instead of 128.

The carved-FFN loader now **refuses a mixed FFN**: gate/up/down must all be packed or all be
int8, so a half-converted layer cannot read as a plausible rate against the wrong byte count.
That refusal is what made two of the brief's own gates unanswerable — see §6.

## 3. G-E63a — the new engine is inert on everything that already exists

Five artefacts, `donor_engine_e63.exe` against frozen `donor_engine_e61.exe`, same deterministic
prompt, **sha256 of the generated ids, zero tolerance.**

| artefact | size | generated ids | prefill logits |
|---|---|---|---|
| `05b_f32` | 1.976 GB | **identical** | identical |
| `05b_tqh` | 0.794 GB | **identical** | identical |
| `05b_i8h` | 1.041 GB | **identical** | identical |
| `15b_i8h` | 2.481 GB | **identical** | identical |
| `a10b_packed_k3` | 5.486 GB | **identical** | identical |

**`G-E63a` FIRES.** The gate asked only for the ids; **the full prefill logit block is
bit-identical too** on all five, which is the stronger statement and the one that matters —
`05b_i8h` runs through exactly the inner loop that was edited, and `a10b_packed_k3` through
exactly the `matvec_colacc` dispatch that was changed.

## 4. G-E63b — the instrument discriminates, at both shapes

E59's law: *a fidelity gate must be measured against a reference the treatment can still move.*
`k = 3` against `k = E` (every group selected) on the treatment's **own** file.

| shape | E | int8, top-1 agreement | max abs logit difference | packed, for reference |
|---|---|---|---|---|
| S05 | 16 | **0.00%** | 0.853 | 0.00% |
| A10B | 256 | **0.00%** | 4.097 | 0.00% |

**`G-E63b` FIRES** at both shapes, with room to spare — the bar was ≤ 60% and the counter reads
**zero agreement on 256 of 256 positions**. A near-100% reading would have meant the row list
selects nothing and would have made §5 `VOID`, not passed. It does not.

## 5. G-E63c — the int8 carve computes what the packed carve computes

Comparator: the packed carve at the same seed, shape and `k` (addendum A.4). `w_i8` draws from
`_codes()` in the same chunks off the same rng stream as `w_packed`, and the scales at the same
point, so the two artefacts hold **the same numbers** and differ in layout and kernel only.
Two entirely different kernels evaluate one matrix.

| shape | k | generated ids | top-1 over 256 positions | max abs logit diff | rel L2 |
|---|---|---|---|---|---|
| S05 | 3 | identical | **100.0000%** | 6.557e-07 | 3.260e-07 |
| S05 | 16 | identical | **100.0000%** | 9.285e-07 | 4.817e-07 |
| **A10B** | **3** | **identical** | **100.0000%** | **1.550e-06** | 3.887e-07 |
| A10B | 256 | identical | **100.0000%** | 1.444e-05 | 3.957e-06 |

**`G-E63c` EQUIVALENT.** The bars were ≥ 99.90% and ≤ 1e-3; the worst cell is **69× inside** the
logit bar and the top-1 count is exact everywhere. Different summation order (block-major
accumulation against row-major dot), so bit-exactness was neither available nor demanded — the
residual grows with the accumulation depth exactly as it should (1.55e-06 at k=3, 1.44e-05 at
k=256, ~9× for ~85× the accumulated terms).

**The 10 B artefact exists and is arithmetically sound: `D:/_ktmp/e63/e63_a10b_i8.bin`,
10,015,507,256 bytes, Gate V3 clean — 195 tensors matching E1's independent layout exactly.**

## 6. Two registered gates could not be answered, and were re-specified BEFORE the run

E4's precedent: *a gate that cannot be answered is MALFORMED, not failed, and may be
re-specified.* Both defects were found while building the apparatus, not while reading a result,
and both were pushed before the run they govern.

| gate | as written | why it could not be answered | replacement |
|---|---|---|---|
| `G-E63b` | int8-**dense** against packed-carved | an int8-dense synthetic artefact is not expressible: the engine refuses a mixed-format FFN, deliberately (§2) | `k=3` against `k=E` on the same carved file — **a better control**, because it exercises the carve itself, the mechanism `G-E63c` certifies |
| `G-E63c` | *"12,264 positions"* of the standard slice | that slice's ids run to 151,936 and **A10B has V = 32,768**; the count was carried from E61 without checking it against the shape E63 measures | a deterministic 256-id in-range prompt, scored at **both** shapes and **both** k |

**A.1 repaired the control and left the certification pointing at the same missing file** —
`G-E63c`'s comparator was *also* "the plain dense `MK_I8` file". Caught one gate later and fixed
in A.4, still before the run. The replacement comparator is **stronger** than the one it
replaces, which would have shared `matvec_sel`'s int8 inner loop with the treatment and tested
only the row list. Its cost is stated where it is charged: it cannot catch an error the exporter
makes *identically* in both formats, so E63 claims equivalence of the two **kernels**, not
correctness of the weights — which, on noise weights, was never on offer.

## 7. Predictions

| # | registered | read | |
|---|---|---|---|
| 1 | `G-E63a` fires on all five, zero differing bytes | all five identical, prefill too | **HIT** |
| 2 | `G-E63b` reads **< 20%** agreement | **0.00%** at both shapes | **HIT** |
| 3 | `G-E63c` reads **100.00%** and max abs diff **< 1e-4** | 100.0000%, worst 1.444e-05 | **HIT** |
| 4 | `G-E63d` reads `DESK-MODEL-OPTIMISTIC`, 32–36 tok/s | — | **OWED (Part B)** |
| 5 | `G-E63e` ratio **0.60–0.72** | — | **OWED (Part B)** |
| 6 | the int8 carved artefact is **10.9 ± 0.2 GB** | **10.016 GB** | **MISSED** |

**3 of 4 answerable. Prediction 6 is the instructive one.** I had doubled the whole packed file
(5.486 → 10.97 GB). Only the **FFN** changes width: the measured delta is **+4,529,848,320 B**,
and a first-principles count of the FFN tensors gives 4,529,848,128 — 192 B apart, on tag widths.
**0.950 GB — 17.3% of the packed file — is attention, head and embeddings, and does not double.**
Same family as the charged-vs-moved denominator: *the thing being scaled was not the whole thing.*

## 8. The anchoring hazard this run created, named before it can do damage

Part A's `--generate` runs printed decode rates. They are **conduct on a box at 34–55% busy over
24 decode tokens** and they are not a measurement of anything: they carry no dispersion, no
repetitions, no interleaving, and the box was running `graphify` for part of the sweep. They are
in the JSON because a runner records what it did.

**They are also, unavoidably, numbers I have now seen before measuring Part B**, which is the
thing `feedback_no_anchoring_producer` exists to prevent. The disciplined response is not to
pretend I did not see them:

- **Predictions 4 and 5 stay exactly as registered.** 32–36 tok/s and a ratio of 0.60–0.72.
  Revising a prediction after a glimpse of the estimand is the precise failure being guarded
  against, and a revision would be worth nothing.
- Part B's runner is written and its thresholds are frozen in source (`DESK_MODEL = 36.6`,
  `OCC_BAR = 4.39`, `REPS = 5`) **before the sweep**, and it refuses to report at all if any
  cell is over the bar.
- If Part B lands far from 32–36, that is a miss and will be scored as one.

## 9. Apparatus defects found

1. **The occupancy meter measured itself** — `generate()` called `Split.close(0)`, so `foreign`
   counted the engine's own CPU and every row printed `sys == foreign` (33.7–55.2%). Exactly
   `G-E44b`'s failure, which this programme has a standing memory about, committed again.
   **Harmless in Part A** (it gates nothing and the axis is deterministic) but **fatal to Part B
   as specified**, whose `G-E63d` admissibility is `foreign < OCC_BAR = 4.39%` — with
   `child_ticks = 0` no box could ever pass it. Fixed by reading `_process_times(p._handle)`
   before the handle is released, the mechanism `e44_interval.one_rep` already used. Part A's
   JSON keeps the mislabelled field and this line says what it is.
2. **Gate V3 fired on the first int8 export** (`+156,893,184 B` = 3 × 4864 × 896 × 24 × 0.5) —
   correct behaviour: E1's independent layout definition did not know the new kinds, and the
   delta matched the writer exactly. Fixed by teaching `nbytes_tagged`/`layout_bytes_v4`.
3. `--i8` requires `--carve` by exporter guard, because the engine refuses a mixed FFN. This is
   what made `G-E63b`/`G-E63c` unanswerable as written (§6). It is a deliberate constraint and
   is kept; the gates moved, not the check.
4. **`sweep(arms, reps=REPS)` bound `REPS` at def time**, so `--reps 9` silently ran five while
   the header printed the nine that had been asked for. Caught from the runner's own third
   output line; **the deviating run was stopped before it produced any number** and relaunched
   at the registered nine. Same family as the `CONFIG` defect: *printed as requested at the top,
   silently not applied at the bottom.*
5. **Found while scoping the follow-up, and belonging to E39, not here:** the sidecar's
   `total_weights` was written without forwarding `rank`, so every ranked artefact reports the
   dense-q/o count — `e39_r512.bin` +4.03%, `e39_r4096.bin` −4.91%, **wrong in both directions
   because the field is constant in `rank`**. E39's `G-E39A` passes the rank itself and is
   unaffected. Fixed and smoke-verified against a freshly exported ranked artefact. Full note in
   brief addendum B.5 and `probes/E39` §0.

## 10. What Part A does NOT claim

- **No rate.** Nothing here moves `SPEED_LEDGER` §60.4. Its 36.6–37.6 tok/s is still a desk
  model, and still the thing Part B will judge.
- **Nothing about quality at 10 B.** `A10B-K3` is noise weights. E63 measures an arithmetic
  equivalence, never a BPB. The trained-carve question is H1's.
- **Nothing that transfers to another shape.** E26 and §61.6's correction both say it: the carve
  penalty is a function of how much of the token is carved, and `A10B-K3` carves 11% of its
  charged weights.

## 11. Owed

1. **Part B — `G-E63d` and `G-E63e`, on an idle box.** Runner written (`e63_part_b_speed.py`,
   28 self-tests), thresholds frozen, refuses to run unless Part A's JSON says all three gates
   closed. Blocked on `COMMUNICATION.md` item 1. **This is the goal's open number.**
2. `G-E61c`'s owed repeat and the 3 B speed cells ride in the same sweep — both already wired
   into Part B, the first *reported and not adjudicated* per E40 addendum A.
3. Everything E62 §11 left owed is untouched by this probe.

---

## 12. INTERIM (addendum C) — the paired ratio, on a box that cannot answer `G-E63d`

**`G-E63d` is `VOID` and remains OWED.** Registered in addendum C **before the run**, including
the non-promotion clause: *this reading may not adjudicate `G-E63d` in either direction.* It
does not, and nothing below is a Part B result.

### 12.1 Why the box could not answer the real question

Measured, not assumed: 90 samples at 1 Hz with nothing of mine running read **median 3.28%,
p75 7.95%, max 26.30%**, 40 of 90 at or over `OCC_BAR = 4.39%`. The load is the user's browser.
In the sweep itself **every one of the 18 cells was over the bar** (4.95%–22.75%), so the
runner printed `VOID`, which is the registered verdict.

### 12.2 What the paired ratio reads

9 interleaved repetitions per arm × 120 decode tokens, `--carve-k 3`, `--threads 6`, 1.5 min.

| arm | median tok/s | bootstrap CI | range | mean foreign |
|---|---|---|---|---|
| `A10B_PACKED` (0.5 B/weight) | 47.220 | [46.250, 51.000] | 44.97–51.16 | 13.51% |
| `A10B_INT8` (FFN at 1 B/weight) | 46.750 | [45.160, 48.220] | 43.91–48.28 | 10.68% |

**Paired ratio int8 ÷ packed = 0.9900, CI [0.9106, 1.0228].**

The arms were **not** contended alike — int8 was the *less* contended by **2.82 points**, which
at E52's `k = 0.262%` per point flatters int8 by at most **0.0074** of the ratio. Correcting in
the direction that hurts the result: **0.9826**.

### 12.3 The control that makes the reading interpretable

E36 measured this exact packed artefact at **49.96 tok/s on a clean box**. At this sweep's
13.51% foreign, E52's `k` predicts **48.19**; the packed arm read **47.22** — **2.0% apart.**
The contention model, the pairing and E36's independent measurement agree, which is the reason
the ratio is worth quoting at all.

### 12.4 What it says, stated at the strength it has

**One byte per weight costs the carved 10 B model about 1.7% of its rate** — not the 30–40% I
registered, and not even the 10% its own byte count implies.

| comparator | value | measured 0.9826 is |
|---|---|---|
| my registered **prediction 5** | 0.60–0.72 | **36% above the top of the band** |
| charged-byte ratio (addendum B) | 0.8974 | **+9.5%** — *faster than its bytes* |
| what `SPEED_LEDGER` §60.4 implies against E36's packed | 0.733–0.753 | **far above** |

Composed with E36's clean 49.96 tok/s — **an estimate with both inputs named, not a
measurement** — the carved 10 B cell at one byte per weight sits at **≈49 tok/s**
(point 49.1–49.5, CI [45.5, 51.1]).

### 12.5 Why §60.4 was pessimistic, now measured rather than argued

Addendum B predicted the mechanism and the interim confirms it. §60.4 priced the one-byte rung
as though the format change applied to the **token**; it applies to **11.4% of it**. Attention
(72.3%), `lm_head` (14.5%) and the router (1.8%) are packed at half a byte in *both* artefacts
and are untouched by the change. A desk model built on a whole-token byte ratio was always going
to read ~0.73 where the truth is ~0.98.

That the measured ratio is *above* even the charged-byte comparator (0.8974) means the engine at
this shape is **not purely bandwidth-bound on the carved FFN** — the int8 kernel is cheaper per
weight (no trit decode, no `pshufb`, no even/odd split) and the gathered-weight penalty E26
priced applies to only 11.4% of the token. E31's locality question gets its first paired reading
here, and it says the 64-weight block does **not** cost what the byte count says.

### 12.6 What this does NOT license

- **It is not 50 tok/s, and it is not a measurement of the rate.** `G-E63d` is `VOID`.
- **Predictions 4 and 5 are still unscored.** They are scored against the clean sweep, by
  addendum C.3, and both are heading for a miss in the direction that is good for the project
  and bad for my forecasting. That is the registration working.
- **The goal is not met.** A ≈49 tok/s estimate with a CI that spans 45.5–51.1, composed from a
  contended ratio and a clean absolute taken on another day, is not "a 10 B model at 50 tok/s".
  One clean hour turns it into one.

---

## 13. CORRECTION to addendum B.4 — the attention lever was already measured

B.4 closed by naming attention as *"the next experiment"* if `G-E63d` lands short. **It is not a
next experiment: E39 ran it at this scale and it won.** `A10B-R512` — the same 9,999,220,736
parameters with `q_proj`/`o_proj` as rank-512 factors — reads **78.456 tok/s** (registered run 1;
run 2's 80.548 may not promote it) against matched-parameter `A10B`, **1.693× within session**,
with a fitted **attention+head+router floor of 99.4 / 100.0 tok/s** — the EXCELLENT target.

I wrote B.4's forward-looking sentence **without opening E39**, which is
`feedback_search_before_claiming_a_gap` in its milder form: not *"nobody measured this"* but
*"this is what to measure next"*, about something already measured. **A statement about the
programme's conduct passes through no gate**, which is precisely why the rule exists.

**Nothing in B.4's arithmetic is withdrawn, and the correction makes it stronger** — the 72.3%
attention share is the *mechanism* for E39's 1.69×. What it also shows is that E63's 1.7%
one-byte cost **does not transfer** to R512, whose FFN is 20.91% of the charged token against
`A10B-K3`'s 11.44% (charged-byte ratio 1.2091 vs 1.1144). That is a separate question with a
separate brief. **`G-E63d` remains `VOID` and OWED, and §13 adjudicates nothing.**

