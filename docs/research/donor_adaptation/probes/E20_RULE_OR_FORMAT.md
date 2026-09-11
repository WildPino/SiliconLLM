# E20 — is it the rule, or the format? `RULE-EXHAUSTED-DRIFT-DOMINATES`

**Brief**: `briefs/BRIEF_E20_RULE_OR_FORMAT.md` @ `e5d438c`, pushed before any arm ran.
**Runners**: `benchmarks/donor_adaptation/ternary/e20_rule_or_format.py` (part A),
`e20b_teacher_forced.py` (part B).
**Results**: `engine/results/e20_rules_on_the_head.json`, `engine/results/e20b_teacher_forced.json`,
and `engine/results/e20_rule_or_format.json` (run 1, **VOID**, kept as the record).
**Machine**: 3600X, 6 threads, CPU only. Part A 3449 s, part B 459 s. **No timing taken.**

---

## 0. The two findings

**A. The rule axis is exhausted and it was never the constraint.** Eight ternary heads — every
rule this repository owns plus two built for E20 — one format, same bytes, same kernel. The best
ternary head anyone here can build costs **`+0.170414` BPB** over the intact donor and sits
**`3.131810` below the chance line**. It agrees with its own donor on **`11` of `160`** greedy
positions, *below* the `12/160` constant-`'\n'` floor. **No arm reaches the `RANKS` band.** E16's
promotion of the constraint from the rule to the format now rests on **six rules, not two**, and
E16 §9's qualifier is discharged on this tensor.

**B. The instrument that says so is measuring two things, and the smaller one is per-step damage.**
With the donor's own context held fixed, **every ternary head still puts the donor's token first at
`67`–`74`% of positions**. Free-running greedy reads `6`–`26`%. The difference — **`75`–`106`
tokens** — is **autoregressive drift**, not argmax failure. An exact identity (§5, `G-B1`, `50/50`)
ties the two harnesses together, so this is arithmetic and not a story.

Both findings point the same way for the goal and neither moves a speed number: a head that
survives two steps in three does not survive thirty-two. But the *description* changes. The
programme has been writing "these models cannot choose a token". The measured statement is
**"these models choose the donor's token two times in three, and cannot survive the third."**

---

## 1. Run 1 is VOID, and the gate that voided it caught the brief's own premise

Recorded first because it is the load-bearing correction. Committed at `6c7b7a4`.

`BRIEF_E20` §0 states that *"every ternarization this programme has ever run is a weight-space rule
that never looks at a single token"*, citing `t1_ternarize.py:97-98`. **That is `R0`.** It is not
the shipped rule. The shipped rule is `qwen_export.quantize` under `--rule R3`:

```
qwen_export.py:84    elif rule == "R3":
qwen_export.py:85        assert act_rms is not None, "R3 needs calibration activations"
qwen_export.py:86        q, a = T2.r3_actsearch(w, act_rms)
qwen_export.py:148   hl = m.lm_head.register_forward_hook(mk(("head", "head")))
```

`t2_rules.r3_actsearch` is a per-row threshold search minimising the **activation-RMS-weighted**
error over the calibration slice, and line 148 hooks `lm_head` — **the shipped head quantization
already looks at tokens.** Run 1's `R3H` arm was therefore `R0` applied to the head, and `G-Q2` —
registered in the brief as *"replicate T2b/E18: BPB `1.106584` to `< 1e-5` and `9/160`"* — returned
`1.319900` and `17/160`. **VOID, exactly as registered.**

**And GPTQ was not untried either.** `t2_rules.r4_gptq` has existed since T2, which ran it on the
FFN and published the result in `probes/T2_TERNARIZATION_RULE.md` §§3, 7:

| T2 arm | what it is | BPB | Δ vs base | note |
|---|---|---|---|---|
| `R3` | act-RMS-weighted threshold search | `2.476967` | `+1.709372` | T2's pre-registered winner, **the shipped rule** |
| `R4` | GPTQ, `mean|w|` scale | `4.299819` | `+3.532224` | **above** the chance line `4.069819` |
| `R5` | GPTQ, act-searched scale | `2.027495` | `+1.259900` | **the best rule ever measured here** |

T2 §3 marks R5 `⚠ POST-HOC and gates nothing` because its brief §4 fixed `best` as the minimum over
R1–R4. **That was correct discipline and is not being second-guessed.** Nor was any of this buried:
`INDEX.md` §3 carries T2's decomposition table with the row *"GPTQ error compensation **on a
well-placed grid**, `−0.449`, significant: yes"* — **in a table this programme maintains and I
wrote the brief against.** The failure was not of the record; it was of reading it.

The consequence, however, is concrete and had gone unstated: **the best ternarization rule this programme has measured is not
implemented in the exporter** — `qwen_export.quantize` dispatches `R0`/`R1`/`R2`/`R3` and has no
`R5` branch — **and, until E20, had never been read in ranking at all.**

**What run 1 measured anyway** (bands unchanged, base and `ID` gates fired):
`R0H` `1.319900` → `17/160`; `OPTH` `1.293924` → **`42/160`**; `GPTQH` `0.938018` → `11/160`.
Both surprises that drive §§3–5 were already visible in the void run, **before** the arm list was
rebuilt. Nothing below is a verdict chosen after the fact.

**Run 2 therefore imports every rule from `t2_rules` rather than restating any of them** — *"one
definition each"*, `qwen_export.py:74-76` — and its `R3H` arm is
`t2b_organs.apply_arm(model, "H", act_rms, None)` itself, so `G-Q2` compares like with like.

---

## 2. Instrument — what is held fixed, and the three gates that fired

**The format is held exactly at what `QWENDON1` stores**: ternary codes `{−1, 0, +1}` and **one
fp32 scale per output row**, same bytes, same kernel, `0.500000` B/weight. Only the choice of codes
and scales changes. Head only (`lm_head`, 151936×1536); `tie_word_embeddings = True` on this donor,
so the head is untied and cloned first and the embedding stays fp32. **Every other weight is
bit-exact in every arm.**

| gate | what it demands | result |
|---|---|---|
| `G-Q0` | `base` reproduces `results/e6/ref.json` | **FIRES** — `160/160`, BPB `0.767595` |
| `G-Q1` | `ID` token-identical to `base`, BPB diff `0` | **FIRES** — identical, diff `0.000e+00` |
| `G-Q2` | `R3H` replicates T2b (`1.1065836079824596`) **and** E18 (`9/160`) | **FIRES** — `1.1065835970951252`, **`abs diff 1.09e-08`**, and **`9/160`** |

`VOID: none`. Two published anchors from two different probes, both hit, before any new arm was
read.

Calibration is the frozen 32×512 seed-42424 slice (16,384 tokens), asserted to be a different
corpus half from the eval slice; eval is the frozen 24×512 heldout, `ids_sha256`
`a1a48dc9…d52f65`, 51,870 scored bytes. A free consistency check on the two statistics captured in
the one pass: `act_rms` must equal `sqrt(diag(H)/n)` identically, measured max abs difference
**`3.81e-06`** (fp32 round-off), which proves both came from the same tokens.

---

## 3. Part A — eight ternary heads, one format

Bands fixed before run 1 and unchanged since E17/E18: floor `12` (E18 part A, best constant-token
predictor), margin `2` (E17), **`AT-FLOOR ≤ 14`**, **`RANKS ≥ 80`**, ceiling `160`.

| arm | what chooses the codes | knows the data? | BPB | Δ vs base | vs chance `4.069819` | zero frac | greedy | band |
|---|---|---|---|---|---|---|---|---|
| `base` | — | — | `0.767595` | — | `−3.302224` | — | **160/160** | — |
| `ID` | identity through the same path | — | `0.767595` | `0.000000` | `−3.302224` | — | **160/160** | — |
| `R0H` | `mean|w|`, RTN (`t1_ternarize`) | no | `1.319900` | `+0.552305` | `−2.749919` | `0.3124` | 17/160 | PARTIAL |
| `R1H` | TWN, `Δ = 0.7·E|w|` | no | `1.280712` | `+0.513117` | `−2.789107` | `0.4262` | 13/160 | AT-FLOOR |
| `R2H` | threshold search, unweighted L2 | no | `1.288465` | `+0.520871` | `−2.781354` | `0.4759` | **41/160** | PARTIAL |
| **`R3H`** | **act-RMS-weighted search — SHIPPED** | yes | `1.106584` | `+0.338989` | `−2.963235` | `0.4478` | 9/160 | AT-FLOOR |
| `R4H` | GPTQ, `mean|w|` scale | yes | `1.031616` | `+0.264021` | `−3.038203` | `0.2877` | 15/160 | PARTIAL |
| `R5H` | GPTQ, act-searched scale | yes | `0.940203` | `+0.172608` | `−3.129616` | `0.4276` | 9/160 | AT-FLOOR |
| `OPTH` | scale grid, unweighted L2 (E20's) | no | `1.293924` | `+0.526329` | `−2.775895` | `0.4661` | **42/160** | PARTIAL |
| **`GPTQH`** | GPTQ, H-diagonal-searched scale (E20's) | yes | **`0.938009`** | **`+0.170414`** | `−3.131810` | `0.4234` | 11/160 | **AT-FLOOR** |

**The registered verdict arm reads `AT-FLOOR`.** `GPTQH` is the best ternary head in the table by
BPB and the fourth-worst by ranking. The brief's registered alternative — *"if `GPTQH` comes back
`RANKS` or `PARTIAL`, the format is not the constraint and the conversion route reopens"* — **does
not fire**: `11/160`, below the `12/160` floor, diverging at token 2.

**`G-Q4` (sanity, not a verdict) holds on its own terms.** The best-BPB arm *is* data-aware, and
`R5H − R3H = −0.166380` here against `−0.449472` on T2's FFN: same sign, smaller magnitude. The
data-aware machinery works on the objective it optimises. **The failure is not an implementation
failure.**

**And the inversion, third occurrence and the strongest yet.** Across the eight converted arms,
**`r(BPB, agreement) = +0.6238` over a BPB span of `0.381891` — the wrong sign.** The three
best-scoring heads (`GPTQH` `0.938009`, `R5H` `0.940203`, `R4H` `1.031616`) score `11`, `9`, `15`.
The two best-*ranking* heads (`OPTH` `42/160`, `R2H` `41/160`) are both **data-free unweighted L2
searches** and are the worst-scoring of the search family. Compare E18 (`+0.4989`) and E19
(`−0.8562`, the expected sign): **inside the ternary regime the sign of this correlation is not
stable, and E17 §9's finding that BPB has resolution where rank does not is now joined by its
converse.**

---

## 4. The `PARTIAL` band is real, and it is one prompt

E14 §5 item 3 has owed an intermediate ranking point since 2026-09-07; E17, E18 and E19 each failed
to supply one. **E20 supplies four** (`17`, `41`, `15`, `42`). Before reading them as partial
competence, the per-prompt split:

| arm | p0 | p1 | **p2** | p3 | p4 | total |
|---|---|---|---|---|---|---|
| `R2H` | 1 | 3 | **32** | 2 | 3 | 41 |
| `OPTH` | 2 | 2 | **32** | 3 | 3 | 42 |
| `R0H` | 3 | 1 | 5 | 0 | 8 | 17 |
| `R4H` | 3 | 0 | 7 | 1 | 4 | 15 |

**`R2H` and `OPTH` reproduce prompt 2 whole — all 32 tokens — and sit at floor noise on the other
four.** The label `PARTIAL` is true of the aggregate and false of the model: this is not 26%
competence spread evenly, it is one prompt at 100% and four at chance.

**Prompt 2 is not degenerate.** Its 32 donor tokens contain **18 distinct ids** and read
`" 212 °F or 100 °C and ice melts at 32 °F or 0 °C. If the temperature of"`. A ternary head chosen
with **no data at all** reproduces that exactly. It is the first whole-prompt reproduction by any
converted arm in this programme.

**And prompt 2 is the only prompt with no near-tie anywhere in it.** Mean and minimum top-2 logit
gap of the donor, from `results/e6/ref.json`:

| prompt | mean top-2 gap | median | **min** | best arm's score |
|---|---|---|---|---|
| 0 | `2.0283` | `1.6767` | `0.0739` | 4/32 |
| 1 | `2.9856` | `2.4771` | `0.0453` | 3/32 |
| **2** | **`6.4656`** | **`6.6464`** | **`1.1630`** | **32/32** |
| 3 | `3.4633` | `2.5401` | `0.2740` | 3/32 |
| 4 | `3.9876` | `2.6788` | `0.0081` | 8/32 |

Every other prompt dips to a margin between `0.0081` and `0.2740` somewhere in its 32 positions.
**One near-tie is enough to end the match**, because everything after it is a comparison between
two different contexts. That is the observation part B was built to test.

**This also bounds the instrument.** 160 positions are not 160 independent trials; they are **five
trials of thirty-two**, and `41/160` differs from `9/160` largely by which prompt survived. Every
ranking number this programme has published since E7 carries an effective *n* of **5**.

---

## 5. Part B — separating per-step fidelity from drift

**Unregistered.** This metric was built after reading part A. Under E14 §6 it **reports and does not
decide**; the verdict in §0A is part A's, against bands fixed before run 1.

Each arm is teacher-forced through the donor's own reference continuation, so all 160 positions are
independent tests of the head's argmax with the context held identical.

| arm | BPB | free-running | **teacher-forced** | drift cost | mean rank of donor token | rank ≤ 5 |
|---|---|---|---|---|---|---|
| `base` | `0.767595` | 160/160 | **160/160** | 0 | `1.00` | 160 |
| `ID` | `0.767595` | 160/160 | **160/160** | 0 | `1.00` | 160 |
| `R0H` | `1.319900` | 17/160 | **107/160** (66.88%) | −90 | `16.92` | 140 |
| `R1H` | `1.280712` | 13/160 | **114/160** (71.25%) | −101 | `12.85` | 139 |
| `R2H` | `1.288465` | 41/160 | **116/160** (72.50%) | −75 | `13.96` | 139 |
| `R3H` | `1.106584` | 9/160 | **110/160** (68.75%) | −101 | `2.98` | 152 |
| `R4H` | `1.031616` | 15/160 | **115/160** (71.88%) | −100 | `2.40` | 148 |
| `R5H` | `0.940203` | 9/160 | **112/160** (70.00%) | −103 | `2.71` | **155** |
| `OPTH` | `1.293924` | 42/160 | **119/160** (74.38%) | −77 | `13.26` | 140 |
| `GPTQH` | `0.938009` | 11/160 | **117/160** (73.12%) | −106 | `2.88` | 151 |

`base` and `ID` read `160/160` exactly, so the instrument fires on the known-positive before any
null is read.

**`G-B1` — an exact identity, and it holds `50/50`.** If an arm matches the donor at every position
`< k` under teacher forcing, then free-running it emits those same tokens, the contexts agree, and
it must first diverge at exactly `k`. Checked across 10 arms × 5 prompts: **50 of 50 agree**. The
two harnesses are mutually consistent and the drift account is arithmetic.

**What the table says.**

1. **Per-step argmax fidelity is `67`–`74`% for every arm and the spread between arms is 12
   tokens.** Free-running spreads them over 33. Most of what the ranking metric has been measuring
   since E7 is *where the first miss happened to fall*, not how damaged the head is.
2. **The two families separate on mean rank, in the direction the theory predicts.** Data-aware
   arms put the donor's token at mean rank `2.40`–`2.98` (rank ≤ 5 on 148–155 positions);
   weight-space arms at `12.85`–`16.92` (139–140). **The Hessian- and activation-weighted
   objectives do exactly what they promise — they preserve the logit geometry** — and that buys
   BPB and neighbourhood, not the top-1/top-2 boundary, which is where argmax lives. That is the
   mechanism of §3's inversion, and it is measured, not asserted.
3. **Drift is sufficient to explain the free-running numbers.** A per-step survival rate `p` gives
   an expected free-running agreement of `5·(1 − p³²)/(1 − p)`: `15.1` to `19.5` across the eight
   arms. Observed: `9, 11, 13, 15, 17` for six of them. `R2H` `41` and `OPTH` `42` are the two
   outliers and are entirely prompt 2, whose margins are not average.
4. **Argmax survival is governed by the donor's own margin.** Splitting the 160 positions into
   terciles of the donor's top-2 gap:

| donor top-2 gap | `R0H` | `R3H` | `R5H` | `OPTH` | `GPTQH` |
|---|---|---|---|---|---|
| `0.0081`–`1.7294` (n=53) | 20 | 24 | 22 | 25 | 24 |
| `1.7358`–`4.3545` (n=53) | 35 | 35 | 38 | 41 | 40 |
| `4.4439`–`15.9636` (n=54) | **52** | **51** | **52** | **53** | **53** |

  **In the top tercile every arm is at 94–98%. In the bottom tercile every arm is at 38–47%.** The
  ternary head is nearly transparent where the donor was sure and close to a coin-flip where it was
  not. Prompt 2 is simply a continuation made entirely of top-tercile positions.

---

## 6. What this does and does not change

**Does not change.** **`6.79 tok/s` is untouched and no timing was taken** — same format, same
bytes, same kernel; a rule change moves no byte. E19's arithmetic stands: FFN-only carving cannot
reach 50 tok/s at 7 B or above. The goal is not closer.

**Does not change.** **The donor route still fails.** 70% per-step fidelity compounds to nothing
over 32 steps, and the best head available costs `+0.170414` BPB for `11/160`. Nothing here
reopens the conversion route, and the registered alternative that would have reopened it did not
fire.

**Changes.** **The claim "ternary conversions cannot choose a token" is too strong and should be
retired in that form.** They choose the donor's token about two times in three at fixed context,
and rank it 2nd–3rd on average when they miss. What they cannot do is survive their own first
mistake. Every probe from E7 onward that leaned on `0/160`-style numbers was reading a compound of
argmax damage and drift, and §5 now separates them.

**Changes.** **E16 §9's rule qualifier is discharged on this tensor.** Six rules, two of them the
literature's strongest post-training quantizers, one format. The best is `0.170414` BPB from the
intact donor and still below the floor. The constraint is the format.

**Changes — actionable.** **`R5` is not in the exporter.** It beats the shipped `R3` by `0.166380`
BPB on the head here and by `0.449472` on T2's FFN. Adding an `R5` branch to
`qwen_export.quantize` would improve **every BPB this programme quotes for a converted artefact**
and, on this evidence, **change no answer** (`R5H` and `R3H` both read `9/160`). It is recorded
here precisely so that nobody later fixes the exporter and believes they fixed the model.

---

## 7. Predictions — one held, three missed

| # | brief §5 | outcome |
|---|---|---|
| 1 | `G-Q0`, `G-Q1`, `G-Q2` all fire | **held** in run 2; `G-Q2` **VOIDed run 1**, which is the gate earning its keep |
| 2 | `G-Q4` holds; Hessian arm wins BPB by `≥ 0.05` over `R3H` | **held** — `GPTQH − R3H = −0.168575`, `R5H − R3H = −0.166380` |
| 3 | **`GPTQH` → `AT-FLOOR`** | **held** — `11/160` |
| 4 | `OPTH` between `R3H` and `GPTQH` in BPB, `AT-FLOOR` in ranking | **missed twice**: `OPTH` `1.293924` is **worse** than both, and it read **`42/160`, PARTIAL** — the best-ranking arm in the table |

Prediction 3 held, which is the verdict. **The premise underneath all four was wrong** (§1), and
nothing in §5 was predicted at all — the drift/argmax separation was not in the brief, is
unregistered, and decides nothing here.

**The pattern from E18 continues.** What protected this reading was not calling the direction; it
was registering `G-Q2` against two published numbers from two different probes. Without it the
brief's false premise would have been published as a measurement, with `R0` labelled as the
shipped rule and T2's R4/R5 unmentioned.

---

## 8. Owed

1. **Healing / QAT** — unchanged as the first owed item since E18 §9, and §5 sharpens the target
   considerably: the thing to repair is a head that is already right 70% of the time per step.
   **Needs GPU; the user launches it.**
2. **Re-read every published ranking number as a compound.** E7, E16, E17, E18, E19 all quote
   free-running greedy agreement. §5 shows it is per-step fidelity convolved with drift over an
   effective *n* of 5. None of their verdicts is withdrawn — each is still a true statement about
   free-running behaviour, which is what a runnable model does — but the mechanism attributed to
   them needs restating.
3. **A ranking instrument with real resolution.** Five prompts is an effective *n* of 5. Teacher-
   forced top-1 over a few hundred positions is cheap (`459 s` for ten arms), has no drift, and
   fires exactly on the known-positive. It should become the default second metric.
4. **`R5` in the exporter** — §6. A branch in `qwen_export.quantize`, then re-quote.
5. **The rule axis on the FFN in ranking.** T2's six rules were read in BPB only; E20 read them on
   the head. The FFN is 80.7% of the weights and has different conditioning.
6. **Unchanged from E19**: carving attention and the head; the D0 §III / D0c §5 re-read; the cliff
   at 7 B.

---

## 9. Appended 2026-09-11 after E21 — the teacher-forced band, used once, and what it caught

§8 item 3 asked that teacher-forced top-1 become the standing second metric. E21 is the first
experiment to use it as a **pre-registered band**: `> 119` = `RANK-IS-CHEAPER`, `107`–`119` =
`RANK-IS-COMPARABLE`, `< 107` = `RANK-IS-WORSE`, taken directly from §5's eight ternary heads.

**It separated what BPB could not.** `H-ACT-256` scores BPB `1.330385` — within `0.011` of `R0H`'s
`1.319900` in §3 — and reads **68/160** teacher-forced against `R0H`'s **107**. Two interventions
indistinguishable in BPB, thirty-nine tokens apart in per-step fidelity. **§6's lesson generalises
in the other direction too: the two metrics are not substitutes, whichever is quoted.**

**And one arm cleared the band from above.** `QO-ACT-512` (activation-weighted rank-512 on
`q_proj`+`o_proj`) reads **144/160**, above every ternary head measured here, at **`+0.052689`**
BPB against the best ternary head's `+0.170414`. That is the first time any intervention in this
programme has beaten the ternary band rather than failing to reach it.

**§5's drift account transfers intact.** `QO-ACT-512` is 144 teacher-forced and 26 free-running —
a drift cost of 118, in the same `75`–`106`+ range, from a *much* higher per-step rate. The drift
model is a property of 32-step greedy generation, not of the damage.

**§8 item 1 is retargeted.** "Repair a head that is already right 70% of the time per step" becomes
**"repair low-rank attention that is already right 90% of the time per step"** — a better-posed
optimisation, and the head is now the named constraint rather than the healing target.
