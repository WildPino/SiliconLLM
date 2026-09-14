# E61 — the chain was not the binder, and the control that proved it was mine

**Verdict: `THE-CHAIN-WAS-NOT-THE-BINDER`.**

Brief: `briefs/BRIEF_E61_THE_CHAIN_ON_THE_BYTE_THAT_WORKS.md`, pushed `ab2a537` before a runner
existed. Apparatus pushed `f9fa2c0` before the run. Runner `engine/e61_one_chain.py` (35/35
self-tests), results `engine/results/e61_one_chain.json`, log `engine/e61_run.log`.
Run: **65.1 minutes**, 2026-09-14 01:31:56 → 02:37:02, one binary (`donor_engine_e61.exe`).

---

## 1. What was asked, and the answer

E8 wired `--mvacc` into `matvec_sel`'s packed and fp32 branches in 2026-09-07 and skipped the
third, writing down why: *"not on the measured path, so not urgent."* E60 made it the measured
path — it is the int8 rung, the only artefact this programme has that is both above the 50 tok/s
bar and faithful. The assembly still read one unbroken `ymm0→ymm1→ymm0` chain, 4 FMAs per 32
weight bytes (`.LBB17_77`).

**I paid the debt. It was worth 5%, not the 19% I derived.**

| | `--mvacc 1` | `--mvacc 4` | paired median | paired 95% CI | registered verdict |
|---|---|---|---|---|---|
| **`05b I8`** | **65.13** | **68.52** | **1.0507** | 1.0482–1.0706 | **REFUTED** (floor 1.06) |
| **`15b I8`** | **20.61** | **22.56** | **1.0662** | 0.9936–1.1768 | **UNDECIDED** |

Predicted 1.12–1.22 and 1.10–1.20. At `05b` the effect is real and resolvable — the CI excludes
1.0 — and it is **small**: a fifth of what the derivation demanded. At `15b` the paired CI
straddles 1.0, so this sweep cannot resolve it at all.

## 2. The control that killed my own mechanism

`G-E61b`, the planted positive, is not decoration here. It is what refutes §3.2 of the brief.

| `05b`, `--mvacc 4 ÷ 1` | m1 GB/s | its single-chain ceiling | fraction of it | **gain from 4 chains** |
|---|---|---|---|---|
| `PACKED` — the planted positive | 16.47 | 18.2 | **90.5%** | **×1.2745** |
| `I8` — the headline | 32.29 | 36.4 | **88.7%** | **×1.0507** |

**Two kernels sitting at the same fraction of their own computed latency wall; one gains 27%, the
other 5%.** The brief's whole discriminating observation was *"every other weight kernel runs at
under a third of its latency wall, the int8 branch runs at 87–91% of its"* — and its own control
arm turns out to sit at 90.5% of its own wall too. **Fraction-of-chain-ceiling predicts nothing.**
The table was arithmetic that happened to be true of two kernels and explanatory of neither.

`G-E61b` **FIRES** at **1.2745** (E8's G-Z3 read 1.2301 on a binary with the slow attention
default). The instrument was alive; the null is not a dead-instrument artefact.

## 3. What actually binds: §3.3's third alternative, the one I ranked last

The brief listed three ways it could be wrong. The winner was alternative 3: *"memory genuinely
binds at ~32 GB/s for a 1 B/weight stream… a finer stream is not obliged to reach the same rate."*

**E61's real product is E60's curve, re-measured with every kernel running four chains** — which
is the exact criticism E61 was launched to make of E60, namely that its byte-ladder table compared
a 4-chain packed arm against a 1-chain int8 arm:

| B/weight | arm | E60 (int8 at 1 chain) | **E61 (all at 4 chains)** |
|---|---|---|---|
| 4.0 | `05b F32` | 37.70 GB/s | **37.64** |
| 1.0 | `05b I8` | 31.67 | **33.97** |
| 1.0 | `15b I8` | 33.09 | **34.90** |
| 0.5 | `05b PACKED` | 21.48 | **20.65** |

Equalising the kernels moves the middle point up ~6% and changes nothing else. **The ordering
survives, the shape survives, and the ladder's verdict survives.** `SPEED_LEDGER` §59.2's
middle denominator is corrected from **31.7–33.1** to **34.0–34.9 GB/s** and is *not* withdrawn:
it is now a property of the byte stream measured at matched kernel quality, not an artefact of
unequal kernel quality. E61 set out to undermine §59 and ended up load-bearing for it.

### 3.1 What it buys, in the units of the goal

| | E60 | **E61** |
|---|---|---|
| `05b` faithful arm | 63.87 tok/s | **68.52** |
| `15b` faithful arm | 21.39 | **22.56** |
| `I8 ÷ F32` at `05b` | ×3.35 | **×3.60** |
| 10 B desk model, 1 B/weight | 34.1–35.6 tok/s (68–71% of bar) | **36.6–37.6 (73–75%)** |
| faithful-to-bar gap at 10 B | ×1.40 | **×1.33–1.37** |

The 10 B line is E36's `A10B-K3` cell — 0.9283 G charged weights/token, **noise weights** (E36 §0)
— divided by §3's measured 1 B/weight bandwidth. **Desk model, not a measurement**, and the
qualification E60 attached still holds: on a *dense* 10 B shape int8 is strictly worse than
ternary, so this number only exists on the sparse target shape, which as a trained artefact does
not exist.

## 4. Fidelity: free, and measured on the instrument E60 owed

`G-E61e` **FAITHFUL**. The metric is the replacement E60 §7 item 5 registered — per-position top-1
under teacher forcing over all 12,264 positions of the standard slice — and it was **registered in
E61's brief before the run that used it**, not assembled afterwards.

| | top-1 vs `--mvacc 1` | dBPB |
|---|---|---|
| `05b I8` | **12264/12264 = 100.0000%** | **+3.27e-09** |
| `15b I8` | **12264/12264 = 100.0000%** | **−5.77e-09** |

`G-E61d` **ADMISSIBLE**: rel L2 of the 4-chain arm against the 1-chain arm **2.4222e-06** (`05b`)
and **3.2506e-06** (`15b`), against E8's unchanged `≤ 1.0e-05`. Inside the predicted band, and at
or under E8's packed 3.3e-06 — int8 codes are exactly representable in fp32, so only the summation
order moves.

### 4.1 `G-E61e0` — the control that makes a 100% reading mean anything

A counter pinned at 100% is on a **ceiling**, and E59's law says a saturated counter cannot show
damage in either direction. So the metric's own control ran **first**:

| pair | required | read |
|---|---|---|
| `F32` run A vs run B (two independent invocations) | 100.000% | **12264/12264 = 100.000000%** |
| `PACKED` vs `F32` | < 50% | **281/12264 = 2.29%** |

`BRACKETS`. The counter has 12,264 positions of resolution instead of E60's 160, pins exactly at
the top when nothing changed, and falls to 2.3% on a known-broken arm. **This is the replacement
for `G-E60e` and it is in service.**

Two incidental readings from that control:

- `05b F32` BPB reproduced E60's **0.871810465558026 to |d| = 0.00e+00**, twice, 710 s apart.
- `05b PACKED` read **4.531246373** against E1's **4.531233734** — **|d| = 1.26e-05**, the first
  re-measurement of that number. E1 predates both E8's `mvacc=4` on the packed branch and
  E49/E50's attention kernel, and both changed summation order. That 1.26e-05 is the accumulated
  float-rounding of every summation-order change since E1. **No attribution of shares is claimed**
  — E8 gated on rel L2 and greedy tokens and never took a BPB, so nobody knows the split.

## 5. The null control went out of band, and this is the run's real defect

**`G-E61c` read 1.0947** where E8's G-Z5 read **1.0029**. The registered null band was 0.97–1.05
and the refutation threshold was 1.10. **It missed refuting my own mechanism by 0.0053.**

The proximate cause is in the conduct, and the conduct is the defect:

| `05b` cell | foreign, mean | cells over `OCC_BAR = 4.39` |
|---|---|---|
| **`F32 m1`** | **19.57%** | **9/9** |
| `F32 m4` | 11.28% | 9/9 |
| `PACKED m1` / `m4` | 9.50% / 11.07% | 5/9 / 7/9 |
| `I8 m1` / `m4` | 5.60% / 4.44% | 4/9 / 5/9 |
| `T1 m1` / `m4` | 8.66% / 9.17% | 6/9 / 8/9 |
| `15b I8 m1` / `m4` | 9.82% / 11.78% | 8/9 / 9/9 |

**This sweep was dirtier than E60's** (4.4–19.6% against 3.7–9.7%) and the contamination inside
the null arm was **asymmetric by 8.29 points**, which at E52's `k = 0.262%` of rate per point
accounts for ~2.2 points of the 9.5. That leaves ~1.07 — still outside 0.97–1.05.

**Two things follow, and the second one matters more.**

1. **The headline survives the criticism, in the direction that matters.** `I8`'s cells were the
   cleanest in the sweep and its `m1` was contended *more* than its `m4`, so 1.0507 is an
   **over**estimate; correcting gives ~1.048. REFUTED gets firmer, not weaker.
2. **`G-E61c` is owed a repeat on an idle box, and it will not be re-run to a pass.** E40 addendum
   A: a gate that fires is doing its job. This one did not fire — it read a zone the brief
   deliberately left unadjudicated — and the runner printed exactly that: *"OUT-OF-BAND — the
   brief registered no verdict here; reported, not adjudicated."* If the repeat also reads ~1.07,
   then **E8's own 1.0029 is the number in question**, not this one: E8 ran before `--attn avx4`
   became the default, so the fp32 weight path is a larger share of the token today than it was
   then, and a small fp32 weight-path gain would show up now and not then. That is a hypothesis
   with a cheap test and it is owed, not asserted.

## 6. `G-E61g` — E60's value-dependence did not replicate

E60 left **1.98 tok/s** unexplained between `I8` (63.87) and `T1` (61.89) at identical bytes
through an identical kernel — the reading that refuted E60's own *"speed is a function of bytes,
not values"*.

| | `I8` | `T1` | gap | `T1` min..max |
|---|---|---|---|---|
| `--mvacc 1` | 65.13 | 64.45 | **+0.68** | 56.9..65.4 |
| `--mvacc 4` | 68.52 | 67.64 | **+0.88** | 63.1..69.0 |

The gap did **not** close when the chain was broken — which is the half of prediction 7 that held,
and it rules the chain out as its cause. But it is **2.9× smaller than E60's**, on a `T1` arm whose
min-to-max spread was **8.5 tok/s**. **This sweep cannot resolve the question either way.**
E60 §7 item 2 stays owed, and the descriptive framing the brief gave this gate was the right one.

## 7. Predictions: 4.5 of 7

| # | predicted | read | |
|---|---|---|---|
| 1 | sha256 identical at both sizes | `1057025199fbe92d` / `1df3baeee7e9ec40`, both | **HELD** |
| 2 | packed fires, 1.15–1.30 | **1.2745** | **HELD** |
| 3 | fp32 null, 0.98–1.04 | **1.0947** | **WRONG** |
| 4 | rel L2 in 1e-07…5e-06 | 2.42e-06 / 3.25e-06 | **HELD** |
| 5 | **1.12–1.22 and 1.10–1.20** | **1.0507 / 1.0662** | **WRONG** |
| 6 | top-1 ≥ 99.7%, \|dBPB\| ≤ 1e-04 | 100.0000%, ~3e-09 | **HELD** |
| 7 | the `I8`−`T1` gap does not close | did not close, but is 2.9× smaller and unresolvable | **PARTIAL** |

**Prediction 5 is the one the brief said could embarrass it, and it did** — the same way it
embarrassed E8, which predicted 1.5–1.7 and measured 1.349. Twice now a chain-latency derivation
has pointed the right way and overstated the size, because removing the chain hands the kernel to
whatever is next rather than to the number the derivation names. **The derivation predicts a
direction. It has never once predicted a magnitude.** That is now a two-experiment record and it
should be written into the next brief that tries it.

## 8. What E61 changed in the engine, and what it did not

Kept, because it costs nothing and pays a debt that was five probes old:

- `matvec_sel`'s `quant==1`/`quant==5` branch honours `g_mvacc`. **`--mvacc 1` is byte-for-byte the
  old loop** — `G-E61a` FIRES, sha256-identical to `donor_engine_e60.exe` at both sizes — so every
  E60 number reproduces, and the default 4 is worth ×1.05 on the faithful arm at zero fidelity
  cost (100.0000% top-1, dBPB ~3e-09).
- `--bpb --top1 <file>`: the argmax per predicted position, int32 LE. NATS is **bit-identical**
  with the flag on or off.
- `one_cell` **refuses** any cell whose `CONFIG` line does not report the `mvacc` it asked for.

Not changed: packed stays the default format, `--attn`/`--fexp`/the witness/the LUT kernels are
untouched, and **the byte ladder's verdict is unchanged** — `THE-CLIFF-IS-BELOW-ONE-BYTE` stands.

### 8.1 The config-line defect, closed

`CONFIG` printed `mvacc=4` on E55, E57 and E60 cells whose kernel was running one chain. Not a
runner forgetting a flag — the flag was defaulted precisely so nobody could forget it
(`donor_engine.c:76`), printed, and recorded in three probes, and **silently not honoured by the
kernel it named**. A third form of [[feedback_config_must_appear_in_output]]: not a config that
fails to print, but a config that prints a value the code path does not implement. E60's numbers
are not retracted — the branch ran one chain consistently in all 9 reps of all 4 of its cells — but
the label on them was wrong, and the ×1.05 measured here is the size of the error that label hid.

## 9. Owed

1. **`G-E61c` on an idle box** (§5), with the *"E8's 1.0029 predates `--attn avx4`"* hypothesis as
   the thing it tests. Cheapest item on the list.
2. **A sweep that meets `OCC_BAR`.** Nine of ten cell-groups here ran over it. E56's zero point for
   `OCC_BAR` is the same problem and is still open.
3. The 10 B cell at 1 B/weight, **measured** (E60 §7 item 4) — now worth more, since §3.1's desk
   model moved.
4. A third scale for the 0.5 B → 1.5 B dBPB growth segment (E60 §7 item 3).
5. `C/T` measured on the donor engine (E60 §7 item 6).
6. Channel-granularity mixed precision (E60 §7 item 7).
7. E60 §7 item 2 — the `I8`−`T1` value-dependence — **back to unresolved** (§6), not closed.
8. `G-E55a2`'s replacement interval gate; E42's void control.
