# E32 — does `--lutblk` survive its own protocol? The `--seqlen 512` re-run E14 owed

**Verdict: `ACTIVATION-COSTLY`.** The registered cell reads **`dBPB(A3) = +0.048990745`** at 1.5 B
under E1's protocol. E14, at the contaminated 12 k context, read **`−0.016961262`** — a *gain*.
**The sign is different, not the magnitude.** Int8 activations are not free on this donor; they
cost about **0.049 BPB**, which is **2.4× E14's own `ACTIVATION-COSTLY` line of 0.020** and
**~10× the `σ_seed ≈ 0.005`** this programme calibrates deltas against (and even that comparison
flatters it — these two arms share identical weights and identical inputs, so the difference is
*deterministic*, not a draw from a seed distribution).

**Brief**: `briefs/BRIEF_E32_THE_OWED_PROTOCOL.md`, pushed at `1fb4c18` before the runner existed.
**Runner**: `benchmarks/donor_adaptation/engine/e32_owed_protocol.py`, committed at `f5e773b`
while the run was in flight.
**Result**: `engine/results/e32_owed_protocol.json`. 3,211 s, eight arms (four × two cells).
**Quality only.** Deterministic, taken under load by the standing law; **no timing is produced
here and none may be quoted from this run.**

---

## 1. What E14 owed, and why it was owed for so long

E14 disqualified its own verdict in its own §1: with no `--seqlen`, the engine treats the whole
ids file as **one sequence** (`donor_engine.c:1556`, `SL = seqlen>0 ? seqlen : (int)n`), so E14
scored a single 12,288-token sequence while the bands it read against were drawn from 512-context
regimes. §5 has carried the re-run ever since; E28 §8 and E30 §8 both restate it.

In the meantime **the entire engine-side story was built on the arm that verdict licensed** —
E28's `61.64 G-w/s`, E30's `33.01 GB/s` and `0.909` of the ceiling, the `1.338×` shape lever. All
of those are measurements of `--lutblk`. What was never re-measured is whether the model still
says the same thing when you use it.

## 2. The gates — all three fire

| gate | what it demanded | reading | |
|---|---|---|---|
| **`G-E32A`** | `A0(0.5 B)` within `0.001` of E1's published `4.531233734` (read in its own table at `e1_05b.log:31`, not from memory) | **4.531236572**, diff **2.838e-6** | **FIRES** — the protocol *is* E1's, by a factor of 350 |
| **`G-E32B`** | `A3 − A1 = 0` exactly, at both cells — E14's known positive | **+0.000e+00** at 0.5 B **and** 1.5 B | **FIRES** — the arms are what they are labelled |
| **`G-E32C`** | `A0(1.5 B)` below the chance line, or the cell is instrument-only | **−0.594113** below | **FIRES** — the verdict cell is readable |

The chance line at this denominator is **4.069819 BPB** (51,870 scored bytes / 12,264
predictions → 4.229452 bytes per token), the programme's frozen value.

**`A0(0.5 B)` sits `+0.461418` ABOVE chance**, so the 0.5 B cell is instrument-only by E14 §2's
own demotion — expected, and exactly why the brief named 1.5 B as the verdict cell before the run.

## 3. The measurement

| cell | arm | BPB | `dBPB` vs `A0` | E14 at 12 k | shift |
|---|---|---|---|---|---|
| **0.5 B** | `A0` fp32 activations | 4.531236572 | — | 4.629292 | −0.098055 |
| | `A1` `--lut` | 4.418035433 | **−0.113201139** | −0.313878918 | +0.102622 |
| | `A2` `--lut --lut-group 32` | 4.532417346 | **+0.001180774** | −0.013006582 | −0.083868 |
| | `A3` `--lutblk` | 4.418035433 | **−0.113201139** | −0.313878918 | +0.102622 |
| **1.5 B** | `A0` fp32 activations | 3.475706372 | — | 3.446375376 | +0.029331 |
| | `A1` `--lut` | 3.524697116 | **+0.048990745** | −0.016961262 | +0.095283 |
| | `A2` `--lut --lut-group 32` | 3.487445395 | **+0.011739023** | −0.005991017 | +0.047061 |
| | **`A3` `--lutblk`** | 3.524697116 | **+0.048990745** | −0.016961262 | +0.095283 |

`A0(1.5 B)` moving `+0.029331` when the context drops from 12 k to 512 is the sane direction — a
shorter context is a harder prediction problem — and is the strongest sign that the protocol
change did what it was supposed to do.

### 3.1 The finding that is not in the verdict cell: `A2` is four times cheaper

`--lut-group 32` costs **`+0.011739`** where `--lutblk` costs **`+0.048991`**. **Finer activation
scaling recovers 76% of the loss.** This is the first number in the programme that separates
*int8 activations* from *one scale per vector*: the cost is not "int8", it is **the granularity of
the scale**, exactly the way E31 found the cost of the carve was not "sparsity" but the
granularity of the read.

**What stops this from being a recommendation** is that `A2` has **no speed number at all** —
E14 §0 recorded that, E32 §7 re-registers it, and it is now the most valuable owed measurement in
the engine branch. E28 timed `--lut` (S15: **22.54** tok/s, *slower* than packed's 29.30) and
`--lutblk` (**36.80**), never `--lut --lut-group 32`. **The fast kernel is the costly one and the
cheap kernel is unmeasured**, which is the whole shape of the problem in one line.

## 4. Predictions — scored as registered. 3 HIT / 2 MISS, and the misses are on the cell that matters

| # | registered | outcome | |
|---|---|---|---|
| 1 | `G-E32A` fires | 2.838e-6 on a 1e-3 bar | **HIT** |
| 2 | `G-E32B` fires, both cells | `+0.000e+00` both | **HIT** |
| 3 | **verdict does NOT flip to `ACTIVATION-COSTLY`; `abs(dBPB(A3))` stays below `0.020`** | **+0.048991** | **MISS** |
| 4 | **`dBPB(A3)` moves TOWARD zero, into `[−0.015, +0.005]`** | moved **away**, to the opposite side, by 3× the interval's width | **MISS** |
| 5 | rank partner stays below 70% | 45.6% (inherited, see §5) | **HIT** |
| 6 | registered as independent: if costly, the operative kernel is packed at 0.639 → ~4.8 tok/s at T10, and E30's architectural conclusion is identical either way | now operative | **holds** |

### 4.1 Why prediction 3 was wrong, stated plainly

I wrote: *"E14's arm-vs-arm comparisons were protocol-invariant by construction, and only the band
reading was contaminated."* **That is false, and it was the load-bearing assumption of the entire
prediction.** An arm-vs-arm delta is protocol-invariant only if the protocol does not change the
thing being quantised. It does: **the activation distribution is a function of context length.**
The quantiser sees different vectors at 512 than at 12 k, so `dBPB` is not a property of the
kernel alone, and there was never a construction that made it one.

Prediction 4's *mechanism* survives its own number. I argued that at 12 k the baseline attends
across 23 unrelated document boundaries and produces over-confident logits that int8 noise
accidentally improves. The reading is consistent with that: remove the pathology and the
accidental benefit vanishes — **but it vanishes and keeps going**, from `−0.017` to `+0.049`, a
swing of `0.066`. So at 12 k the gain was not merely a gain: it was a real `~0.049` cost hidden
under a larger pathological benefit. **My mechanism explained the direction and under-called the
size by a factor of four.** It stays labelled a hypothesis; nothing here measures it.

## 5. The RANK partner — mandatory under E14 §3, and it is verified at source, not assumed

E14 §3's law: **every SCORE metric needs a RANK partner.** E14 read greedy top-1 agreement
`A3` vs `A0` at 1.5 B of **45.6% (73/160)** while BPB was calling the same arm *cheap*.

That number is **not re-run here, and the reason is structural rather than convenient**:
`e14_gn3_greedy.py` invokes the engine with `--generate` and never passes `--seqlen`, and
`donor_engine.c`'s generate branch **returns at `:1513`, before `SL` is computed at `:1556`**.
`--seqlen` is therefore *unreachable* from the generation path — the greedy measurement was never
touched by the defect that voided the BPB one, and it stands at this protocol as measured.

**So both halves now agree, which is new.** At 12 k the score said cheap and the rank said broken,
and the honest reading was "a cheap BPB with a broken trajectory". At 512 **the score says costly
and the rank says broken.** The divergence that made E14 hard to read was itself an artifact of
the contaminated protocol.

## 6. What this does to the record — corrections go at the TOP of E28 and E30, as §6 of the brief required

E28's headline (`61.64 G-w/s`, `CONTAINER-COSTS`) and E30's engine row (`33.01 GB/s`, `0.909` of
the measured ceiling) are **both read off `--lutblk`**. Those measurements are not withdrawn — the
kernel exists, the bytes moved, the timings were taken on an idle box with interleaved reps.
**What is withdrawn is the quality licence to treat that kernel as the engine's operative path.**

| quantity | as published (`--lutblk`) | operative after E32 (packed) | source |
|---|---|---|---|
| T10 engine rate | 6.213 tok/s | **4.363 tok/s** | E30, same session, same reps |
| fraction of the 36.30 GB/s ceiling | 0.909 | **0.639** | E30 |
| charged numerator | 61.64 G-w/s | **46.16 G-w/s** | E28, same session, both arms |
| gap to 50 tok/s at T10 | 8.05× | **11.46×** | E30 rates ÷ 50 |

Each row pairs arms **measured in one session** — the two rates from E30's reps, the two charged
numerators from E28's — because mixing sessions is how this programme has been burned before
(E26's absolutes against E28's, addendum to E33 §2). The charged numerators use T10's
10.6032005 G active weights per token and are in the CHARGED convention; the GB/s figures are in
the MOVED convention. The two never meet inside a fraction here.

**E30's architectural conclusion is untouched, exactly as prediction 6 registered before the
run**: a *perfect* kernel at T10 still reads 6.83 tok/s on this box, 50 tok/s still needs 7.32× the
machine's entire read bandwidth, and the resident-core arithmetic (≈32 M weights free, ≤1.45 G
streamed) is a property of the machine, not of a kernel. What changes is that the engine is
**1.57× from the wall rather than 1.10×** — there is *more* kernel headroom than E30 reported, and
it is headroom that must be bought without `--lutblk`'s quality cost, or bought with `A2`'s
granularity if `A2` turns out to be fast.

## 7. What E32 cannot claim

- **Nothing at the goal's shape.** Registered in the brief §7 before the run: T10 is a noise
  export, activation outliers are a property of *trained* weights, and a synthetic T10 reading
  would systematically **understate** the int8 cost. **The int8 cost at a 10 B shape is unmeasured
  and unmeasurable with what is on this disk.** It is not reported as small; it is not reported.
- **Nothing about speed**, including `A2`'s. No timing was taken.
- **Nothing about the head** (`--lut-no-head` is a separate lever).
- **Nothing about other donors.** Two cells of one family, both Qwen2.5, both TQH.

## 8. Owed after E32

1. **Time `A2` (`--lut --lut-group 32`).** Now the highest-value engine measurement available: it
   is the only known arm that is cheap in quality, and nobody has ever timed it. If it lands near
   `--lutblk`'s 36.80 at S15, the quality licence comes back at a quarter of the cost.
2. **The 12 k-vs-512 mechanism** (§4.1) is a hypothesis with a swing of 0.066 attached to it and
   no measurement. A context ladder (512 / 1 k / 2 k / 4 k) on `A0` and `A3` would settle it, is
   cheap, and would say whether int8's cost grows or shrinks with the context the goal will
   actually serve.
3. **`--lutblk` parity at T10** (E30 §8 item 1) — still owed, still unmeasurable in quality terms
   for the reason in §7.
4. **E28 §8's per-matrix `--lut` guard** at `donor_engine.c:1437` is now *more* interesting, not
   less: if the cheap granularity is fast, the guard is what blocks it on rank and carve arms.
