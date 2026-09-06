# BRIEF E5 — decompose `R`: what is the other 81% of the attention organ?

Pre-registered 2026-09-06, **pushed before the arms exist**. Successor to
`probes/E4_ATTENTION_ACCUMULATORS.md` §7 item 1, which named this experiment, and to `INDEX.md`
§4 item 0, which E4 installed in first place as it closed the item that used to hold it.

---

## 1. What this asks

E4 split the attention organ in two with a planted control and then made one half 6.53× faster:

| | `T10` @800, `--attn avx4`, 6 threads | share of the organ |
|---|---|---|
| `X` — the `Q·K` dot loop | **2.242 ms** | 18.6% |
| `R` — *everything else in the per-head body* | **9.816 ms** | **81.4%** |
| organ | 12.058 ms | 100% |

`R` was obtained by subtraction (`serial2` = the dot loop twice, bit-identically; `X = serial2 −
serial`; `R = serial − X`), and the split was checked against a third point it was not fitted to:
`serial3` = 3× landed **−0.56%** from `3X + R`. So `R` is a measured quantity, not a leftover.
**Nothing has ever measured what is inside it.**

By construction `R` contains at least three things:

```c
float sum=0.0f;
for(int t=0;t<=pos;t++){ a[t]=expf(a[t]-mx); sum+=a[t]; }        //  S  -- the softmax pass
float rs=1.0f/sum;
for(int i=0;i<HD;i++) out[i]=0.0f;
for(int t=0;t<=pos;t++){                                          //  Y  -- the A.V loop
    const float* vt=s->vcache+((size_t)l*s->maxseq+t)*KVO+(size_t)kvh*HD;
    float w=a[t]*rs;
    for(int i=0;i<HD;i++) out[i]+=w*vt[i];
}
```

plus **`P`** — everything that is neither loop: the `#pragma omp parallel for` fork/join that runs
**once per layer, 48 times per token** at `T10`, the load imbalance of 32 heads over 6 threads, the
per-head address arithmetic, and the `out[]` init. `P` has never been named before this brief and
is not a rounding term: §3 shows that my own a-priori accounting of `S` and `Y` **does not reach
`R`**, and the gap is 3–7 ms.

**Why it matters, in the goal's own units.** At `T10` @800 after E4, `f = 12.735 ms`, the organ is
**94.7% of `f`**, and `R` alone is **77.1% of `f`**. The ceiling `1000/f` is **78.5 tok/s**; with
`R` removed entirely it would be `1000/2.919` = **342.6 tok/s**. Nothing else on the non-weight side
is within an order of magnitude of this term. E4 moved the ceiling 40.5 → 78.5 by fixing 18.6% of
the organ; `R` is the other 81.4%, and it is unexamined.

**What this is NOT.** It is not an optimisation. Every arm below makes the engine *slower* on
purpose. E5 produces a decomposition and a named next lever; it adopts nothing. As in E4, anyone
reading this for a tok/s headline is reading the wrong document — at `T10` @300 the whole organ is
4.540 ms of a 309 ms token.

---

## 2. Arms — a new flag, orthogonal to E4's

`--attnr {none,sm2,sm3,av2,av3,fork2}`, composed with `--attn avx4` on every arm, so the `Q·K` half
is held at its fastest measured form and `R` is the largest share of the organ it can be. The
dispatch is **outside the `t` loop**, per head, exactly as E4's was.

| arm | what it does | isolates |
|---|---|---|
| `none` (= plain `--attn avx4`) | unchanged | baseline: `X + S + Y + P` |
| `sm2` | the softmax pass run **twice** over the same unmodified `a[]`, second result kept | `X + 2S + Y + P` |
| `sm3` | the softmax pass run **three times**, third result kept | `X + 3S + Y + P` |
| `av2` | the `A·V` loop run **twice**, second result kept | `X + S + 2Y + P` |
| `av3` | the `A·V` loop run **three times**, third result kept | `X + S + 3Y + P` |
| `fork2` | one **extra empty `#pragma omp parallel for` over the heads**, per layer | one extra fork/join |

**Value preservation is by construction, not by tolerance.** E4's `serial3` returned
`d1+(d2-d1)+(d3-d1)` where the three accumulators are bitwise equal, so the two corrections are
exactly `+0` and the arm is *bit-identical*. The same shape is used here: the discarded passes are
folded back in as exact zeros, so **every arm must produce a bit-identical `NATS_TOTAL`** (gate G2).
`sm2`/`sm3` read `a[]` before the kept pass overwrites it, so all passes see the same input; `av2`/
`av3` accumulate the discarded passes into a `float tmp[128]` on the stack (`HD ≤ 128` in every
measured shape) — that is `2·HD·4 = 1 KB` of extra L1 traffic per head against the `pos·HD·4` the
loop already moves, i.e. **1/400 at `pos=400`**, and it is stated here so it is not discovered later.

`fork2`'s extra region writes one float per head into a per-head slot that is summed into a
`volatile` sink, so it cannot be elided, and it does nothing else.

---

## 3. Predictions, with their arithmetic, before any arm is built

`T10` @800: `L=48, NH=32, HD=128, NKV=8`, mean position 400.5, 6 threads, ~3.6 GHz.

**`S` — the softmax pass.** `L·NH·pos = 48·32·400.5 =` **615,168 scalar `expf` calls per token**,
102,528 per thread. The CRT's `expf` is not vectorised and is not inlined; 25–60 cycles is the
plausible range, giving 0.7–1.7 ms. The `sum +=` chain is a serial FP reduction of the same length,
4 cycles per element ⇒ 0.11 ms, and it is *not* hidden behind the `expf` because it depends on it.
**Predicted `S` = 0.8–2.0 ms.**

**`Y` — the `A·V` loop.** `L·NH·HD·pos = 78.7 M` FMAs per token, exactly as many as `Q·K`. Unlike
`Q·K` this is **not a reduction**: `out[i] += w*vt[i]` accumulates `HD` independent chains over
contiguous `i`, so clang may vectorise it without reassociation and without `-ffast-math` — it should
*already* be AVX2. It reads `V` at exactly the byte rate `Q·K` reads `K`: **78.7 MB unique, 315.0 MB
touched**, and E4 measured that same path at 2.242 ms (35.1 GB/s unique).
**Predicted `Y` = 2.2–3.5 ms.**

**`P` — the residual.** `S + Y` predicted = **3.0–5.5 ms** against a **measured `R` = 9.816 ms**.
So either one of the two estimates above is badly wrong, or **4–7 ms of the attention organ — a
third of `f` — is in neither loop.** It is pre-registered that I expect the second, that I cannot
today say what it is, and that `fork2` exists to price the largest single candidate.

**`fork2`.** 48 extra parallel regions per token; an OpenMP fork/join on 6 threads costs ~2–20 µs.
**Predicted 0.1–1.0 ms**, and this arm prices *one* extra region against the *one* already there, so
the baseline's own fork/join cost is read directly off it.

**A named candidate that E5 does not test.** `NH=32` heads over 6 threads with `schedule(static)`
gives two threads 6 heads and four threads 5 — the region runs at `6/5.33 = 1.125×` its mean, a
**12.5% tax on the whole organ**, `X` included. If `P` comes back large, this is the first thing to
look at, and it is cheap. It is written here so that it counts as predicted rather than as a
discovery made after the fact.

---

## 4. Gates, fixed before the arms exist

**G1 — the instrument must fire, and the model must be able to miss.**
`S` and `Y` are solved from the 1× and 2× points *by construction*, so those two points prove
nothing. The 3× arms are the test:

```
S = sm2 − none            predict sm3 = none + 2S       required |error| <= 3%
Y = av2 − none            predict av3 = none + 2Y       required |error| <= 3%
```

3% is E4's own tolerance, chosen there before its 3× point landed at −0.56%. **A component whose 2×
arm does not move the organ by more than 0.30 ms at `T10` @800 is reported `INCONCLUSIVE`, not
zero** — a doubling that costs nothing is equally consistent with a small component and with a
compiler that deleted the extra pass, and E5 will not claim to tell those apart. (E4's `serial2`
establishes that clang does not delete an identical duplicated loop *in this file, at this
optimisation level*; that is evidence, not proof, and it does not transfer for free to a loop with
a `libm` call in it.)

**G2 — parity must be BIT-IDENTICAL, not merely within σ_seed.**
Every arm is value-preserving by construction (§2), so `--bpb` on `qwen25-05b_tqh.bin` over the
pinned 24×512 slice must return a `NATS_TOTAL` **bitwise equal** to `--attn avx4 --attnr none`.
Any difference at all means the exact-zero construction is not exact and **that arm is void**.
This is a stronger gate than E4's G2 and it costs nothing.

**G3 — one sweep, and drift is named as drift.**
E4's closing law: **between-sweep dispersion on this machine is 5–10%, within-sweep 0.000–0.015
tok/s.** All six arms at all four points run in a **single sequential sweep** under one binary. The
`none` arm's organ is additionally compared to E4's 12.058 ms at `T10` @800; any gap is reported as
machine drift and **is not allowed to enter a ratio** — every number E5 concludes from is a
difference between arms measured minutes apart in the same sweep.

**G4 — no threshold inside the instrument's dispersion.**
If the §5 label depends on a share that lands within 3 percentage points of a boundary, the label is
withheld and an interleaved A/B/A/B/A/B run decides it, exactly as E4's did (there the baseline arm
itself drifted 10.55% across the pairs while the ratio held to 2.98%).

---

## 5. The decision rule, fixed now

Report `R = S + Y + P` with all three numbers, their shares of the organ and of `f`, at all four
points. The label is set at `T10` @800:

| condition | label |
|---|---|
| `S >= 0.50·R` | **SOFTMAX-DOMINATED** |
| `Y >= 0.50·R` | **AV-DOMINATED** |
| `P >= 0.50·R` | **OVERHEAD-DOMINATED** |
| none of the above | **SPLIT** |

and E5 must, for whichever term wins, state `1000/f` if that term went to zero — the ceiling that a
follow-up could buy — using the same `f` arithmetic E3 and E4 used. Terms are *not* rounded into
each other: if two are within 3 points of 0.50 the label is `SPLIT` and both are named.

**What each label costs us next**, so the conclusion is not free to be reinterpreted afterwards:
`SOFTMAX-DOMINATED` points at a vectorised `expf` and at fusing `S` into `Y` (one pass over `a[]`
instead of two — the decode-side of what flash-attention does); `AV-DOMINATED` points at the same
place `X` was, and at the KV layout, since `Y` reads `V` exactly as `X` read `K`; `OVERHEAD-DOMINATED`
points at the OpenMP region — 48 forks per token, the 32-over-6 imbalance — and would mean the
attention organ has been mis-attributed to arithmetic for two probes running.

---

## 6. What E5 does not do

- It does not adopt anything. No arm here is a candidate kernel; all six are slower on purpose.
- It does not touch the `Q·K` loop, the weight path, the KV cache format, or any exporter.
- It says nothing about quality: G2 requires bit-identity, so there is no BPB result to report
  beyond "unchanged, bitwise".
- It does not measure `f` past 800 tokens of context (E4 §7 item 3, still open) and does not extend
  to the six shapes E4's last sweep covers — `T10` and `S05`, @300 and @800, as in E4's run 2.
- It does not re-open E4's numbers. `X = 2.242` and `R = 9.816` are inputs here, and if the `none`
  arm disagrees with them by more than G3's band that is a finding about the machine, reported in
  E5 and not silently absorbed.

---

## 7. Reproduction

```
cd benchmarks/donor_adaptation/engine
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine.exe -lm
bash e5_run.sh                      # sequential; 6 arms x {T10,S05} x {300,800} + parity
python e5_analyse.py ../results/e5/arms.json
```

Never `-ffast-math` (Phase 35): it would let the compiler reassociate the very reduction whose
latency E4 measured, and every number in both probes would stop being about the code that ships.
