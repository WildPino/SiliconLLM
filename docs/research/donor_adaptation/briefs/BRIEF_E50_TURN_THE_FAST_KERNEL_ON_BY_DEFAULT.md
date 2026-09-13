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
