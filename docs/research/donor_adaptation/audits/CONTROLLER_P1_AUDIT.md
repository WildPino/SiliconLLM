# CONTROLLER AUDIT — P1 Nibble Packing (Stage 1 + Stage 2), and the stride finding

**Controller** · 2026-08-30 · target `docs/research/donor_adaptation/probes/P1_NIBBLE_PACKING.md`,
brief `briefs/BRIEF_P1_NIBBLE_PACKING.md` incl. Amendments 1 and 2, code
`benchmarks/donor_adaptation/nibble_pack{.h,_test.c,_bench.c}`, `benchmarks/phase60/engine.c`,
and every artefact under `benchmarks/donor_adaptation/p1/results/`.

**Machine condition, verified by me, not from any banner.** At the start of this audit `llama-server.exe`
was resident at 20.28 GB (started 19:34:40) and no timing was possible. Mid-audit I re-checked and found the
machine clear — no `llama-server`, no python, processor queue 0, 62.2 GB free, largest process `chrome` at
858 MB — and **took one fresh timed run** (§1.4). I re-checked during and after: the window closed partway
through, with two `python` processes at 10.6 GB and 1.8 GB and queue 16 appearing. **The completed run in
§1.4 finished inside the clean window** (its own dispersion confirms it: between-invocation CV 1.1–8.6%).
The second run I wanted, §1.6, was **not taken**; it is built, it compiles, and it is left as a FLAG with its
exact command. Every other finding is from artefacts, source, and internal consistency.

---

## Verdict table

| # | finding | verdict |
|---|---|---|
| **A1** | The stride effect is **real**, **reproduces independently on a clean machine to three significant figures**, and is **attributable to the stride** — not to allocation size, alignment, or TLB | **PASS** (strengthened) |
| **A2** | The **stated mechanism** (Zen2 L1d, 64 sets, `Mpad/64 mod 64`) is at the **wrong cache level** and, as written, names a cost that cannot be paid | **BLOCK** |
| **A3** | "Predicted **4 of 4**" — the same arithmetic scores **7 of 8** on the report's own table; the miss is undisclosed; the independent n is **2 strides**, not 4 shapes | **BLOCK** |
| **A4** | Effect size: the mechanism predicts the **sign only**. At one fixed stride the recovery ranges **1.52×–3.28×**. The `1.43–1.68×` band describes four shapes, not the mechanism | **BLOCK** |
| **A5** | `k/v`'s non-recovery is **correctly predicted**, but not by the stated L1 arithmetic | **FLAG** |
| **B1** | D5 reconciliation: the flat corrected sweep and the aliasing explanation are **compatible and mutually reinforcing** — D5's mechanically-unexplained `f = 2.0` **is** the effect P1 found | **PASS** |
| **B2** | "the fall **past L3**" — the L3 threshold is **not** the operative variable. Measured: full penalty at a **16 KB** reuse footprint. The finding is larger than claimed; the sentence is wrong | **FLAG** |
| **C1** | **"Moved GB/s" is charged bytes.** `moved_B == charged_B` in **all 200 cells**; the column is `matvecs/s × 0.5` exactly. On `+0` rows it undercounts real traffic ~2× and inverts the physical reading | **BLOCK** |
| **C2** | A1.3's discriminator is **degenerate** — every branch of the classifier is a function of `matvecs/s` alone. "The moved GB/s column is what saves the experiment" is false | **BLOCK** |
| **C3** | P1 confounds **three** things, not the two disclosed: bytes, load count, **and reuse working set** | **FLAG** |
| **C4** | The `+0` `4.127×` "artefact" and the `+64` `3.286×` "finding" are the **same signature**, treated inconsistently. The `4.127×` flag is right for the wrong reason | **FLAG** |
| **C5** | `k/v` t6 exclusion: **immaterial** to every conclusion, **post-hoc**, keyed to a **quantity that is not the estimator's uncertainty**, and **not applied consistently within the document** | **FLAG** |
| **C6** | What the dispersion figures are the dispersion **of** — the "trust" column uses per-rep jitter, not the precision of the cell mean | **FLAG** |
| **D1** | Machine record reproduces **exactly**. Queue spikes **cannot** be tied to cells (no timestamps in the raw CSV) — but I show the conclusions are **invariant** between clean and contended invocations | **PASS** |
| **E1** | Stage 1 counts, hashes, byte ratios, diffstat: **all reproduce exactly** | **PASS** |
| **E2** | **Is C1 capable of failing?** Demonstrated for the `full` path only — **46.6%** of the 523,395 comparisons. `tileskip` has a **live vacuity hole** | **FLAG** |
| **E3** | `--g3c --pack nibble` hard-errors and **cannot be reached accidentally** — verified by reachability, not by reading the guard | **PASS** |
| **E4** | `--kselftest` and `--exprate` **silently ignore** `--pack` | **FLAG** |
| **F1** | Brief correction 1: the byte factor is `2T/(T+1)` | **PASS** |
| **F2** | Brief correction 2: no novelty claim, no "pshufb optimum" anywhere in the report | **PASS** |
| **G1** | **The D0 signature is present.** Every table exact; **seven** prose over-statements, all leaning the same way | **BLOCK on the prose** |

---

# 1. THE STRIDE FINDING

## 1.1 The cache arithmetic, from first principles — the set count is right

Zen 2 (Matisse) L1d: **32 KB, 8-way, 64 B lines**. Sets = `32768 / (8 × 64)` = **64**. With 64 B lines the
line offset occupies bits `[5:0]`, so the set index is bits `[11:6]` = **`(addr >> 6) & 63`**. L1d is VIPT
and 32 KB / 8 ways = 4 KB per way = the page size, so the virtual and physical index agree and the formula
is usable on virtual addresses. **The report's `64 sets` and `(addr >> 6) & 63` are both correct.**

The loop nest is the load-bearing detail and I verified it in source rather than from the prose
(`nibble_pack_bench.c:64-71`; identical in `engine.c:128` and `nibble_pack.h:matvec_lut_full_n`):

```
OMP_PFOR for(int base=0; base<M; base+=32)                    // OUTER, parallel
    for(int t=0; t<T; t++)                                     // INNER
        ... _mm256_loadu_si256(codes + t*Mpad + base)          // 32 B load
```

Consecutive inner loads are **one plane stride apart**, so the set index advances by `(stride/64) mod 64` per
load. **`Mpad/64 mod 64` is the right invariant for the set advance.** Confirmed from first principles.

## 1.2 I computed the invariant myself, for every shape — and there are eight, not four

The report lists four shapes. Its own S2.4 table contains **eight cells** — four shapes × two arms — and the
set arithmetic is *identical* in both arms, because **the stride does not depend on the arm**. Only the plane
count does. So the mechanism makes eight predictions and the report scores four.

| shape | `Mpad` | `Mpad/64` | **adv mod 64** | L1 sets `64/gcd(adv,64)` | arm | planes | **+64 / +0 measured (t1 / t6)** |
|---|---|---|---|---|---|---|---|
| `q/o` | 8192 | 128 | **0** | 1 | byte | 4096 | 1.562 / 1.675 — **recovers** ✔ |
| `q/o` | 8192 | 128 | **0** | 1 | **nibble** | 2048 | 2.517 / 1.466 — **recovers** ✔ |
| `gate/up` | 28672 | 448 | **0** | 1 | byte | 4096 | 1.429 / 1.681 — **recovers** ✔ |
| `gate/up` | 28672 | 448 | **0** | 1 | **nibble** | 2048 | **0.951 / 1.034 — DOES NOT** ✘ |
| `down` | 8192 | 128 | **0** | 1 | byte | 14336 | 1.506 / 1.634 — **recovers** ✔ |
| `down` | 8192 | 128 | **0** | 1 | **nibble** | 7168 | 1.671 / 2.016 — **recovers** ✔ |
| `k/v` | 1024 | 16 | **16** | 4 | byte | 4096 | 0.930 / 0.951 — does not ✔ |
| `k/v` | 1024 | 16 | **16** | 4 | **nibble** | 2048 | 0.958 / 1.022 — does not ✔ |

The four values the report lists (`0`, `0`, `0`, `16`) are **arithmetically correct**. Two things follow that
the report does not say.

**(a) The three "recovering" byte shapes carry only TWO distinct strides.** `q/o` and `down` are both
`Mpad = 8192` — the same stride, the same set advance, the same value of every variable the mechanism names.
The sweep tests `adv = 0` (two strides: 8192 and 28672) against `adv = 16` (one stride: 1024). **In the
variable that matters this is n = 2 versus n = 1, not "4 of 4".** — **A3, BLOCK.**

**(b) `gate/up` nibble has `adv = 0` and does not recover.** The stated mechanism makes an identical
prediction for it and gets it wrong. Scored against the report's own S2.4 table, the mechanism is **7 of 8**,
and the one miss sits in that same table, unremarked. **"Predicted 4 of 4" is a score computed on a
hand-picked half of the available evidence, where the excluded half contains the counter-example.** — **A3,
BLOCK.**

## 1.3 The stated mechanism names a cost that cannot be paid — A2, BLOCK

> *"Every plane load in a `q/o`, `gate/up` or `down` matvec lands in the SAME L1 set. Eight ways, thousands
> of planes: **the walk evicts itself on every access.**"*

**Inside the inner `t` loop there is no reuse to lose.** Each address `t*Mpad + base` is touched exactly once
per matvec. A pure streaming walk that self-evicts costs **nothing** — nothing in the evicted lines will be
asked for again during that walk. Taken literally, the stated mechanism describes a penalty that does not
exist.

The reuse that *is* destroyed lives one loop level out:

- The load is **32 B**; the line is **64 B**. Every stride in the sweep is a multiple of 64.
- Therefore iterations `base` and `base + 32` read **the two halves of the same 64 B line**.
- Between those two touches lies one complete inner sweep: **`planes` distinct lines**, i.e.
  `W = planes × 64` bytes of reuse working set.
- With `adv = 0`, the whole inner sweep collides into one L1 set, `64/gcd(·,64)` L2 sets, `16384/gcd(·,16384)`
  L3 sets. The capacity available to *hold* `W` until the `base+32` sweep collapses to a fraction of the cache.
- If `W` does not survive, **every 64 B line is fetched twice per matvec.**

That is the mechanism: **loss of half-line spatial reuse under set conflict.** It is genuinely caused by the
stride, and — unlike the self-eviction story — it predicts a *bounded* factor: at most **2× the traffic**,
hence at most ~2× recovery, which is the right order of magnitude for 1.43–1.68×. The report's version
predicts no bound at all.

**This also explains `k/v`, where the report's L1 arithmetic cannot.** The report rests on *4 L1 sets rather
than 1*. Four sets × 8 ways = **32 lines**, against a reuse working set of **4096 lines**. **32 against 4096
is not a margin**; at L1 the `k/v` walk is as hopeless as the others, so the stated mechanism does **not**
predict its non-recovery. At **L3** it does: `gcd(16, 16384) = 16` → 1024 reachable sets × 16 ways =
**1 MB against W = 256 KB**, a 4× margin, while `q/o` gets `gcd(128, 16384) = 128` → 128 sets × 16 =
**128 KB against 256 KB** and loses it. **The right prediction, from the wrong level.** — **A5, FLAG.**

## 1.4 THE FRESH RUN — attribution settled, effect size falsified

Taken on the clean machine described at the head of this document. **Byte kernel only, single thread,
M = 8192, 512 MiB rotating replica pool, per-rep MEAN (min-of-reps never computed), 3 invocations, plane
count swept at fixed stride.** Three arms:

- **A** — `stride = 8192`, the shipped geometry.
- **B** — `stride = 8256`, the report's `+64`.
- **C** — `stride = 8192` with the **base pointer advanced by 64 B**: same stride, different alignment, same
  extra allocation. **This is the attribution control the report does not have.**

| planes | `W = planes×64` | A `stride 8192` | B `stride 8256` | C `off +64` | **B/A** | **C/A** | cv(A) |
|---|---|---|---|---|---|---|---|
| 256 | 16 KB | 1772.1 | 3539.5 | 1532.4 | **1.997** | 0.865 | 7.6% |
| 512 | 32 KB | 873.3 | 2058.5 | 797.3 | **2.357** | 0.913 | 1.1% |
| 1024 | 64 KB | 410.2 | 1005.1 | 385.6 | **2.451** | 0.940 | 5.2% |
| 2048 | 128 KB | 143.2 | 469.6 | 156.5 | **3.279** | 1.093 | 5.6% |
| **4096** | **256 KB** | **56.7** | **88.6** | 56.1 | **1.562** | 0.989 | 8.6% |
| 8192 | 512 KB | 27.2 | 41.4 | 26.9 | **1.525** | 0.989 | 3.8% |

(units: matvecs/s, mean of 3 invocation means.)

Three results.

**(i) The attribution holds — and this strengthens the report.** Arm **C** changes the total allocation and
the base alignment by exactly what arm B changes, and leaves the stride alone. It recovers **nothing**
(0.87–1.09×). Padding a stride changes the set index, the allocation size, and the buffer alignment
together; **only the set index matters.** Page and TLB behaviour is separately ruled out by arithmetic: at
stride 8192 the inner sweep touches 4096 distinct 4 KB pages, and at 8256 it touches the same 4096 — the pad
does not move TLB pressure at all. **The mandate's item 3 is answered in the Builder's favour, by a control
the Builder did not run.**

**(ii) Independent replication of the headline.** At `planes = 4096` — exactly the `q/o` byte shape — I
measure **1.562×**. The report's S2.4 reports **1.562×**. Three significant figures, different harness,
different binary, different day, different machine condition. **The number is not an artefact of the
Builder's harness or of the contended window it was taken in.**

**(iii) The effect size is NOT a signature of the mechanism — A4, BLOCK.** At **one stride**, with the set
arithmetic **identical in every row of the table**, the recovery ranges **1.52× to 3.28×** and is
**non-monotone**, peaking at `planes = 2048`. The report presents `1.43–1.68×` as what the mechanism buys.
It is one point on a curve the report never sampled, and the curve is twice as wide as the band.

**So: does 8-way associativity predict 1.43–1.68×, or only its sign?** **Only the sign, plus a ceiling —
and even the ceiling is violated.** The half-line derivation of §1.3 bounds the recovery at ~2× (you cannot
save more than a duplicate fetch), and 1.43–1.68× sits credibly under it. But §1.4 measures **3.28×**, which
is *above* my own corrected mechanism's ceiling. At intermediate working sets the reuse is not merely being
lost to DRAM; it is falling several levels of the hierarchy at once and latency, not traffic, dominates.
**No mechanism currently on the table predicts the observed magnitudes.** The finding is a robust
*phenomenon* with an *unfinished* explanation, and the report presents the explanation as settled.

**Reproduction.** `benchmarks/donor_adaptation/p1/audit/ctrl_stride_probe.c` →
`benchmarks/donor_adaptation/p1/audit/ctrl_stride_probe.csv` (54 cells, 3 invocations).
Build: `clang -O3 -mavx2 -mfma -march=znver2 -Wall .../ctrl_stride_probe.c -o bin/ctrl_stride_probe.exe -lm`.
**Note for anyone re-running it:** my first build of this probe was **silently optimised away** — every cell
reported ~0.025 µs, the bare `QueryPerformanceCounter` overhead — because `y` was written and never read.
The committed version consumes `y` into a `volatile` sink after each rep. This failure mode is invisible in
the output unless you sanity-check the absolute time against the byte count, and it is worth knowing that a
matvec harness can be deleted wholesale by the compiler while still printing a plausible-looking CSV.

## 1.5 The one run that would close the largest remaining question — FLAG, needs a clear machine

The report's stated invariant makes a sharp, cheap, untested prediction: recovery should scale with
`64/gcd(adv, 64)`, and `adv = 16` — `k/v`'s value, 4 reachable L1 sets — should already be most of the way
recovered. Sweep the pad at fixed shape and read the curve:

```sh
clang -O3 -mavx2 -mfma -march=znver2 -Wall \
      benchmarks/donor_adaptation/p1/audit/ctrl_stride_sweep.c -o bin/ctrl_stride_sweep.exe -lm
for i in 1 2 3; do ./bin/ctrl_stride_sweep.exe $i; done
# pad in {0,64,128,256,512,1024,2048,4096} at M=8192, planes=4096, 1 thread
```

The file is written and **verified to compile**; it was **not run** — the machine went to 10.6 GB of python
and queue 16 before I could take it. `pad = 1024` gives `adv = 16`: **`k/v`'s exact set advance at `q/o`'s
working set.** If that row recovers, the report's L1 story is right and §1.3 is wrong. If it does not,
`adv = 16` is not why `k/v` is flat and §1.3 stands. **This is the discriminating shape the mandate's item 4
asks about, and it was never run — by the Builder or by me.**

---

# 2. THE D5 RECONCILIATION — compatible, and the report undersells it — B1, PASS

`CONTROLLER_D5_126_AUDIT.md` §A2 found the corrected `Mpad` sweep **flat at 21.4–22.7 GB/s of moved bytes
across a 28× change in `Mpad`**. P1 says the disputed fall was aliasing. The coordinator is right that these
are not obviously compatible. **They reconcile completely, and the reconciliation is the strongest
corroboration of P1's finding in the file — which the report does not notice.**

**The d5m sweep holds `T` constant at 4096.** From the D5 audit's own table, `EB / Mpad` = 4096 in every row
(4 MiB/1024, 8 MiB/2048, 16 MiB/4096, 32 MiB/8192, 112 MiB/28672). So **`W = 4096 × 64 = 256 KB` in every
row** and the *only* thing that varies down the sweep is the stride. Applying §1.3:

| `Mpad` | `Mpad/64` | adv | L3 sets `16384/gcd` | L3 capacity for `W` | half-line reuse held? | **D5's measured `f`** |
|---|---|---|---|---|---|---|
| 1024 | 16 | 16 | 1024 | 1024 KB | yes — 4× margin | **1.0** |
| 2048 | 32 | 32 | 512 | 512 KB | yes — 2× margin | **1.0** |
| 4096 | 64 | 0 | 256 | 256 KB | **exactly at capacity** | **1.4** |
| 8192 | 128 | 0 | 128 | 128 KB | **no** — 2× over | **2.0** |
| 28672 | 448 | 0 | 256 | 256 KB | **at capacity** | **2.0** |

`f` is the charged→moved correction the D5 audit applied in order to flatten the sweep. **In that document it
was derived from block residency and left mechanically unexplained. It is, row for row, the half-line
double-fetch factor.** `f = 2.0` means each 64 B line is fetched twice — which is precisely what losing the
`base` / `base+32` reuse does, and precisely what the `+64` pad repairs.

**The reconciliation, stated plainly:**

- The corrected sweep is **flat in moved bytes** because DRAM is saturated near ~22 GB/s in every row.
- The uncorrected sweep **fell in charged bytes** (22.7 → 10.7) because on the aliased rows **half of that
  saturated bandwidth was being spent re-fetching line halves**.
- P1's `+64` pad removes the duplicate fetch, so the same useful work needs half the traffic — hence the
  wall-clock recovery.

**A flat moved-byte sweep, a falling charged-byte sweep, and a ~1.5× recovery from removing the waste are one
consistent picture, not three.** Two audits, working from opposite directions and months apart, arrived at
the same factor of 2. **There is no conflict to report.**

**But the report's own sentence about it is wrong — B2, FLAG.** It says the fall *"past L3"* is
*"cache-set aliasing, **not capacity**"*. That is a false dichotomy: aliasing **is** a capacity constraint,
imposed on a restricted subset of ways rather than on the whole cache. Worse, **"past L3" is not the
operative variable at all** — §1.4 measures the **full 2.0× penalty at `planes = 256`, a 16 KB reuse
footprint inside a 2 MB block**, orders of magnitude within L3. **There was no L3 threshold to explain.**
Both D5 rounds and this report have been reading a *stride* effect as a *size* effect, because in the d5m
sweep stride and size were varied together — which is exactly the confound Amendment 1 §A1.5 named and
exactly what the `+64` arm was added to separate. **The finding is larger than the report claims, and the
sentence describing it is wrong in a way that keeps it artificially small.**

---

# 3. STAGE 2 — the measurement

## 3.1 "Moved GB/s" is charged bytes — C1, BLOCK

I parsed all 200 cells of `stage2_raw.csv`. **`moved_B == charged_B` in every single one.** This is not a
coincidence of shape, it is an identity: the counter increments 32 B per load, and
`loads × 32 = T × (M/32) × 32 = M·K/2`, which is the *charged* definition. The report discloses that the two
columns are numerically equal (S2.1) and reads it as a benign consequence of `Mpad == M`. It is worse:

- **The "two independent readings" of Stage 1 have no Stage 2 counterpart.** In Stage 2 there is one
  instrument, printed twice.
- **The moved GB/s ratio is `matvecs/s ratio × 0.5`, exactly, in all 16 S2.5 cells.** I checked each:
  1.912→0.956, 1.734→0.867, 1.688→0.844, 1.581→0.791, and so on. **The column carries zero independent
  information.**
- **On the `+0` rows the figure understates real traffic by ~2×**, because the aliasing this same document
  discovers causes every line to be fetched twice. So S2.4 reads `q/o` t6 byte as *7.52 moved GB/s at `+0`,
  12.60 at `+64`* — while the memory system actually moved ~**15.0** GB/s at `+0` against 12.60 at `+64`.
  **The machine moved MORE bytes per second before the fix, not fewer.** S2.4's GB/s columns invert the
  physical story on the exact rows the headline finding rests on.

This is a milder form of the units error for which the D5 audit BLOCKed §12.6 — **a charged numerator wearing
a moved label** — committed in a document whose §S2.1 exists specifically to restate A1.4's convention and
avoid it.

**Reproduction:** parse `stage2_raw.csv`; compare fields `moved_B` and `charged_B` across all 200 rows;
verify `moved_GBs_ratio == 0.5 × matvecs_per_s_ratio` cell by cell.

## 3.2 A1.3's discriminator is degenerate — C2, BLOCK

Because moved bytes are fixed per arm by construction, `moved GB/s = moved_B × matvecs/s`, so the two columns
are perfectly anti-correlated by algebra. A1.3's "two mutually exclusive and jointly exhaustive outcomes" are
**one number read twice**. `aggregate_stage2.py` makes this explicit without noticing: its classifier reduces
to `rgb > 1.15 ⟺ rmv > 2.30`; `bandwidth-limited ⟺ 1.80 ≤ rmv ≤ 2.30`; `compute/port-limited ⟺ rmv ≤ 1.15`.
**Every branch is a function of `matvecs/s` alone.**

So S2.5's *"The moved GB/s column is what saves the experiment, and it is the reason A1.4's convention was
worth insisting on"* is **false**. Nothing was saved; a threshold was applied to the only quantity measured.
Amendment 1's central methodological claim — *"this formulation has no denominator to get wrong"* — is true
and beside the point: **the formulation has no second measurement.**

**What would fix it:** an independent traffic meter — hardware performance counters for L3-miss / DRAM
transactions, or at minimum a second software counter that counts distinct **64 B lines** rather than 32 B
loads, which would have caught the `+0` double-fetch directly and turned §1.3 into a measurement rather than
a derivation. Without one, "moved bytes" in this programme means "charged bytes" and should be labelled so.

## 3.3 Three confounds, two disclosed — C3, FLAG

The Builder's disclosure deserves to be on the record as good: *"P1 halves the bytes AND halves the load
count, simultaneously and inseparably."* Candid and correct. **There is a third.**

A contiguous 2 bits/weight layout also **halves the reuse working set** `W = planes × 64` — and §1.4 shows
that `W` moves the recovery by up to 2× at fixed stride. So the nibble arm is not the byte arm with fewer
bytes; it is the byte arm **at half the cache footprint**, sitting at a different point on the §1.4 curve.
This is the mechanism behind every superlinear cell:

| cell | `matvecs/s` ratio | ceiling if bytes bind | ceiling if loads bind |
|---|---|---|---|
| `gate/up` t1 `+0` | **4.127×** | 2× | 2× |
| `q/o` t1 `+64` | **3.286×** | 2× | 2× |
| `gate/up` t6 `+0` | **2.818×** | 2× | 2× |
| `gate/up` t1 `+64` | **2.747×** | 2× | 2× |

**No arrangement of the existing data separates bytes from loads.** Both are halved by the same edit, in
every cell, in both stride arms, at both thread counts. Separation requires an arm that moves one without the
other: a byte-packed layout read with **64 B loads** (same bytes, half the loads), or a nibble layout read
with **two 32 B loads** per plane (same loads, half the bytes). Neither exists in the sweep and neither can be
synthesised from it.

**Therefore the "jointly limited, leaning bandwidth" verdict is not supported by this experiment.**
"1.58–1.91× on matvecs/s, 0.79–0.96× on GB/s" is equally compatible with pure load-issue limitation, pure
bandwidth limitation, and pure working-set relief — and §3.1 shows the second column cannot arbitrate between
them. It is one reading among three that fit identically well.

## 3.4 The `>1.15` exclusion and the `4.127×` flag are the same signature — C4, FLAG

The report's reasoning for taking four cells off-table — *"you cannot beat a byte ceiling by moving fewer
bytes if bytes were the binding constraint"* — is **correct**, and **the exclusion hides nothing**: I
confirmed that exactly four cells exceed 1.15 (2.064, 1.409, 1.373, 1.643 → `matvecs/s` 4.127, 2.818, 2.747,
3.286) and that they are the four with the *largest* `matvecs/s` ratios. Nothing favourable is being removed;
if anything the exclusion is self-denying.

**But the treatment is inconsistent.** The report flags `+0 gate/up`'s **4.127×** as *"physically impossible
… an aliasing artefact"* on the ground that a 2× byte reduction cannot buy 4×, then in the next section reads
`+64 q/o`'s **3.286×** as a *physical finding* ("the byte arm was access-limited, not byte-limited").
**Access-limitation caps the gain at 2× exactly as hard as bandwidth-limitation does** — halving the load
count cannot buy more than 2× if loads are the binding constraint. So 3.286× is precisely as superlinear as
4.127×, under precisely the same argument, and the report explains one away while building on the other.

**Verifying the `4.127×` flag as instructed: the flag is right for the wrong reason.** It is **not** a broken
measurement in the other direction. I reproduce 4.127× from the raw CSV; its between-invocation CV is 9.2%
(byte) / 12.1% (nibble), among the tightest in the entire sweep; the paired per-invocation range is
3.711–4.468. It is a **real, repeatable measurement of a third effect** (§3.3), not an artefact. Labelling it
an artefact discards the single cell that most clearly demonstrates the undisclosed confound.

## 3.5 The `k/v` t6 exclusion — C5, FLAG

Three questions, and they come apart cleanly.

**Is it material?** **No.** Retaining the `k/v` t6 cells changes nothing: their nib/byte ratio is 1.688×,
which sits *inside* the headline band `1.58–1.91×` the report quotes, and the `k/v` byte non-recovery
(0.951×) is already reported in S2.4. **The conclusions with the cells retained are the conclusions as
printed.** So the exclusion buys the report nothing and cannot have been an act of selection.

**Was it symmetric — would a cell favouring the hypothesis with equally bad CV have been dropped?**
**Untestable, and that is the answer.** No cell outside `k/v` t6 exceeds **42.2%** within-invocation CV, so a
hypothesis-favouring cell with comparable dispersion **does not exist in this sweep**. The criterion was
never put to a test that could have exposed asymmetry. Its symmetry is *unestablished*, not violated.

**Was the stated criterion the criterion applied? No.** S2.6 excludes *"`k/v` t6 (any stride)"* — a
shape × threads bucket. But the `k/v +64` t6 **nibble** cell has within-invocation CV **66.1%**, comfortably
below the stated 100% threshold, and is dropped along with the rest. **The rule executed is "drop `k/v` at 6
threads"; the rule written is a CV threshold.** Post-hoc, and not the rule described.

**And the exclusion is not applied consistently inside the document.** S2.6 states the cells are *"excluded
from every conclusion drawn above."* S2.4 sits above it and uses `k/v` t6 byte **0.951×** as one of the four
shapes in *"the measurement matches 4 for 4"* — the single load-bearing sentence of the whole report.
**The cell is excluded where it would add noise and retained where it supports the mechanism.** That is the
asymmetry, and it runs in the direction of the finding.

**One correction that runs the other way, in the Builder's favour.** Splitting the invocations (§3.7),
`k/v` byte `+64/+0` at t6 is **1.041× in the two clean invocations** and **0.875× in the three contended
ones**. The reported 0.951× is a contention average; the clean reading is **~1.0, i.e. no change**. So the
*qualitative* claim — `k/v` does not recover — is **better supported** than the report shows, and the
apparent small regression is an artefact of the very spikes the Builder disclosed.

## 3.6 What the dispersion figures are the dispersion OF — C6, FLAG

The coordinator asks this specifically, on the back of D0's "wrong quantity" defect. P1's Stage 2 is a timing
measurement, so D0's unseeded-RNG problem does not transfer — but **the same species of error is present in
milder form.**

S2.6's "trust" column is keyed to the **within-invocation CV**. That statistic is the dispersion of
**individual per-rep timings** — dominated by OS scheduling and, at 6 threads, by OpenMP fork/join jitter
against a `k/v` matvec of only ~100–240 µs. **It is not the uncertainty of the cell's reported value.** The
cell reports a *mean over `reps`*, and with `reps` between 170 and 9536 the standard error of that mean is
`CV/√reps` — for `k/v` t6, `136.7% / √4768 ≈ 2%`. **By the estimator's own precision those cells are among
the best determined in the sweep, not the worst.**

The statistic that *does* justify distrusting them is the **between-invocation CV** — 17.8–33.0% for `k/v`
t6, against 8.7–19.9% for the large shapes — because that captures the run-to-run drift the per-rep
distribution cannot see. **The exclusion is defensible; the reason given for it is the wrong quantity.**

**A second, anti-flattering instance.** S2.5's *"range over 5 inv"* is not a range over 5 values:
`aggregate_stage2.py` computes `min(nibble)/max(byte)` to `max(nibble)/min(byte)`, an **unpaired worst-case
envelope**. Paired by invocation, `q/o` t6 `+64` is **1.650–2.141**, not the reported 1.209–2.925.
**The report understates its own precision by roughly a factor of two.** Mislabelled — but in the
conservative direction, and worth saying so.

## 3.7 Machine record — verified exactly, and the conclusions survive it — D1, PASS

Every figure in S2.0 reproduces from `machine_samples.csv`:

| claim | verified |
|---|---|
| 494 samples, 16:39:39 → 16:55:54, no gaps | **yes** — 494 rows, span 975 s, max inter-sample gap 4 s |
| largest non-self process `Memory Compression`, max 1354 MB | **yes** — 494 of 494 samples |
| 0 samples with `llama-server` / `python` resident | **yes** |
| free physical 59,708–65,269 MB | **yes** |
| queue 0 in 80.0% of samples, mean 2.19 | **yes** — 80.0%, 2.19 |
| queue > 6 in 21 of 494 (4.3%), peak 143 | **yes** |

**Two defects in the sampler, neither of which launders anything.** It excludes competitors by process
*name* (`powershell`), so **any** PowerShell job on the machine is invisible to its "largest other process"
column; and it records only memory, never CPU, so a CPU-heavy low-memory competitor appears solely through
`cpu_queue`.

**The claim I could not verify is the one that matters:** *"Those spikes coincide with the 6-thread cells."*
**`stage2_raw.csv` carries no timestamp field**, so no cell can be mapped to any sample. That association is
an inference the artefacts cannot support and it should not have been stated as an observation. It is,
however, **partly corroborated and ultimately harmless**:

- Summed timed work is **801.2 s inside a 975 s span**, and per-invocation timed totals are **144.7, 141.6,
  163.9, 181.0, 170.0 s** — invocations 3–5 ran **13–25% slower**. All 21 queue spikes fall after 16:47:44,
  i.e. within invocations ~3–5. The trend and the spikes agree, so contention was real and **not random**.
- **But every load-bearing ratio is invariant across that split**, because byte/nibble and `+0`/`+64` cells
  run adjacent in time within each invocation and contention hits both members of each pair:

| ratio | inv 1–2 (clean) | inv 3–5 (spiked) | all 5 |
|---|---|---|---|
| `q/o` t6 stride `+64/+0`, byte | 1.664 | 1.684 | 1.675 |
| `q/o` t6 nib/byte @ `+64` | 1.889 | 1.931 | 1.912 |
| `gate/up` t6 stride `+64/+0`, byte | 1.604 | 1.745 | 1.681 |
| `gate/up` t6 nib/byte @ `+64` | 1.782 | 1.697 | 1.734 |
| `down` t6 nib/byte @ `+64` | 1.577 | 1.585 | 1.581 |

**The contention was real, the disclosure was honest, and the conclusions do not depend on it.** The
paired-in-time cell ordering is what protects them. The report does not claim this defence; it is available,
it is strong, and it should be made.

**Reproduction:** parse `machine_samples.csv` (494 rows) for the six statistics; recompute per-invocation
`Σ reps × mean_us` from `stage2_raw.csv`; recompute every ratio restricted to `inv ∈ {1,2}` and
`inv ∈ {3,4,5}`.

---

# 4. STAGE 1 — the controls

## 4.1 Everything countable reproduces exactly — E1, PASS

| claim | independent check |
|---|---|
| **523,395** exact int32 comparisons, 15 shapes, 0 mismatches | **exact.** Recomputed from the shape table and the harness's accounting: 243,993 `full` + 162,662 `tileskip` + 81,600 `rows` + 35,140 row-major = **523,395** |
| **18/18** plants, each changing exactly 1 row by the exact predicted delta | **18 lines in the log, all "as predicted"** — see the over-count note below |
| C2b reports `n/a` rather than a pass where `M == Mpad` | **correct** — 3 `n/a`, 3 real (7967 / 2048 / 31 padding bytes) |
| C2c on 3 odd-`T` shapes (`T` = 513, 4095, 7) | **correct** |
| C3 SHA-256 identity, 4 configs, 2 models | **verified by re-hashing all 11 `.logits` files myself.** Every hash matches `C3_SUMMARY.txt` |
| C4 `2.000000×` in-engine (3.000→1.500 MiB dense, 9.000→4.500 MiB MoE) | **verified.** The `_msize` reading is genuinely independent of the load counter |
| `engine.c` **+50 / −19** | **exact** — `git diff --numstat 565b4dd 637bab9 -- benchmarks/phase60/engine.c` → `50  19` |

**The tables are clean. I found no numerical error anywhere in Stage 1.**

**Two small prose over-counts.** (a) *"3 sites × 6 shapes"*: on `M33_T2`, `T/2 = 1`, so site 2's search
region collapses onto site 0's and the log records the **identical** plant twice (`t=(0,1) m=1 lo=8 hi=5`,
predicted −16, both times). **17 distinct plants, not 18.** (b) C3's *"4 configurations"* includes
`--mlp fp32 --exp exact`, where the LUT path is unused and byte-vs-nibble identity is **true by
construction**. It is labelled honestly and contributes nothing evidentially. **3 informative configs.**
Neither changes a verdict.

## 4.2 Is C1 capable of failing? Only across 46.6% of it — E2, FLAG

This is the question the mandate puts hardest, and it has a specific answer. C1 covers four paths.
**C2a re-runs exactly one of them.**

`c2_on()` plants into `cN` and then calls **`matvec_lut_full_n` only** (`nibble_pack_test.c`, the C2a block).
It never re-runs `matvec_lut_tileskip_n`, never `matvec_lut_rows_n`, and never the row-major scalar accessor
`np_rm_code`. By the breakdown in §4.1, the `full` path is **46.6%** of the 523,395 comparisons.
**For the remaining 53.4%, no known positive has ever been fired, and their passes therefore carry weight the
programme's own standing law says they cannot carry.**

**And one of the three unfired paths has a live vacuity hole.** Inside the trial loop,
`matvec_lut_tileskip_n` writes into `yN` — the same buffer `matvec_lut_full_n` filled *correctly* two
statements earlier:

```c
matvec_lut_full_n(cN, lut, yN, M, Mpad, T);          // yN := correct
...
matvec_lut_tileskip_n(cN, lut, yN, M, Mpad, act, na); // if this were a no-op, yN is STILL correct
if (cmp_i32(yN, yref, M, ...)) { FAIL }               // ... and this PASSES
```

**A tile-skip kernel that did nothing at all would pass C1 silently.** `matvec_lut_rows_n` is protected by
accident — it compares `yN[i]` against `yref[row0+i]` with `row0 ≠ 0`, so a no-op is caught at large `M` —
and the row-major scalar path is computed inline in the harness and cannot be vacuous. **`tileskip` is the
exposed one, and `--skip on` is the production configuration.**

**Reproduction / what would close it:** add a fourth plant site that corrupts a byte at a `t` present in
`act[]` and re-runs `matvec_lut_tileskip_n`, and a fifth inside `matvec_lut_rows_n`'s `[row0, row0+nrow)`
window; both are a handful of lines in `c2_on()`. Independently and more cheaply: `memset(yN, 0xFF, ...)`
between kernels, which converts every no-op into a failure.

**What is not in doubt.** `ref_t3` reads `Wt` directly and shares no code with either encoder or kernel, so
the `full`-path comparison is a genuine end-to-end check of encoder **and** kernel together; and C2a shows it
fires with the exactly predicted delta at 17 distinct sites across 6 shapes including two donor widths.
**That part of C1 is as strong as the report says.** The gap is coverage of the other three paths, not
soundness of the first.

**One coverage note.** Shapes with `trials = 1` — which is every donor-width shape — run only `tr = 0`, i.e.
weight fill mode 0 and activation mode 0 (uniform random). The all-`−1`, all-`+1` and sparse fills are
exercised only on the small shapes. Bit-exactness is structural so this is minor, but *"15 shapes"* is not
15 shapes × 4 fills.

## 4.3 The `--g3c` hard error — verified by reachability — E3, PASS

The report claims a hard error is the right behaviour and that it cannot be reached accidentally. **Both hold,
and I verified the second by reachability rather than by reading the guard:**

- `matvec_lut_full_K` is called at **three sites only** — `engine.c:709, 710, 713` — all inside
  `forward_block_lm`.
- `forward_block_lm` is defined at `:666` and is called **only** from the `run_g3c` region. `run_g3b`
  (`:607`) and `run_verify` (`:570`) are both defined *before* `:666` with no forward declaration, so neither
  can reach it.
- `emu_setup` (`:644`) has exactly one caller, `:748`, inside `run_g3c`.
- The guard sits at `:978`, immediately in front of the sole `run_g3c` call, and `g_pack_nib` is set by a
  **pre-scan at `:922`** that runs before the main argument loop — so no argument ordering can defeat it.

**`matvec_lut_full_K` is unreachable under `--pack nibble`.** The refusal is complete, and returning `1`
rather than warning is the correct call. **Agreed with the report, on evidence rather than on assertion.**

**Reproduction:** `grep -n 'matvec_lut_full_K\|forward_block_lm\|emu_setup\|g_pack_nib' benchmarks/phase60/engine.c`.

## 4.4 Two paths that ignore `--pack` silently — E4, FLAG

The same standard is not applied twice more:

- **`--kselftest` (`engine.c:870–900`)** builds its own arrays with `bc_tm` / `bc_rm` and calls
  `matvec_lut_full`, `matvec_lut_tileskip`, `matvec_lut_rows` **unconditionally**. Under `--pack nibble` it
  tests the **byte** arm and prints PASS. **The project's CI kernel self-test does not cover the nibble
  kernels at all** — which bears directly on S2.8's recommendation to adopt them.
- **`--exprate` (`engine.c:859, 862`)** calls `matvec_lut_full` on a synthetic byte pool regardless of
  `--pack`, so any timing number taken under `--pack nibble` would silently be a byte-arm number.

Neither is a correctness hazard — both are internally self-consistent — but both are **silent** where
`--g3c` correctly **refuses**. If a hard error is the right behaviour for one unconverted path, it is the
right behaviour for these two.

## 4.5 The two brief corrections — F1, F2, both PASS

**`2T/(T+1)`, not an unconditional 2×.** Derived independently: the byte arm streams `T` planes, the nibble
arm `Tp = ⌈T/2⌉ = (T+1)/2` for odd `T`, so the ratio is `T / ((T+1)/2) = 2T/(T+1)`. Checked against the
report's own C1/C4 table: `T=1` → 1.0000× (32 B vs 32 B ✔), `T=7` → 1.7500× (224 vs 128 ✔), `T=513` →
1.9961× (65,664 vs 32,896 ✔), `T=4095` → 1.9995× ✔; every even-`T` shape → 2.0000× ✔. **The correction is
right, and the report states it as a correction to the brief rather than burying it.** The associated
observation — that `Mpad` rounding dominates at small `M` and the nibble lever does not attack it — is also
correct and correctly scoped as negligible at donor width (`Mpad == M` at all four organs).

**No novelty claim, no "pshufb optimum".** I grepped the report for `optimum|novel|first to|unprecedented`.
**Four hits, all of them disclaimers or withdrawals**: S2.7's heading, *"no novelty claim enters any document
from this work"*, the explicit withdrawal of the optimum phrasing, and *"P1 is a catch-up, not a frontier"*.
The withdrawn phrasing appears **nowhere** in assertive form. bitnet.cpp TL1 is cited as prior art for the
identical `(w0+1)*3+(w1+1)` alphabet, and TL2's 1.667 bits/weight is cited as the reason 2.0 is not an
optimum. **Clean.**

**Reproduction:** `grep -niE 'optimum|novel|first to|unprecedented' docs/research/donor_adaptation/probes/P1_NIBBLE_PACKING.md`.

---

# 5. THE D0 SIGNATURE — present, and specific — G1, BLOCK on the prose

`CONTROLLER_D0_AUDIT.md` found: measured tables exact, prose over-stating, **every over-statement leaning the
same way.** The coordinator asks me to check P1's prose **against its own tables**, not only the tables
against the artefacts. **I did, and the signature is present — with more instances.**

**Every number I could recompute, reproduced.** 523,395 comparisons; 18 plant lines; all 11 SHA-256 hashes;
`2.000000×`; `+50 / −19`; all 16 S2.4 stride ratios; all 16 S2.5 nib/byte ratios; all six machine-sampler
statistics; the `2T/(T+1)` table. **Zero numerical discrepancies in the entire document.** As in D0, the
Builder's arithmetic is not the problem.

**Seven prose over-statements, every one in the direction of the finding, and — critically — five of the
seven are contradicted by the report's own tables:**

1. *"the measurement matches **4 for 4**"* — **contradicted by S2.4 itself**, which contains the `gate/up`
   nibble counter-example (§1.2).
2. *"`k/v` … is exactly the shape the mechanism predicts it should not help"* — the stated L1 mechanism does
   not predict that; 4 sets against 4096 lines is not a margin (§1.3).
3. *"worth **1.43–1.68×**"* presented as what the mechanism buys — 1.52–3.28× at one fixed stride (§1.4).
4. *"**moved** GB/s"* — charged bytes; **contradicted by the raw CSV**, where `moved_B == charged_B` in all
   200 rows (§3.1).
5. *"The moved GB/s column is **what saves the experiment**"* — **contradicted by S2.5's own table**, where
   it is `matvecs/s × 0.5` in every row (§3.2).
6. *"the fall past L3 is substantially cache-set aliasing, **not capacity**"* — false dichotomy, and "past
   L3" is not the operative variable (§2).
7. *"excluded from **every** conclusion drawn above"* — **contradicted by S2.4**, which uses the excluded
   cell in the report's single load-bearing sentence (§3.5).

Each has a benign reading in isolation. **Seven of them, all pushing one way, in a document whose
recommendation is "adopt this now", is the pattern D0 named.**

**The Builder's candour is real and belongs on the record.** It disclosed a confound the brief did not ask
about; contradicted the brief's own 2× claim; flagged its own superlinear cell as an artefact; reported a
machine spike that does not help it; used the per-rep mean and explicitly refused min-of-reps; guarded every
timed cell with a bit-exactness check; and refused a `--g3c` path rather than faking it. **None of that made
the prose accurate.** Candour and calibration are separate disciplines, and this document has the first
without the second.

---

# 6. WHAT SURVIVES

**Banked, unconditionally:**

- **Stage 1.** Bit-exactness on the `full` path against an independent scalar reference, end-to-end SHA-256
  logit parity across 3 informative configs and 2 models, the regression guard against the pre-patch binary,
  the padding and odd-`T` derivations *with* their empirical demonstrations (C2b/C2c), the `2T/(T+1)`
  correction, and the `--g3c` refusal. **The 2× byte reduction is available and bit-exact.** Coverage of
  `tileskip` / `rows` / row-major is *asserted*, not *demonstrated* (§4.2).
- **The stride effect exists, is large, is caused by the stride and by nothing else, and I reproduced its
  headline value to three significant figures on a machine I verified myself** (§1.4). The attribution
  control the report lacks comes out in the report's favour.
- **The D5 reconciliation** (§2) — the strongest corroboration in the file, and one the report does not make.

**Blocked pending correction:**

- The mechanism as stated (§1.3); the "4 for 4" score (§1.2); the effect-size band presented as a mechanism
  signature (§1.4); the moved-byte label (§3.1); and A1.3's discriminator (§3.2).
- **S2.8 recommendation 2 — *"then adopt nibble packing … Stage 2 says it buys a further 1.58–1.91×"* — is
  not supported.** That figure is confounded three ways (§3.3) and its second column cannot arbitrate
  (§3.2). Stage 1's 2× byte reduction is banked; the **speed** claim attached to it is not.
- **S2.8 recommendation 1 — adopt the `+64` stride pad — IS supported**, on a corrected mechanism, with the
  caveat that the magnitude is shape-dependent in a way nobody has yet modelled and should be **measured per
  organ** rather than assumed at 1.63–1.68×.
- **S2.8 recommendation 4 — *"do not re-litigate compute-vs-memory with another shape sweep"* — is correct
  and, on this audit's evidence, understated.** Every `Mpad` in the existing evidence base is a power of two
  or a multiple of 4096, and §1.4 shows the penalty is present at *every* working-set size at such a stride.
  **Every donor-width bandwidth number this programme holds should be re-read as partly a stride
  measurement** — including, per §2, both D5 rounds.

**Open, and what would close it:**

- **§1.5, the stride-advance sweep.** Built, compiles, not run. It tests the report's own stated invariant
  and would settle whether `adv = 16` is why `k/v` is flat. Needs a clear machine and about four minutes.
- **§3.3, an arm that separates bytes from load count.** Nothing in the existing data can do it.
- **§4.2, plants on `tileskip` and `rows`.** A handful of lines; until then 53.4% of C1 is unfired.

---

**Reproduction index.** Stage 1: `bin/nibble_pack_test.exe` → `p1/results/c1_c2_c4.log`; re-hash
`p1/results/c3/*.logits` and compare against `C3_SUMMARY.txt`. Stage 2: parse
`p1/results/stage2/stage2_raw.csv` (200 `P1S2CSV` rows, 5 `P1S2DONE`, 0 `P1S2ERR`) and `machine_samples.csv`
(494 rows). Engine diff: `git diff --numstat 565b4dd 637bab9 -- benchmarks/phase60/engine.c`. Reachability:
`grep -n 'matvec_lut_full_K\|forward_block_lm\|emu_setup' benchmarks/phase60/engine.c`. This audit's fresh
run: `benchmarks/donor_adaptation/p1/audit/ctrl_stride_probe.c` → `ctrl_stride_probe.csv` (54 cells, 3
invocations, clean machine). The unrun discriminator: `.../ctrl_stride_sweep.c`.

**No fixes implemented. Nothing committed. Nothing pushed. STOP.**
