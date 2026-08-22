# Controller Audit — `docs/prompts/master_prompts/DONOR_MODEL_ADAPTATION.md`

**Auditor:** the Controller (independent adversarial review, read-only on all repo code)
**Target:** the mandate document, revision "Revised 2026-08-20 by the Architect", 1135 lines
**Date:** 2026-08-20
**Scope:** (A) arithmetic; (B) repo-sourced claims; (C) internal contradictions and gate satisfiability
**Method:** every number recomputed independently from the document's own stated inputs; every repo
reference checked against the file at the named line; the sealed gate set tested for a non-empty
feasible region using the repo's own measured rate model (`docs/PHASE64_BUDGET.md` §1b/§2).

**Standing note on units.** Unless stated, I read `GB = 10^9 B` and `KB/MB = 10^3/10^6 B` for rates and
footprints, and `KiB` where the artefact is a cache line / expert block. Where the choice changes a
verdict I say so. The document never declares its unit convention; that is itself finding #21.

---

## Summary of verdicts

| # | tag | subject |
|---|---|---|
| 1 | **BLOCK** | §2.2 per-token compute floor "≈1.14 ms" is not the sum of its own decomposition |
| 2 | **BLOCK** | SKU-B's sealed gate set is provably empty over 1–100B |
| 3 | **BLOCK** | The 10 tok/s gate and the sealed fp32-SSM-projection finding are jointly unsatisfiable above ~2B |
| 4 | **BLOCK** | §8.A's headline eligibility claim is sourced to §5 but is the exact claim §5 forbids |
| 5 | **BLOCK** | §9 stage-1 ("all desk, all free", eliminates donors) requires an artefact from stage 3 |
| 6 | FLAG | Expert-overhead decomposition is falsified by the engine's own 1-thread datum |
| 7 | FLAG | "4.2 GB/s" is a unit-mixing artefact; 17.0/4.2 ≠ 3.9× as stated |
| 8 | FLAG | "5 trits per byte-pair" is off by 2×; "~2.4-2.5×" is exactly 2.5× |
| 9 | FLAG | "4 bits/weight (base-3, g=2 codes)" mis-describes the mechanism; 2× is already free |
| 10 | FLAG | §5 table uses 42 GB/s where the repo's own budget model prices streamed organs at 34-40 |
| 11 | FLAG | "Ternary shrinks bytes, not this floor" is false for ≥52% of the floor |
| 12 | FLAG | 848 / 1652 tok/s is an unmatched protocol pair; the repo's registered figure is 827→2029 |
| 13 | FLAG | §5 head arithmetic assumes tied embeddings without saying so |
| 14 | FLAG | "120B" example is outside the mandate's own 1B–100B donor range |
| 15 | FLAG | §S2 "≤ 64 GB" vs 60 GB: the "4 GB left" claim is unit-dependent |
| 16 | FLAG | "native 128K" (S3) is a sealed contract term that is never defined |
| 17 | FLAG | §8.F's "24-32M" is quoted against a single-CCX budget while the keystone is stated as aggregate |
| 18 | FLAG | §2.3 "matched total params" applies to only one of the two MoE comparisons |
| 19 | FLAG | §2.4's constant enumeration silently omits `AQ 63`, which §8.G.5 then depends on |
| 20 | FLAG | S4 retention gate is unmeasurable with S1 resources for the donors §5 admits |
| 21 | FLAG | No unit convention declared anywhere; several figures only reproduce under one reading |
| 22 | PASS | §5 streaming table — all four rows reproduce |
| 23 | PASS | §5 head figures — all four reproduce |
| 24 | PASS | §5 "120B × 0.5 = 60 GB" |
| 25 | PASS | §S2 SKU ceiling table — all four cells |
| 26 | PASS | Expert kernel figure "2.88 µs per 48 KB expert" |
| 27 | PASS | §2.4 code references — every one verified at the named line |
| 28 | PASS | §12 file table — every named path exists |
| 29 | PASS | §2.1 artefact table and §2.3 quality constants — all sourced |

---

# Class A — Arithmetic

## 1. BLOCK — the per-token compute floor "≈1.14 ms" is not the sum of its own decomposition

**Quoted claim** (§2.2, "Per-position compute floor"):

> "At the reference 8.3M config the engine's per-token compute floor is ≈ **1.14 ms**, decomposed
> (µs/token, 1 thread → 6 threads): proj-GEMV 662 → 272 (×2.43); LUT-MLP dense 313 → 207 (×1.51), MoE
> 573 → 543 (**×1.06, thread-flat** — the overhead above); scan-recurrence 169 → 46 (×3.66); SWA 92 →
> 65; head 40 → 8; norms and glue ~7."

**Recomputation.** Summing the 1-thread column exactly as printed:

```
dense, t1 : 662 + 313 + 169 + 92 + 40 + 7 = 1283 µs = 1.283 ms
MoE,   t1 : 662 + 573 + 169 + 92 + 40 + 7 = 1543 µs = 1.543 ms
dense, t6 : 272 + 207 +  46 + 65 +  8 + 7 =  605 µs = 0.605 ms
MoE,   t6 : 272 + 543 +  46 + 65 +  8 + 7 =  941 µs = 0.941 ms
```

**Neither t1 total is 1.14 ms.** The stated figure is reproduced *exactly* by summing only the first
three organs:

```
662 + 313 + 169 = 1144 µs = 1.144 ms
```

i.e. **the quoted "compute floor" silently omits SWA (92 µs), head (40 µs) and norms/glue (7 µs)** —
139 µs, 10.8 % of the true dense t1 total.

**Corrected values:** dense t1 **1.283 ms**; MoE t1 **1.543 ms**; dense t6 **0.605 ms**; MoE t6
**0.941 ms**.

**Why this is a BLOCK and not a rounding FLAG.** §4 leans its entire route-(iii) argument on the
fractions derived from this decomposition:

> §4: "at the sandbox config the scan recurrence is ~13-15% of the compute floor while proj-GEMV is
> ~52%."

Check the denominators. 662 / 1283 = **51.6 %** ✔; 662 / 1144 = 57.9 % ✘. 169 / 1283 = 13.2 %;
169 / 1144 = 14.8 %. **§4's "~52%" is only reproducible against 1283 µs — the number §2.2 says is
1.14 ms.** The two sections therefore use two different totals for the same quantity, in the same
document, and the §4 figure is the correct one.

**Repo evidence that these are two different measurements glued together.** The repo carries both,
labelled differently and never as a decomposition:

- `docs/SCALEUP_ARCHITECTURE.md:172` — "Current anchor: **C ≈ 1.14 ms/token** at Dn·N·L = 512·96·6"
  — C is the per-position cost from the V-G3 block-verify *emulation* (`docs/ENGINE_PLAN.md:133`:
  "the per-position compute floor (1.14 ms) exceeds the all-cold traffic (0.71 ms)").
- `docs/PHASE64_BUDGET.md:8` — the 662/313/573/169/92/40/7 table, from the separate 64.0 registered
  per-component run. The repo never claims it sums to 1.14 ms.
- `docs/SCALEUP_ARCHITECTURE.md:174` — "proj-GEMV dominates (**51.6 %** ≈ P61's 52.7 %, reproduced)",
  and `:186` — "proj-GEMV ≈53%, LUT-MLP ≈24%, scan recurrence ≈13%". 313/1283 = 24.4 % ✔.
  **The repo's own percentages are computed against 1283 µs.**
- `docs/SCALEUP_ARCHITECTURE.md:47` gives the honest range: "this floor is **~1.1-1.3 ms/token**".

The mandate collapsed an emulation anchor (1.14 ms) and a component table (1.283 ms) into one
sentence with the word "decomposed" between them. Per the document's own reading rule 1, that is
exactly a number quoted without its protocol.

**Third inconsistency in the same paragraph.** §2.2 also states "176 → 848 tok/s dense single-thread".
848 tok/s = **1.179 ms/token**. The dense t1 decomposition sums to 1.283 ms — the parts exceed the
measured whole by 8.8 %. The t6 column does not have this problem (605 µs → 1653 tok/s, matching the
quoted 1652 exactly). An additive decomposition that closes at t6 and over-closes by 9 % at t1 is a
measurement artefact that the document presents as an exact budget. It is the input to every donor
extrapolation in §5 and §9-stage-1.

**Required correction.** State the dense/MoE, t1/t6 totals as four numbers; state that 1.14 ms is the
V-G3 emulation anchor and is *not* the sum; propagate 1.283 ms into §4's percentages (which already
use it).

---

## 2. PASS — the §5 streaming table, all four rows

**Quoted formula** (§5): `streamed_bytes_per_token ≈ P_active × 0.5 B`;
`physical_time_lower_bound ≈ streamed_bytes_per_token / 42 GB/s`.

| donor | P_active | doc: streamed | my calc | doc: µs | my calc | doc: tok/s | my calc |
|---|---|---|---|---|---|---|---|
| ~30B/3B active | 3e9 | 1.5 GB | 1.500 GB ✔ | ~36,000 | 35,714 ✔ | 28 | 28.00 ✔ |
| ~120B/5B active | 5e9 | 2.5 GB | 2.500 GB ✔ | ~60,000 | 59,524 ✔ | 17 | 16.80 ✔ |
| ~8B dense | 8e9 | 4.0 GB | 4.000 GB ✔ | ~95,000 | 95,238 ✔ | 10.5 | 10.50 ✔ |
| ~70B dense | 70e9 | 35 GB | 35.00 GB ✔ | ~833,000 | 833,333 ✔ | 1.2 | 1.20 ✔ |

**Verdict: PASS.** Every cell reproduces to within its stated rounding, under decimal GB. See finding
#10 for the objection to the *rate* used, and #4 for the objection to what the table is then used for.

---

## 3. PASS — the head arithmetic, all four figures

**Quoted claim** (§5): "At D = 2048 and V = 152K the embedding/head matrix alone is ~311M parameters:
**1.24 GB in fp32, read every token for the output projection** — a ~30 ms *physical lower bound* at
42 GB/s".

```
params  = 2048 × 152,000            = 311,296,000  → "~311M"        ✔
fp32    = 311,296,000 × 4 B         = 1,245,184,000 B = 1.245 GB    ✔ ("~1.24"; 1.25 is the nearer round)
time    = 1.245e9 / 42e9            = 29.65 ms                      ✔ ("~30 ms")
implied throughput cap              = 1 / 0.02965 s = 33.7 tok/s
```

**Verdict: PASS** on all four. Two riders, logged separately as #13 and #10:
- 33.7 tok/s is only 3.4× the sealed 10 tok/s gate, from **one organ**. At the repo's fully-streamed
  floor of 34-40 GB/s (`PHASE64_BUDGET.md:32`) it is 31.1-36.6 ms → **27.3-32.1 tok/s**. Composed
  with the §5 row for a 120B/5B-active donor (60 ms), the head alone drops that row from 17 tok/s to
  **11.1 tok/s** — inside 11 % of the sealed gate. The document never composes these two of its own
  numbers.

---

## 4. PASS — "120B × 0.5 B/weight = 60 GB" and the 4 GB residual

`120e9 × 0.5 B = 60e9 B = 60 GB` ✔. `64 GB − 60 GB = 4 GB` ✔ under decimal GB.
**Verdict: PASS.** See #15 for the unit rider and #14 for the range rider.

---

## 5. PASS-with-rider — §2.2/§8.F "~24-32M active ternary params for a 16 MB budget"

```
16e6 B / 0.5 B per weight   = 32.0e6 weights   → the "32M" end is the exact zero-overhead ceiling ✔
16 MiB (16,777,216 B) / 0.5 = 33.55e6 weights  → 32M understates by 4.9 % under a MiB reading
the "24M" end implies 12 MB of codes + 4 MB (25 %) of scales/activations/pollution — plausible,
   but the document states no derivation for it
```

**Repo:** `docs/SCALEUP_ARCHITECTURE.md:43` — "**~24-32M active ternary params per token**"; and
`docs/SIZING.md:14` — "≤ 16 MB, ≈ 24-32M ternary". **Sourced correctly. PASS.**
Rider logged as #17 (single-CCX vs aggregate).

---

## 6. FLAG — the expert-streaming decomposition: arithmetic reproduces, the *inference* does not

**Quoted claim** (§2.2): "**Expert-pool streaming rate: kernel-pure 7.45 GB/s (1 thread) → 17.0 GB/s
(6 threads) = 2.88 µs per 48 KB expert.** The engine-integrated figure is 4.2 GB/s, and the ~3.9× gap
decomposes as **~8.4 µs per expert of dispatch overhead around the kernel** … That is an engineering
lever, **not** a bandwidth wall".

### 6a. The kernel figure: PASS

```
48 KiB = 49,152 B;  49,152 / 17.0e9 = 2.891 µs   → "2.88 µs" ✔
```
Corroborated independently from the engine's own geometry: an expert is `3 × HID_E × D` ternary
weights = `3 × 128 × 256 = 98,304` weights × 0.5 B = **49,152 B = 48 KiB exactly**
(`engine.c:52,69` → `D 256`, `HID_E 128`). The "48 KB expert" is not an approximation; it is the
literal object.

### 6b. The stated ratio does not reproduce from the stated rates — FLAG (#7)

```
17.0 / 4.2 = 4.048×      ← what a reader who checks gets
doc says   ~3.9×
```
The 3.9× *is* recoverable, but only from the time-domain figures the document does not print:
```
integrated per expert = 543 µs/token ÷ 48 experts/token = 11.3125 µs
11.3125 / 2.891 = 3.913×          → "~3.9×"   ✔
11.3125 − 2.891 = 8.42 µs         → "~8.4 µs" ✔
```
So the decomposition **is** internally consistent — but the "4.2 GB/s" figure it is presented against
is not. Recomputing that rate correctly:
```
bytes/token = 48 experts × 49,152 B = 2,359,296 B (= 2304 KiB)
rate        = 2,359,296 / 543e-6    = 4.345 GB/s
```
`PHASE64_BUDGET.md:9` writes "2304 KB / 543 µs ≈ 4.2 GB/s" — treating the numerator as
2,304,000 decimal bytes while the expert is a 48 **KiB** block. **The correct integrated rate is
4.34 GB/s, not 4.2**, and the document inherits the error. Minor in magnitude (3 %), but it makes the
headline ratio unreproducible from the headline numbers, which is precisely the failure mode the
document's own reading rule 1 exists to prevent.

### 6c. The decomposition is falsified by the engine's own 1-thread datum — FLAG, load-bearing

The claim is that ~8.4 µs/expert is *fixed dispatch overhead around the kernel*. Apply the same
subtraction at 1 thread, using the document's own numbers (kernel-pure 7.45 GB/s, MoE 573 µs/token):

```
kernel, t1     : 49,152 / 7.45e9        = 6.598 µs/expert
integrated, t1 : 573 µs ÷ 48            = 11.938 µs/expert
overhead, t1   : 11.938 − 6.598         = 5.34 µs/expert
overhead, t6   :                          8.42 µs/expert   (§6b above)
```

**The "fixed dispatch overhead" grows by 58 % when threads are added.** A serial glue cost cannot do
that. Run the model forward as a prediction and it fails:

```
predicted MoE t1 = (6.598 + 8.42) × 48 = 721 µs/token
measured  MoE t1 =                       573 µs/token      → 26 % over-prediction
```

The residual is equally consistent with *contention* that the weight-free microbenchmark
(`benchmarks/phase64/bench_64_1b.sh`, synthetic weights, no dequant/dReLU/combine, no L3 pollution
from a live engine) simply does not exercise. That reading predicts the observed sign — overhead rising
with thread count — while the "engineering lever" reading does not.

This is the document's own law 8 (§6) and its own Phase-61 lesson, violated in its own §2.2: *a
microbenchmark rate does not compose to an engine rate.* The document cites that law four times and
then subtracts a microbenchmark from an engine measurement and calls the residual a component.

**Required correction:** report the residual as a bracket, not a component — "the integrated expert
path runs at 4.34 GB/s against a 17.0 GB/s kernel-pure ceiling; the 3.9× gap is unattributed and its
magnitude is thread-dependent (5.3 µs at t1, 8.4 µs at t6), which is inconsistent with a fixed serial
dispatch cost." Then it is honest, and it is still an engineering lead.

---

## 7. FLAG — the ternary packing claim: right number, wrong mechanism, understated headroom

**Quoted claim** (§8.G): "Ternary weights are packed at **4 bits/weight today** (base-3, g=2 codes) =
0.5 B/weight. The dense trit-pack (**5 trits per byte-pair, ~1.6 bits/weight = 0.2 B/weight**) is
designed but NOT built and would be a **~2.4-2.5×** reduction".

### 7a. `4 bits/weight = 0.5 B/weight` — arithmetic PASS, verified in code

`4 / 8 = 0.5` ✔. Verified against the artefact, not the table (§6 law 5):
`engine.c:160-162`, `bc_tm`: `int T = K/2; … codes[t*Mpad+m] = (int8_t)((w0+1)*3 + (w1+1));`
→ `K/2` **bytes** for `K` weights = **0.5 B/weight** ✔. Confirmed by allocation:
`egate_cd[l] = xmalloc(TUP * MPAD_GU)` with `TUP = D/2 = 128`, `MPAD_GU = 4096` → 524,288 B for a
`GH×D = 4096×256 = 1,048,576`-weight matrix = 0.5 B/weight ✔.

### 7b. "(base-3, g=2 codes)" mis-attributes where the 4 bits go — FLAG (#9)

`(w0+1)*3 + (w1+1)` produces a value in `[0, 8]` — **9 states, which need 4 bits total, i.e. 2 bits
per weight.** The engine stores that 9-state code in a **full int8 byte**, because `_mm256_shuffle_epi8`
needs one index per byte lane. So:

```
information in a g=2 base-3 code : log2(9)  = 3.17 bits per PAIR  = 1.585 bits/weight
code width as designed           : 4 bits   per PAIR              = 2.0   bits/weight
as actually stored               : 8 bits   per PAIR              = 4.0   bits/weight   ← today
```

The parenthetical "(base-3, g=2 codes)" reads as if 4 bits/weight is what that scheme costs. It is
**2× what that scheme costs.** Half of every stored byte is pshufb lane padding, not code.

**Consequence the document misses.** It presents trit-pack as the only route to 2.4-2.5×. But
nibble-packing the *existing, already-validated* g=2 code — no new code design, no new information
theory, the same `bc_tm` output packed two codes per byte and unpacked in the shuffle path (which is
what T-MAC and bitnet.cpp do, both already cited in §11) — gives **0.25 B/weight, a 2× win**, and the
remaining trit-pack gain over *that* is only 1.25×, not 2.5×. §8.G.2 calls trit-pack "the
highest-value known-quantity engineering item in the document" while omitting the cheaper 2× sitting
in front of it. That materially misprices the §8.G.2 decision the Architect is asked to schedule.

### 7c. "5 trits per byte-pair" is off by 2× — FLAG

```
5 trits in 16 bits (a byte-PAIR) = 3.2 bits/weight     ← what the document literally says
5 trits in  8 bits (a byte)      = 1.6 bits/weight  ✔  (3^5 = 243 ≤ 256, 99.06 % efficient)
10 trits in 16 bits (byte-pair)  = 1.6 bits/weight  ✔  (3^10 = 59,049 ≤ 65,536, 99.06 %)
```
The stated grouping and the stated bit rate are inconsistent. Correct wording: **"5 trits per byte"**
or **"10 trits per byte-pair"**. Shannon floor `log2(3) = 1.58496` bits/weight; 1.6 is 0.95 % above it.

### 7d. "~2.4-2.5×" is exactly 2.5× — FLAG

```
4 bits / 1.6 bits  = 2.500
0.5 B  / 0.2 B     = 2.500
```
There is no derivation anywhere in the document for a 2.4 lower end. Either state the overhead that
produces it (per-row scales? unpack cost?) or write 2.5×. As written it is a hedge with no referent,
and the document's own rule 1 forbids exactly this.

---

## 8. PASS — §S2 SKU ceiling table, all four cells

```
SKU-A @0.5 B/w : 16e9 / 0.5 = 32e9   → "≲ 32B"  ✔
SKU-A @0.2 B/w : 16e9 / 0.2 = 80e9   → "≲ 80B"  ✔
SKU-B @0.5 B/w : 64e9 / 0.5 = 128e9  → "≲ 128B" ✔
SKU-B @0.2 B/w : 64e9 / 0.2 = 320e9  → "≲ 320B" ✔
```
**PASS**, and the document is right to label the table "a deliberately unusable upper envelope".

---

# Class B — Repo-sourced claims

## 9. PASS — every §2.4 code reference verified at the named line

| claim | verification |
|---|---|
| "`engine.c:51-75` declares V 1024, D 256, N 96, H 8, L 6, DN 512, DTR 16, CONV 4, WIN 128, SWA_LAYER 5, MLP_HID 1024, E 32, HID_E 128, KTOP 8 as preprocessor constants" | ✔ `engine.c:51` = `#define V 1024`; `:52` = `#define D 256`; `:70` = `#define KTOP 8`; `:75` = `#define NLAYER (L+2)`. All 14 named constants fall inside 51-75, with the derived ones (`HD`, `AQ`, `TUP`, `TDN`, `GH`, `MPAD_*`, `TDE`, `NLAYER`) also in range. **PASS** — see #19 for the omission rider. |
| "loader at `:194-240`" | ✔ `static void load_weights(const char* path){` is at `:194`; the next function `load_meta` starts at `:241`. The span is exact. **PASS** |
| "`is_swa[l] = (l == SWA_LAYER)` at `:202`" | ✔ `:202` reads `for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);`. **PASS** |
| "`load_weights` validates only `E`/`hid_e`/`k` (MoE) or `mlp_hid` (dense) against its own defines" | ✔ `:198` `if(g_moe && ((int)h[11]!=E \|\| (int)h[12]!=HID_E \|\| (int)h[13]!=KTOP)){…exit(1);}`; `:199` `if(!g_moe && (int)h[11]!=MLP_HID){…exit(1);}`. No other header word is compared to a define. **PASS** |
| "The weight file's 16-word header *carries* V, D, N, H, L, Dn, dt_rank, conv, win, swa_layer, E, hid_e, k" | ✔ `e4_export.py:39` `struct.pack("<16I", MAGIC, V,D,N,H,L,Dn,DTR,CONV,WIN,swa,E,hid_e,k,1,0)`; `e4_export.py:8` documents the same layout. **PASS** |
| "A file with different dimensions is read as garbage or fails on a short read" | ✔ `:188` `static float* rd(FILE*f,size_t n){ … if(fread(p,4,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} … }` — sizes come from the defines (`rd(f,(size_t)V*D)` etc.), never from the header. A dimension mismatch that happens to leave enough bytes is read as garbage; the only backstop is `if(pos!=end) fprintf(stderr,"WARN %ld trailing bytes")` — **a warning, not a refusal.** The document's characterization is correct and if anything understated. **PASS** |

Two additional validations exist that §2.4 does not mention, neither of which changes the claim: the
magic word (`h[0] ∈ {0x45314D31, 0x45344D31}`, `:197`) and `has_packed = h[14]` (`:200`, enforced at
`:214`). Note for the record that the dense format's `h[11..13]` are `mlp_hid, gated, ternary`
(`e1_export.py:67`), not `E/hid_e/k` — the document's field list is the MoE variant, which its own
"(MoE) or mlp_hid (dense)" phrasing covers.

## 10. PASS — every file named in the §12 table exists

| §12 entry | status |
|---|---|
| `benchmarks/phase60/engine.c` | ✔ exists, 956 lines; all three line refs verified above |
| `benchmarks/phase60/e4_export.py`, `e1_export.py` | ✔ both exist; magics `0x45344D31` / `0x45314D31` confirmed |
| `Makefile` (`make engines`, `make selftest`, `make gates`) | ✔ all three targets exist (`Makefile:32,36,53,63`) |
| `docs/SCALEUP_ARCHITECTURE.md` | ✔ |
| `docs/PHASE64_BUDGET.md` | ✔ (rate curves confirmed at §1b) |
| `docs/PHASE64_DECISIONS.md` | ✔ (D1-D9 present; D9 = "Precision map: fp32 organs, sealed for the ladder", `:72`) |
| `docs/SIZING.md` | ✔ (its "Explicit unknowns" section is real, `:49`) |
| `docs/CANONICAL_EVAL.md` | ✔ |
| `docs/REPRODUCE.md` | ✔ |
| `benchmarks/phase56/` | ✔ |
| `HANDOFF.md`, `docs/silicon_book/` | ✔ |
| `graphify-out/` (`graph.json`) | ✔ |
| `archive/benchmarks/phase60_stage_engines/` (§2.1) | ✔ |
| `bin/recall_probe.exe` (§2.1) | ✔ |
| `results/phase60/*.bin` (§2.4 parity target) | ✔ 9 files: `e1_model.bin`, `e4_model.bin`, 6 goldens |
| `--block K`, `--threads N`, `--kselftest` (§2.1, §8.G) | ✔ `engine.c:899`, `:898`, `:921`; `--kselftest` also wired into `make selftest` across all five stage engines (`Makefile:54-58`) |

**Verdict: PASS.** No dangling reference. This part of the document is clean.

## 11. PASS — §2.3 and §2.2 quality constants are all sourced

| claim | source |
|---|---|
| "E32×h128 top-8 measured BPB 0.8589 vs dense-1024 0.8799 and dense-4096 0.8674 … E8×h512 top-2 = 0.8637" | ✔ `docs/CANONICAL_EVAL.md:36,38,39,40` (`sp58_base` 0.8799, `moe_dense_big` 0.8674, `moe_gran` 0.8589, `moe_coarse` 0.8637). See #18 for the "matched total params" rider. |
| "ternarizing the SSM projections: +0.018 to +0.022 BPB → rejected" | ✔ `docs/ENGINE_PLAN.md:87` — arm A 0.8940, arm B 0.8972 vs gate ≤0.8899, "+0.018 / +0.022 vs the fresh base" |
| "σ_seed = 0.005 BPB" | ✔ `docs/CANONICAL_EVAL.md:57`, `ENGINE_PLAN.md:87` |
| "dReLU vs SiLU: +0.0006 BPB" | ✔ `docs/CANONICAL_EVAL.md:57` |
| "ternary QAT MLP +0.028 BPB" | ✔ `docs/EXTERNAL_REVIEW_01.md:20,46` |
| "in-place predictability 86-92 %" | ✔ `PHASE64_DECISIONS.md:87`, `PHASE64_TRAINING_PLAN.md:64` (property-gate band). Note the Phase-62 *code* measurement is 83-88 % (`ENGINE_PLAN.md:3`); the document quotes the gate band, which is defensible, but the two coexist in the repo and the document does not say which it means. |
| recall tier "29.05 µs/token at 128K at 6 threads (52.4 µs at 1 thread, scalar ADC), 1.69 MB searchable + 64 MB values" | ✔ `PHASE64_DECISIONS.md:54`, verbatim |
| r(size) curve "187/185/134/60.5/55.7/45.5/45.3/36.5 at 4/8/16/24/32/48/64/96 MB; t1 30.5 → 23.3" | ✔ `PHASE64_BUDGET.md` §1b(a), verbatim |
| "DRAM cold-stream 21-26 GB/s single, aggregate 40-44 GB/s, saturated at 3 threads" | ✔ `PHASE64_BUDGET.md:5` |
| "~48 KB bulk-contiguous chunk loads pay only ~2.5×" | ✔ `PHASE64_BUDGET.md` §1b corollary (ii) |
| "ternary parts stored both dequant-fp32 and packed int8+scale" | ✔ `e4_export.py:5,12`; `engine.c:210-212` reads `egate_f` (fp32) then `egate_wt` (int8) + `egate_sc` |

**Verdict: PASS.** Class B is the strongest part of the document.

## 12. FLAG — 848 / 1652 tok/s is an unmatched protocol pair, and the repo's headline threading figure is missing

**Quoted claim** (§2.2): "**Speed achieved:** 176 → 848 tok/s dense single-thread across the Phase 60
optimization ladder (4.8×); **1652 tok/s dense at 6 threads**, bit-identical."

As printed this reads as one machine, one protocol: 848 → 1652 = **×1.95** threading gain. The repo's
registered threading result is different:

- `docs/ENGINE_PLAN.md:113` (63.T, adopted): "**T-G2 PASS: dense-full 827 → 2029 tok/s at t6-spread =
  2.45×; MoE-full 635 → 1519 at t6-close = 2.39×**".
- `docs/PHASE64_BUDGET.md:38` (64.0 protocol): "dense-t6 = predicted 980-1707 tok/s vs **measured
  1652**".
- `docs/ENGINE_PLAN.md:113` also carries the unresolved bookkeeping: "attribute the t1 deltas vs the
  historical singles (**dense 827 vs 848, −2.5 %**; MoE 635 vs 701.7, −9.5 %) … the −9.5 % on MoE is
  too large to leave unattributed." **That item is still open in the repo.**

So 848 (Phase-60 ladder, znver2 build, no OpenMP) and 1652 (64.0 protocol) come from different builds
and different runs, and the document composes them into an implied ×1.95 that contradicts the
project's own adopted ×2.45. It also omits the pinning policy — `t6-spread` vs `t6-close` is a
measured 2.45×-vs-2.39× difference in this repo — which S4 requires ("declared thread/pinning policy").

**Corrected statement:** dense t1 827 (generic x86-64-v3, OMP built in) or 848 (Phase-60 historical,
−2.5 % unattributed); dense t6 2029 at spread pinning (63.T) or 1652 under the 64.0 protocol; the
adopted threading factor is ×2.45.

---

# Class C — contradictions, un-gated hatches, and gate satisfiability

## 13. BLOCK — §8.A's headline eligibility claim is sourced to §5 and is the exact claim §5 forbids

**§5 states its own disclaimer, unambiguously:**

> "**This table does not predict engine tok/s.** 42 GB/s is the aggregate DRAM ceiling, while the
> existing integrated MoE path measures 4.2 GB/s and includes dispatch overhead; neither scales
> automatically to a donor layout. … **It may rank candidates, but it cannot establish the 10 tok/s
> gate.**"

**§8.A then writes, under "Known / strongly suspected":**

> "**§5's arithmetic says activation sparsity in the donor is worth more than small total size.** A 30B
> sparse-MoE donor is a better candidate than an 8B dense one **on the streaming term alone**."

That comparison is only true if both donors are priced at 42 GB/s. Price them at the rates the repo
actually measured for the two paths involved — `PHASE64_BUDGET.md:32-33`, the project's own budget
model, which prices **LUT-MLP (streamed experts) at [4.2-17.0 GB/s]** and **fully-streamed proj-GEMV
at [34-40 GB/s]**:

```
30B MoE, 3B active  → 1.5 GB/token via the EXPERT path
   at 4.34 GB/s (integrated today) : 346 ms →  2.9 tok/s
   at 17.0 GB/s (kernel-pure t6)   :  88 ms → 11.3 tok/s

8B dense, 8B active → 4.0 GB/token via the RESIDENT/streamed proj path
   at 34 GB/s (streamed floor)     : 118 ms →  8.5 tok/s
   at 40 GB/s (streamed ceiling)   : 100 ms → 10.0 tok/s
```

At the **measured integrated rate**, the 8B dense donor beats the 30B MoE donor by **2.9-3.4×** — the
opposite of §8.A's ranking. The MoE donor only wins if the ~8.4 µs/expert overhead is fully
eliminated, which finding #6 shows is not established, and which §8.A does not state as a
precondition. §8.A presents a conditional as "Known / strongly suspected", cites §5 as its authority,
and drops §5's disclaimer.

**This is not cosmetic.** §9 stage 1 uses this ranking to *eliminate donors before any compute*, and
§8.A.3 and §8.D.3 both build on the "sparse-MoE donor is the best class" premise. A wrong ranking at
stage 1 propagates into every downstream stage and the mandate has no mechanism to revisit it.

**Required correction:** §8.A must price both candidates at the measured integrated rates and state
explicitly that the MoE advantage is *conditional on the expert-dispatch fix landing first* — which
makes the dispatch fix a stage-0 blocker, not an §8.G "engineering lever" for later.

---

## 14. BLOCK — the 10 tok/s gate and the sealed fp32-SSM-projection finding are jointly unsatisfiable above ~2B

The document seals two things and never multiplies them together:

- **S4:** "the complete engine must sustain **≥10 generated tok/s in decode** for each SKU" — SEALED.
- **§2.2:** "ternarizing the SSM *projections*: +0.018 to +0.022 BPB → **rejected, they stay fp32**"
  (Phase 61; `PHASE64_DECISIONS.md:72` D9 "fp32 organs, sealed").
- **S3:** attention on a **minority** of layers ⇒ the **majority** of donor layers become SSM layers,
  each carrying fp32 SSM projections.

The repo's own scaling law for that organ (`PHASE64_BUDGET.md:32`):

> proj-GEMV bytes = **11 MB · (D/256)² · (L/6)**; t = bytes / r(bytes), fully-streamed floor
> **[34-40 GB/s]**

I verified the 11 MB anchor against the artefact rather than the table. Per SSM layer at the reference
config (`engine.c:52-58`, `:204-208`): `in_proj 2·Dn·D = 262,144` + `x_proj (DTR+2N)·Dn = 106,496` +
`out_proj D·Dn = 131,072` + `A Dn·N = 49,152` + `dt_proj Dn·DTR = 8,192` + conv/bias/Dskip ≈ 3,584
= **560,640 fp32 params = 2.243 MB/layer × 5 SSM layers = 11.2 MB** ✔.

Now evaluate the gate. 10 tok/s ⇒ 100 ms/token for **everything**. The projections alone, at the
optimistic 40 GB/s end:

| donor | D | L | (D/256)²·(L/6) | proj bytes/token | t_proj @40 GB/s | tok/s from projections **alone** |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 2048 | 16 | 170.7 | 1.88 GB | 47 ms | 21.3 |
| Qwen2.5-1.5B | 1536 | 28 | 168.0 | 1.85 GB | 46 ms | 21.7 |
| Qwen2.5-3B | 2048 | 36 | 384.0 | 4.22 GB | 106 ms | **9.5 ✘** |
| Llama-3.2-3B | 3072 | 28 | 672.0 | 7.39 GB | 185 ms | **5.4 ✘** |
| Phi-3-mini 3.8B | 3072 | 32 | 768.0 | 8.45 GB | 211 ms | **4.7 ✘** |
| Llama-3.1-8B / Mistral-7B | 4096 | 32 | 1365 | 15.0 GB | 375 ms | **2.7 ✘** |
| Qwen3-30B-A3B | 2048 | 48 | 512.0 | 5.63 GB | 141 ms | **7.1 ✘** |
| ~32B dense | 6144 | 64 | 6144 | 67.6 GB | 1690 ms | **0.6 ✘** |

Solving the gate directly: `11 MB · (D/256)² · (L/6) < 100 ms × 40 GB/s = 4.0 GB` ⇒
**(D/256)² · (L/6) < 364**. Essentially every released donor above ~2B parameters violates it — and
this is before a single byte of MLP/expert weight, before the head (finding #3: 30 ms on its own),
before KV traffic, before the LUT-MLP term, and before §5's entire table.

**§5 admits the omission and then does not price it:** "It omits scales/padding, per-selected-expert
dispatch, **the per-position compute floor**, head, attention/KV traffic, and all glue." That omitted
term is the **dominant** one at donor scale — 51.6 % of the floor at 8.3M and growing as D², the
fastest-growing term in the entire budget. A reader following §5 will conclude the 8B-dense row is the
marginal case at 10.5 tok/s. It is not: it is **2.7 tok/s**, a 3.9× miss.

**And §2.2 actively misdirects here:** "**Ternary shrinks bytes, not this floor.**" That is false for
the dominant component. `PHASE64_BUDGET.md:9` states plainly that proj-GEMV at t6 runs at
"**≈40 GB/s = the aggregate bandwidth ceiling** — the fp32 projections already run at aggregate BW
when threaded". A term that is running at the memory ceiling is *bytes*-bound by definition, and
ternarizing it would cut it 8×. Phase 61 rejected that on **quality** grounds at 8.3M with a
from-scratch QAT model — a completely different regime from a donor conversion, as §8.C.1 itself
insists ("They do not transfer to a post-training setting and must not be quoted as if they did"). The
document quotes it as if it did. Logged separately as #11.

**Even granting ternary projections** (0.5 B/w, 8× fewer bytes), Llama-3.1-8B: proj 1.88 GB → 47 ms,
plus MLP 5.6G params × 0.5 B = 2.8 GB → 70 ms, plus head → **≥120 ms = 8.3 tok/s.** Still a miss.

**Constructive half — the one route that survives.** Keep attention on the *majority* of layers
(§4 route (ii), which requires an S3 amendment) and ternarize everything including attention.
Llama-3.1-8B: QKVO = 4·D²·L = 2.15G params × 0.5 B = 1.07 GB → 27 ms; MLP 3·4096·14336·32 = 5.64G ×
0.5 B = 2.82 GB → 70 ms; ternary head 0.16 GB → 4 ms; total ≈ **101 ms ≈ 9.9 tok/s** — marginal, and
only at the 40 GB/s ceiling. **The sealed gate set therefore selects route (ii) — the route the
document frames as "concedes part of the thesis" — before any measurement is taken.** That is worth
knowing at stage 0 and the document does not know it.

---

## 15. BLOCK — SKU-B's sealed gate set is provably empty over the entire 1-100B range

This is the finding the audit was asked to look for, and it holds.

**Definition of the question.** SKU-B exists to serve donors that do **not** fit SKU-A. From §S2, a
donor needs SKU-B only if its resident inventory exceeds 16 GB, i.e. at 0.5 B/weight
**> 32B total parameters** (and more than that once non-ternary organs, KV and scratch are charged —
§S2's own inventory equation). SKU-B's own gates are: ≤64 GB resident, ≥10 tok/s decode, **128K native
context**, ≥90 % global retention.

**The elimination.** Every donor above 32B fails 10 tok/s by an order of magnitude, on the fp32
projection term alone (finding #14):

```
32B-class donor (D≈6144, L≈64), SSM-converted, projections fp32 per §2.2/D9:
  proj bytes/token = 11 MB · (6144/256)² · (64/6) = 67.6 GB per token
  → 1690 ms/token at 40 GB/s = 0.59 tok/s              (17× under the gate)
  → and 67.6 GB of resident fp32 projections EXCEEDS SKU-B's entire 64 GB budget by itself,
    so the donor also fails §S2 before it fails §S4.

70B dense, ternary everything, streaming term ONLY (§5's own row):
  35 GB/token → 833 ms = 1.2 tok/s                      (8.3× under the gate)

100B dense (the top of the mandate's range), ternary:
  50 GB/token → 1190 ms = 0.84 tok/s                    (12× under the gate)

120B/5B-active MoE (§5's own row), best case:
  2.5 GB/token at 17.0 GB/s kernel-pure = 147 ms = 6.8 tok/s   (still under the gate)
  at the measured integrated 4.34 GB/s  = 576 ms = 1.7 tok/s
  + 30 ms head (finding #3) either way
```

**Therefore: no donor in 1-100B simultaneously (a) requires more than SKU-A's 16 GB and (b) reaches
10 tok/s.** The intersection is empty. Under trit-pack (0.2 B/weight — explicitly "**designed but NOT
built**", §8.G) the threshold moves from 32B to 80B and the 70B/100B rows improve to 3.0/2.1 tok/s —
**still empty.** SKU-B cannot be rescued by the one unbuilt lever the document names.

**What this means operationally.** §S2 defines two SKUs and §S4 seals gates "for each SKU". One of the
two SKUs has no admissible donor. Every §S2/§S3/§8.K deliverable that says "the same binary must serve
both SKUs", every KV-at-128K-native calculation for SKU-B, and stage 5's "measure the sealed retention,
10 tok/s, RAM, and SKU-context gates" for SKU-B, are work against an empty set. Per §1.1 this is an
**AMENDMENT REQUEST**, and it is available *now*, at stage 0, from desk arithmetic — exactly what §9
stage 1 says desk arithmetic is for.

**The precise amendment to request:** either (a) drop SKU-B, or (b) drop SKU-B's 10 tok/s gate to a
value the arithmetic admits (≈1-3 tok/s for a 70-100B donor, which is an honest "usable for batch, not
for chat"), or (c) re-scope SKU-B to the 8-32B band, in which case it overlaps SKU-A and the two-SKU
structure collapses to one. This is a decision for the Owner and the Architect; I do not adjudicate it.

**Falsification path for this finding** (I want it broken if it is wrong): the finding rests on
`11 MB · (D/256)² · (L/6)` (repo `PHASE64_BUDGET.md:32`, anchor independently verified above), on the
[34-40 GB/s] fully-streamed floor (repo `:32`), and on projections staying fp32 (repo D9 + §2.2). Break
any one of those three and re-run the table. In particular, a donor conversion that chooses an SSM
expansion factor below the engine's `Dn = 2D` moves the D² coefficient directly, and nothing in the
mandate forbids that — but nothing in the mandate *proposes* it either, and it is not a free choice
(it changes the QKV→SSM dimensional map that §8.B.2 is built on).

---

## 16. BLOCK — §9's stage 1 requires an artefact produced at stage 3

**Quoted, §9 stage 1:** "**The arithmetic pass.** Redo §5 properly for a shortlist of real donors:
footprint inventory, streaming budget, KV budget, head cost, per-SKU eligibility. **All desk, all
free.** … gate: a ranked shortlist with the arithmetic shown; **any donor whose arithmetic fails is
eliminated here, before any compute**."

**Quoted, §5, defining what that arithmetic must be:**

> "The stage-1 arithmetic must replace it with an implementation-aware budget:
> `t_token_estimate = Σ(bytes_organ / measured_engine_rate_organ) + Σ(selected_expert_calls ×
> measured_dispatch_overhead) + …`
> **Measure the relevant kernel and integrated-engine rate at donor-like dimensions before promoting a
> candidate.**"

**The contradiction.** `measured_engine_rate_organ` **at donor-like dimensions** cannot be measured,
because §2.4 establishes that the engine has no runtime dimensions and stage 3 is where it gets them.
Stage 1 is therefore either:
- (a) genuinely desk work, in which case it must use the *extrapolated* rate model, not measurements,
  and §5's "measure … before promoting a candidate" is unsatisfiable at stage 1; or
- (b) a measurement stage, in which case it is neither "desk" nor "free" and it must be sequenced
  after stage 3.

The document asserts both. And it is a gate-bearing contradiction: stage 1 **eliminates donors
permanently** ("before any compute"), so an under-specified stage 1 permanently removes candidates on
an unmeasured basis, and the §9 stopping-point list gives no route to reopen an elimination.

**Related but weaker (FLAG, not BLOCK):** stage −1 runs a fidelity comparison ("perplexity/
log-likelihood … under a pre-registered threshold") before stage 4a builds and planted-control-validates
the eval harness that §8.N says must exist "**before** the harness is used to judge any conversion".
Stage −1 does carry its own "planted corruption control", which is the mitigation §6.3 asks for, so
this is a documentation gap rather than a defect — but the two sections should be made to agree
explicitly about which comparator stage −1 is allowed to use.

**Required correction:** split stage 1 into 1a (desk, extrapolated rate model, eliminations marked
*provisional*) and 1b (measured at donor dimensions, after stage 3, eliminations *final*); or move the
"measure at donor-like dimensions" requirement out of §5 and into stage 5's precondition.

---

## 17. FLAG — "native 128K" is a sealed contract term that is never defined

**Quoted, S3:** "**Context contract, SEALED.** SKU-B must provide **128K native context** within its
64 GB whole-process budget. SKU-A must provide a **128K user-visible context contract** within 16 GB:
native retained attention may use only a measured 8K-32K window, and the engine recall tier must supply
the long-range path for the remainder."

"Native" is never defined. Three readings are all consistent with the document, and they have different
gates:

1. **Full (non-windowed) attention over 128K on the retained attention layers.** Then KV read traffic
   at decode is the entire cache every token. Worked example, Llama-3.1-8B geometry (L=32, n_kv=8,
   d_h=128) with attention retained on 8 of 32 layers and KV at int8:
   `2 × 8 layers × 8 kv-heads × 128 × 128K × 1 B = 2.15 GB/token` → **51 ms at 42 GB/s, i.e. half the
   entire 10 tok/s budget consumed by KV reads alone**, on top of finding #14's projection term. This
   reading makes SKU-B doubly impossible.
2. **Sliding-window attention (already implemented, `WIN 128`) plus SSM state.** KV is bounded and
   trivial — but then SKU-B is architecturally *identical* to SKU-A minus the recall tier, and the
   SKU-A/SKU-B distinction in S3 collapses.
3. **SSM O(1) state counts as "native".** Then every configuration is trivially "native 128K" and the
   contract has no teeth at all.

Under reading 2 or 3 the sealed distinction between the SKUs is vacuous; under reading 1 it is
unreachable. A SEALED contract whose central term admits a vacuous reading and an impossible reading
and nothing in between is not yet a contract. §8.I.1 asks the Adapter to "compute it explicitly per
donor" — but which of the three it is computing is the Architect's call, not the Adapter's, and it
should be fixed before stage 1 since it reorders the entire shortlist (§8.I.1's own words).

---

## 18. FLAG — S4's retention gate is unmeasurable with S1's resources for exactly the donors §5 promotes

**S4, SEALED:** "**Primary metric: retention versus the unconverted donor**, measured on standard
general benchmarks, **same harness, same prompts, same decode settings**."
**§8.N.2:** "How do you evaluate a 60 GB model on the reference machine? Possibly you cannot, and the
donor baseline must be established elsewhere (**published numbers are not an acceptable substitute**)."
**S1:** T4 / P100 class (16 GB VRAM), ~9-hour sessions, ~90 GPU-h/week.

The document poses this as an open question (§8.N.2) but it is closed by arithmetic. A 30B donor at
bf16 is 60 GB — it does not fit a 16 GB T4 or P100 even 4-way sharded within one session, and the
reference machine (80 GB RAM per the project's own perf rules) would run it at CPU speeds where a full
general-benchmark suite is a multi-week job. So:

- **the donors §8.A promotes (30B sparse-MoE) cannot have their S4 baseline measured** under S1;
- **the donors whose baseline can be measured (≤ ~7B at fp16 on a single T4, or ≤ ~13B 4-bit) are
  exactly the ones finding #14 shows fail 10 tok/s** once converted.

Composed with #14 and #15, the *actually admissible* donor window under the full sealed set is roughly
**1-3B, converted via route (ii)**. That is a legitimate research program; it is not the program §1
describes ("1B-100B class"). The document should say so at stage 0 rather than discovering it at
stage 5.

**Secondary note on the same gate.** S4 correctly guards against the low-baseline loophole
("A low-performing donor cannot pass only because it retains a high fraction of a low baseline") by
mandating chance-normalized retention. That is the right instrument, but it has a numerical failure
mode the document does not flag: `retention = (converted − chance) / (donor − chance)` becomes
ill-conditioned as `donor → chance`, so for weak donors on hard benchmarks the retention-% is unstable
in exactly the regime the sealed set forces the Adapter into. §8.N.4's σ-per-metric requirement must be
stated *on the normalized quantity*, not on the raw score, or the 90 %/80 % gates will be read against
noise.

---

## 19. FLAG — §2.2 "Ternary shrinks bytes, not this floor" is false for the dominant component

**Quoted, §2.2:** "**Ternary shrinks bytes, not this floor.** A design that fixes bandwidth and ignores
the per-position floor has fixed nothing".

**Contradicting repo evidence, `docs/PHASE64_BUDGET.md:9`:** "proj-GEMV at t6 = 11 MB / 272 µs ≈
**40 GB/s = the aggregate bandwidth ceiling** — the fp32 projections already run at aggregate BW when
threaded".

Check: `11e6 B / 272e-6 s = 40.4 GB/s` ✔. A component running at the machine's aggregate memory ceiling
is bytes-bound by construction; cutting its bytes 8× (fp32 → 0.5 B/weight) cuts its time ~8×. proj-GEMV
is **51.6 %** of the t1 floor and **45 %** of the t6 floor. So ternary shrinks roughly half of "this
floor", not none of it. The genuinely irreducible parts are the scan recurrence (169/46 µs, elementwise)
and norms/glue — together 13-14 % at t1.

The sentence's *intent* — don't assume bandwidth fixes are sufficient — is right and worth keeping. Its
*mechanism claim* is wrong and, per finding #14, it is the specific sentence that hides the binding
constraint on the whole mandate. Per §6 law 10 ("deflate your own claims first") and §11's law
("a mechanism claim must be derived or measured, never asserted because it sounds right"), it needs
rewriting to: *"ternary shrinks the projection and MLP terms; the scan recurrence and glue (~13-14 % at
t1) are elementwise and do not shrink. The projections are held at fp32 by a quality decision (D9), not
by a compute bound — that decision is therefore a speed decision at donor scale and must be re-priced."*

---

## 20. FLAG — smaller items, each with its recomputation

**#10 — §5 uses 42 GB/s where the repo's own model prices streamed organs at 34-40.**
`PHASE64_BUDGET.md:32` prices the fully-streamed proj floor at **[34-40 GB/s]** and warns "cache-region
rates are upper bounds for the integrated engine — streamed experts pollute L3, pushing real rates
toward the streaming tail". The §5 table's 42 GB/s is the *cold-stream ceiling* (40-44), not the
integrated streamed floor. Re-running §5 at 36 GB/s (the repo's stated asymptote, `:15` "asymptote
≈ 34-36 GB/s"):

| donor | doc tok/s @42 | recomputed @36 | Δ |
|---|---|---|---|
| 30B/3B | 28 | 24.0 | −14 % |
| 120B/5B | 17 | 14.4 | −15 % |
| **8B dense** | **10.5** | **9.0** | **−14 %, crosses below the sealed 10 tok/s gate** |
| 70B | 1.2 | 1.03 | −14 % |

The row that matters — the only one within a factor of 1 of the gate — **flips from pass to fail under
the repo's own rate.** Fix the rate or add the sensitivity row.

**#13 — the head arithmetic assumes tied embeddings, silently.** "the embedding/head matrix alone is
~311M parameters" is one matrix. Untied input+output embeddings (common: Llama-3.x untied, Qwen3
untied at most sizes) is **622M params = 2.49 GB fp32**, and while only the output half is read per
token, both halves are charged against the S2 RAM inventory. §8.H's list of candidate mathematics leads
with "tied input/output embeddings", which is a *transformation* the donor may not permit token-for-token
under the SEALED tokenizer contract. State the tied assumption or double the footprint line.

**#14 — the "120B" example is outside the mandate's own donor range.** §1 seals the problem as
"1B-100B class" and §8.A repeats it; §5's footprint headline is "**120B** × 0.5 B/weight = 60 GB" and
§5's table row 2 is "~120B total". Either widen §1 or re-anchor §5 to 100B (100B × 0.5 = 50 GB, leaving
14 GB — which materially changes the "not yet a SKU-B fit" conclusion the sentence draws).

**#15 — "leaving only 4 GB in SKU-B" is unit-dependent.** 64 GB decimal − 60 GB = 4 GB ✔; 64 GiB
(68.72e9 B) − 60e9 B = **8.7 GB**, a 2.2× different residual. The document draws a qualitative
conclusion ("so it is not yet a SKU-B fit") from a figure that changes by 2.2× under an undeclared
convention.

**#17 — §8.F quotes a single-CCX budget against an aggregate keystone.** The bullet opens "the active
per-token slice must fit the **aggregate LLC** at the operating thread count (reference: **2 × 16 MB**
at 6 threads … )" and closes "At 0.5 B/weight that is ~24-32M active ternary params for a **single-CCX
16 MB** budget." The two halves differ by 2× (aggregate would be ~48-64M). Both are defensible reads of
the repo (`SIZING.md:14` uses ≤16 MB; the t6 r(size) curve at `PHASE64_BUDGET.md` §1b shows ~185 GB/s
holding to 16 MB and breaking from 24 MB), but the document must pick one, because §8.F.1 asks the
Adapter to test "does it fit?" against it.

**#18 — "matched total params" covers one of the two comparisons.** "Granular MoE beats dense at
matched total params. E32×h128 top-8 measured BPB 0.8589 vs dense-1024 0.8799 and dense-4096 0.8674."
E32×h128 = 4096 total hidden ⇒ dense-**4096** is the matched-*total* arm (0.8589 vs 0.8674, Δ 0.0085 =
1.7 σ_seed); dense-**1024** is the matched-*active* arm (top-8 × 128 = 1024) (0.8589 vs 0.8799,
Δ 0.021 = 4.2 σ_seed). Both are real results; the caption applies to only one. Given σ_seed = 0.005 and
these being single-seed, the matched-total delta is **1.7 σ** — worth flagging as such wherever it is
quoted as a design premise, which it is (§8.E.3).

**#19 — §2.4's constant enumeration omits `AQ 63`.** `engine.c:62` `#define AQ 63` is the activation
int8 clip used by `quant_i8` (`:167-169`, per-token scale, `amax/63`). It is a compile-time constant
inside the quoted `:51-75` range, it is not in §2.4's list, and §8.G.5 then raises exactly this organ
("the LUT path takes int8 activations (per-token scale …). Donor activations have known outlier
channels at scale. Per-group activation scales … are **not implemented**"). An enumeration meant to
scope the generalization work should not omit a constant that a later axis names as a likely required
work item.

**#21 — no unit convention is declared.** Findings #5, #6b, #15 and the whole §5 table each resolve
differently under decimal vs binary units, and finding #6b traces a published project figure (4.2 GB/s)
to a KiB/KB mix inside the repo itself. One sentence in §0 fixes this permanently.

---

# Numbers I could not check, and why

1. **The 176 → 848 tok/s ladder, the 4.8× factor, and the 701.7 MoE figure.** Sourced to
   `docs/ENGINE_PLAN.md:3`, but those are run outputs, not derivations. `176 × 4.8 = 844.8 ≈ 848` ✔
   internally, and 848/176 = 4.818 ✔ — the *ratio* checks; the *measurements* would need re-running the
   Phase-60 stage engines on the reference machine, which is out of scope for a read-only audit and
   requires the Owner's hardware.
2. **The r(size) curve and the DRAM cold-stream ceiling.** Transcribed verbatim and correctly from
   `PHASE64_BUDGET.md` §1b(a) and `:5`. Whether those microbenchmarks were correctly instrumented is a
   Phase-64.0/64.1b apparatus question; I verified transcription, not measurement. Note that finding #6c
   gives an independent reason to distrust one of them *as applied to the engine*.
3. **The ρ-law "≈14× worst case, ~2× in-cache, ~4.3× in DRAM".** Probe-3 numbers, no primary artefact
   inspected in this pass. The derived "~2.5× at 48 KB granularity" is corroborated at
   `PHASE64_BUDGET.md` §1b corollary (ii).
4. **All BPB deltas (+0.028, +0.0006, +0.013 ± 0.005, +0.018-0.022, σ_seed = 0.005).** Verified as
   correctly *cited* (finding #11). Their validity is a training-run question requiring GPU time.
5. **"~2× perplexity at 7B-70B in at least one published study"** (§8.C). The document itself labels
   this "Verify this figure yourself before building on it" and names no source — correctly marked,
   uncheckable here, and it is the Researcher's item.
6. **"86-92 %" vs "83-88 %" in-place predictability.** Both exist in the repo (see #11). Which one the
   document means is a question for the Architect, not a defect I can resolve from the files.
7. **Donor geometries in finding #14/#15** (D, L, n_kv, d_h for Llama-3.x, Qwen2.5/3, Mistral, Phi-3).
   These are from public model configs, not from this repo, and I could not verify them against
   downloaded artefacts in a read-only offline pass. **They are the load-bearing inputs to findings #14
   and #15 and should be re-derived from the actual `config.json` of each candidate before those
   findings are acted on.** The *shape* of the conclusion is robust to ±20 % on any single geometry —
   the margins are 3.9× to 17× — but the borderline rows (Qwen2.5-3B at 9.5 tok/s) are not.
8. **KV geometry in finding #17** — same caveat as (7).
9. **`archive/benchmarks/phase60_stage_engines/` contents.** Directory confirmed to exist; I did not
   verify that the e1-e4 archival parity oracles inside it still build or still reproduce parity.
10. **Whether `graphify query` is currently functional.** §12 instructs "use `graphify query
    "<question>"` before grepping"; `graphify-out/graph.json` exists, but I did not invoke the tool
    (running it is a state-changing operation on a shared artefact and outside a read-only mandate).

---

# Overall verdict

**BLOCK.** The document must not be issued to the Adapter in this revision.

Class B is clean: **every** repo reference in §2.1, §2.4 and §12 checks out at the named line, and every
quality constant is correctly sourced. The document's discipline about *citation* is real and it is
better than most of what it is auditing.

Class A is mostly clean but has one structural error and one mis-described mechanism. The §5 table, the
head figures, the SKU ceilings and the expert kernel time all reproduce exactly. But **§2.2's headline
compute floor is the sum of three of its own six organs** (finding #1, corrected: 1.283 ms dense /
1.543 ms MoE at t1), it contradicts §4's own percentages, and the packing section mis-describes the
mechanism in a way that hides a free 2× (finding #7).

Class C is where the document fails. Three findings compound:

- **#14:** the sealed 10 tok/s gate and the sealed fp32-projection decision are jointly unsatisfiable
  above ~2B parameters, because the projection term grows as D² and §5 explicitly omits it.
- **#15:** consequently **SKU-B is an empty SKU** — no donor in 1-100B both needs its 64 GB envelope
  and reaches its 10 tok/s gate, and trit-pack does not rescue it. This is an §1.1 AMENDMENT REQUEST
  available today from desk arithmetic.
- **#13:** the eligibility ranking the whole shortlist is built on (30B MoE > 8B dense) inverts under
  the engine's own measured integrated rate, and §5 says so on the page before §8.A asserts it.

Taken together, the sealed set does not describe the search space §1 describes. Its actual feasible
region is roughly **1-3B donors converted with attention retained on the majority of layers** — i.e.
§4's route (ii), the route the document frames as a concession requiring an S3 amendment. **The sealed
gates have already chosen the answer to §4's "first substantial job", by arithmetic, before any
measurement.** An Adapter that follows §9 faithfully will spend stages −1 through 4b before discovering
this at stage 5.

**Required before issue:**
1. Correct #1 (compute floor total) and propagate 1.283 ms into §4.
2. Add the proj-GEMV D² term to §5's model and re-run §5's ranking against it (#14).
3. Take #15 to the Owner as an amendment request on §S2/§S4, with the three named options.
4. Retract or condition §8.A's MoE-over-dense ranking (#13).
5. Split §9 stage 1 into desk-provisional and measured-final (#16).
6. Define "native 128K" (#17).
7. Fix the packing description and the 2.4-2.5× hedge (#7).
8. Declare a unit convention (#21).

Nothing in this audit disputes the *project*. The engine is real, the measurements are real and
carefully recorded, and the document's citation discipline is genuinely good. What it does dispute is
that the sealed gate set describes a reachable target — and per §13 item 2, that is offered as the
contribution the document asked for.
