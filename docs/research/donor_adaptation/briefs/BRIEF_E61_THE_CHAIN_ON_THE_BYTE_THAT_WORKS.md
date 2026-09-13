# E61 — the only kernel that works is the only kernel still running a single accumulator

**Pre-registered. Pushed before any runner exists and before any measurement is taken.**
Opened by `probes/E60_THE_RUNG_THAT_WAS_NEVER_BUILT.md` §7 item 1, by `SPEED_LEDGER.md` §59.2's
priced corollary, and by a debt first written down on **2026-09-07** and carried forward, unpaid,
five times.

---

## 1. Why this exists

E60 published a curve: this box has **three bandwidths, not one** — 37.7–38.8 / 31.7–33.1 /
21.5–22.7 GB/s at 4 / 1 / 0.5 bytes per weight — and `SPEED_LEDGER.md` §59.2 promoted those three
numbers to *"the three operative denominators on this box"*.

**E61 asks whether the middle one is a bandwidth at all.**

The 1 B/weight arm is not an ordinary arm. It is the only artefact this programme has ever
produced that is simultaneously **fast** (63.87 tok/s at 0.5 B, above the 50 tok/s bar) and
**faithful** (dBPB `+0.000066`, 1.3% of `σ_seed`). Every speed number that matters from here runs
through it. And it is the one weight kernel in `donor_engine.c` that E8 never touched.

## 2. Checked, not assumed

Searched by artefact name — `mvacc`, `--mvacc`, `single accumulator`, `non-packed` — across
`docs/` and every `*/results/`. **Every brief and probe the search returned was opened.** What
each one actually said:

| where | what it says | bearing on E61 |
|---|---|---|
| `probes/E8_MATVEC_DEPENDENCY_CHAIN.md` §9 item 3 | *"The non-packed ternary path still has a single accumulator. `--mvacc` was applied to the packed and fp32 kernels only. **Not on the measured path, so not urgent**, but it is a known remaining chain."* | The debt, stated correctly, and correctly deprioritised — **on 2026-09-07, when the branch was dead**. E60 made it the measured path on 2026-09-14. |
| `probes/E9_...md`, `probes/E10_...md`, `probes/E11_...md`, `probes/E13_...md` | the same sentence, carried forward verbatim four more times | Chain of custody. Nobody paid it, and nobody claimed to. |
| `BRIEF_E11_LUT_KERNEL_CEILING.md` §1 / `probes/E11_...md` §1 | *"that packed arm still had E8's single accumulator, and `matvec_lut` already used four, so E8 never touched it"* | The LUT kernel is **already** multi-accumulator. It is not an arm here: `--lut`/`--lutblk` require `--quant packed` (`donor_engine.c:1553`) and are structurally excluded from the int8 rung. |
| `BRIEF_E13_BLOCKED_TILE_MAJOR.md` | `packed (--mvacc 4)` ≈36 vector instructions per 64 weights vs `matvec_lut` ≈12 | Instruction counts exist for packed and for lut. **None exists for the fallback.** |
| `BRIEF_E50_TURN_THE_FAST_KERNEL_ON_BY_DEFAULT.md` §2 | `donor_engine.c:76`: *"Defaulted rather than left as a flag because a flag a runner must remember to pass is exactly the defect of s9"* | See §4 — that defence was defeated anyway, in a new way. |
| `BRIEF_E55_...md` A, `BRIEF_E57_...md` A, `probes/E60_...md` | each prints `CONFIG … mvacc=4 …` on every cell | And on E60's `I8` and `T1` cells **that line was false**. §4. |
| `results/e53_fexp.json`, `results/e55_table.json`, `e53_c01.log`, `e55_run.log` | the only results artefacts that mention `mvacc` at all; all record `mvacc=4`, all on packed or fp32 arms | **No measurement of the fallback branch at more than one accumulator exists anywhere in the repo.** |

The claim of a gap is made **after** the search, not before it.

## 3. The mechanism, verified in the emitted assembly before it is claimed

Not an assumption. `clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp -S`, current source, today.
`matvec_sel`'s third outlined region, hot loop `.LBB17_77` — clang unrolls it 4×, and every FMA
consumes the previous one's result:

```
vpmovsxbd    -24(%r12,%rbp), %ymm1
vcvtdq2ps    %ymm1, %ymm1
vfmadd132ps    (%r9,%rbp,4), %ymm0, %ymm1   # ymm1 = ymm1*mem + ymm0
vpmovsxbd    -16(%r12,%rbp), %ymm0
vcvtdq2ps    %ymm0, %ymm0
vfmadd132ps  32(%r9,%rbp,4), %ymm1, %ymm0   # ymm0 = ymm0*mem + ymm1
vpmovsxbd     -8(%r12,%rbp), %ymm1
vcvtdq2ps    %ymm1, %ymm1
vfmadd132ps  64(%r9,%rbp,4), %ymm0, %ymm1   # ymm1 = ymm1*mem + ymm0
vpmovsxbd      (%r12,%rbp), %ymm0
vcvtdq2ps    %ymm0, %ymm0
vfmadd132ps  96(%r9,%rbp,4), %ymm1, %ymm0   # ymm0 = ymm0*mem + ymm1
```

`ymm0 → ymm1 → ymm0 → ymm1 → ymm0`: **one unbroken chain, 4 FMAs, 32 weight bytes.** The same
signature E8 read on the packed kernel, in the same function, one branch further down.

**The compiler cannot fix this, by our own rule.** Breaking the chain re-associates a float sum;
`-ffast-math` is forbidden (Phase 35). Only a source change alters it.

### 3.1 What the chain costs, arithmetically

Zen 2, 6 cores, `--threads 6`, 3.793 GHz base / ~4.2 GHz all-core upper. `vfmadd*ps ymm`:
**latency 5 cycles**, throughput 2/cycle. A serial chain runs at latency.

| branch | weight bytes per **serial** FMA | chains | cycles/FMA | B/cycle/thread | ceiling @3.793 | @4.2 |
|---|---|---|---|---|---|---|
| fp32, `--mvacc 1` | 32 | 1 | 5 | 6.4 | 145.6 GB/s | 161.3 |
| fp32, `--mvacc 4` (**default**) | 32 | 4 | 1.25 | 25.6 | 582 GB/s | 645 |
| packed, `--mvacc 1` | 4 | 1 | 5 | 0.8 | 18.2 GB/s | 20.2 |
| packed, `--mvacc 4` (**default**) | 4 | 4 | 1.25 | 3.2 | 72.8 GB/s | 80.6 |
| **int8 fallback — no other mode exists** | **8** | **1** | **5** | **1.6** | **36.4 GB/s** | **40.3** |

### 3.2 The discriminating observation

E60 §3's measured achieved bandwidths, set against **each kernel's own** chain ceiling:

| arm | achieved GB/s | its own chain ceiling @3.793 | fraction of it |
|---|---|---|---|
| `05b F32` | 37.70 | 582 | 6.5% |
| `15b F32` | 38.78 | 582 | 6.7% |
| `05b PACKED` | 21.48 | 72.8 | 29.5% |
| `15b PACKED` | 22.74 | 72.8 | 31.2% |
| **`05b I8`** | **31.67** | **36.4** | **87.0%** |
| **`15b I8`** | **33.09** | **36.4** | **90.9%** |

**Every other weight kernel in this engine runs at under a third of its latency wall. The int8
branch runs at 87–91% of its.** That is the whole hypothesis in one table, and it is why the
middle denominator of §59.2 is suspect: 31.7–33.1 GB/s sits within 10–13% of a number derived
from instruction latency and nothing else.

### 3.3 Three ways this is still wrong

1. **Sustained all-core AVX2 clock is below 3.793 GHz**, pushing the true ceiling under the
   measurements, in which case something else already binds and 87–91% is a coincidence of my
   clock assumption.
2. **The front end binds, not the chain.** Per 8 weight bytes the loop issues one
   `vpmovsxbd`(mem), one `vcvtdq2ps`, one `vfmadd132ps`(mem) — roughly 3–4 uops and 2 loads. At
   even 2 cycles/iteration that is 4 B/cycle/thread = 91 GB/s, 2.5× above the chain ceiling, so
   this is the *weaker* of the two explanations — but arithmetic alone does not exclude it.
3. **Memory genuinely binds at ~32 GB/s for a 1 B/weight stream.** The fp32 arm reaches 37.7–38.8
   on the same box, but with a 4× coarser weight stream; a finer stream is not obliged to reach
   the same rate.

**§6's two controls distinguish these, and both were already measured — by an experiment that did
not know E61 would need them.**

## 4. The config line was true about the flag and false about the kernel

`donor_engine.c:76` defaults `g_mvacc=4` rather than leaving it a flag, with the reason written in
the source: *"a flag a runner must remember to pass is exactly the defect of s9"*. `CONFIG` then
prints `mvacc=4` on every cell — and E55, E57 and E60 all recorded it.

**On E60's `I8` and `T1` cells that line was false.** The branch never reads `g_mvacc`. The runner
did not forget the flag: the flag was defaulted, printed, and recorded in three probes, and
**silently not honoured by the kernel it named**.

This is [[feedback_config_must_appear_in_output]] in a third form — not a config that fails to
print, but a config that prints a value the code path does not implement. The defence against s9
(*default it so nobody forgets*) does not cover this and never could.

**E60's numbers are not retracted.** The branch behaved as `mvacc=1` in all 9 reps of all 4 of its
cells, consistently; the measurement is sound and the label on it was wrong. What is withdrawn is
the *reading* of §59.2's middle denominator as a property of the box.

## 5. What is built

**One change, one loop.** `matvec_sel`'s final fallback (the `quant==1`/`quant==5` branch) gains
the `MA = g_mvacc` structure already present, twice, in the same function — the fp32 branch's
shape exactly, with the `vpmovsxbd`+`vcvtdq2ps` pair inserted:

- **`--mvacc 1` is a byte-for-byte copy of the existing loop.** Gated on sha256 (`G-E61a`), so
  every number E60 published stays citable and reproducible on the new binary.
- `--mvacc 2` / `4` partition the sum into 2 / 4 independent chains. **Same bytes read, same FMA
  count, same order within a chain** — only the partition of the sum changes.
- **Not bit-identical, and not claimed to be.** Re-partitioning a float sum changes rounding, so
  this is gated on parity and fidelity (`G-E61d`, `G-E61e`), never on sha256 — Phase 60's law.

Nothing else is touched. Packed stays the default format; `--attn`, `--fexp`, the contention
witness and the LUT kernels are untouched.

Binary: `donor_engine_e61.exe`, from the frozen line
`clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine_e61.exe -lm`.

## 6. Arms and gates — thresholds fixed here, before anything runs

Apparatus is E60's, unchanged: `--threads 6`, `--bench 160`, **9 reps, arms interleaved within
each rep**, occupancy witness on every cell, `OCC_BAR = 4.39`. Weight files are E60's, byte for
byte. Fidelity slice is the standard one: 24×512, **51,870 scored bytes, 12,264 predicted
positions**, chance BPB 4.069819.

### G-E61a — the instrumentation is inert

| test | pass |
|---|---|
| `--logits` sha256, `donor_engine_e61.exe --mvacc 1` vs `donor_engine_e60.exe`, on `05b_i8h` and `15b_i8h` | **byte-identical, both** |

**FAIL ⇒ the change is rejected as built.** E8's G-Z0a precedent.

### G-E61b — the PLANTED POSITIVE must fire

`05b`, **packed**, `--mvacc 4` ÷ `--mvacc 1`, paired median over 9 interleaved reps.

| result | meaning |
|---|---|
| **≥ 1.10** | **FIRES** — the flag works and the sweep can see a chain effect it is known to contain |
| < 1.10 | **the instrument is dead and every null below it is void** |

E8's G-Z3 read **1.2301** on this arm. The reproduction is registered **directionally, not
numerically**: E8 ran before `--attn avx4` became the default (E49/E50), so the token-level ratio
is not the same quantity today. Demanding `1.23 ± ε` would be a gate that cannot be answered —
E4's malformed-gate precedent. `≥ 1.10` can be.

### G-E61c — the PLANTED NULL must stay null

`05b`, **fp32**, `--mvacc 4` ÷ `--mvacc 1`, same sweep.

| result | meaning |
|---|---|
| **0.97 – 1.05** | as predicted; fp32's chain has 15× slack and must buy nothing |
| **≥ 1.10** | **§3's mechanism is wrong**, whatever the int8 arm does — the effect would then be something `--mvacc` merely correlates with (code layout, unrolling, the loop tail), and **the int8 result may not be attributed to the dependency chain** |

E8's G-Z5 read **1.0029**. Same wording, same threshold, deliberately.

**Both controls run in the same sweep, on the same binary, on the same box, interleaved with the
treatment.** The planted-control law is satisfied in **both** directions at once — a known
positive that must fire and a known null that must not.

### G-E61d — parity, admissibility before speed

`--logits`, rel L2 of `--mvacc 4` against `--mvacc 1`, on `05b_i8h` **and** `15b_i8h`.

| pass |
|---|
| **≤ 1.0e-05** |

**FAIL ⇒ the arm is rejected whatever its speed.** E8's G-Z2a threshold, unchanged.

### G-E61e — fidelity, on the metric E60 registered as its replacement

**This gate defines the metric E60 §7 item 5 owed, and it is registered here, before the run that
uses it.** E60's `G-E60e` was declared *defective*: a greedy counter over 5×32 positions measures
tie-breaks, not fidelity. The replacement, stated once:

> **Per-position top-1 agreement under teacher forcing, over the 12,264 predicted positions of the
> standard slice**, reported with `|dBPB|` from the same pass.

Measured for `I8 --mvacc 4` against `I8 --mvacc 1`, at both sizes.

| clause | pass |
|---|---|
| top-1 agreement | **≥ 99.0%** |
| \|dBPB\| | **≤ 0.0005** (σ_seed ÷ 10) |

**`G-E61e0` — the metric's own planted control, because ~100% is a CEILING and a counter on its
ceiling cannot show damage either** (the other half of [[feedback_gate_is_not_a_progress_meter]]):

| pair | must read |
|---|---|
| `F32` vs `F32` | **100.000%** — the instrument agrees with itself |
| `PACKED` vs `F32` | **< 50%** — the instrument can read a known-broken arm as broken |

**If `G-E61e0` does not bracket, `G-E61e` is VOID** and the fidelity question is reported as
unanswered, not as passed.

### G-E61f — the rate, the headline

`I8`, `--mvacc 4` ÷ `--mvacc 1`, paired median over 9 interleaved reps, `05b` and `15b`.

| paired median ratio | verdict |
|---|---|
| **≥ 1.35** | **OVERSHOOT** — above §3.1's own arithmetic; the mechanism is wrong in the other direction and **must not be published as the explanation**, whatever the speed is worth |
| **1.12 – 1.35** | **CHAIN-CONFIRMED-ON-THE-BYTE** — §59.2's middle denominator was a kernel, not a box |
| 1.06 – 1.12 | **UNDECIDED** — real, but far under the derivation; reported as such and not dressed up |
| **≤ 1.06** | **REFUTED** — the branch was already memory-bound; 31.7–33.1 GB/s **is** a property of this box at 1 B/weight and §59.2 stands as written |

**1.06 clears the dispersion measured on this exact axis**, as the standing rule requires: E60's
`I8` CI half-widths were **1.76%** (`05b`: 63.87 on 61.76–64.01) and **0.35%** (`15b`: 21.39 on
21.35–21.50). 6% is ≥3.4× the worse of the two.

### G-E61g — the value-dependence diagnostic (descriptive, no verdict attached)

E60 §3.1 left **1.98 tok/s** unexplained between `I8` (63.87) and `T1` (61.89) at **identical
bytes through an identical kernel** — the reading that refuted E60's own *"speed is a function of
bytes, not values"*. `T1` runs at both `--mvacc` settings alongside `I8`. Reported as a table;
it pays part of E60 §7 item 2 and decides nothing on its own. **No `-ffast-math`, in any arm, for
any reason** (Phase 35).

## 7. Predictions, fixed now, each with what falsifies it

| # | prediction | falsified by |
|---|---|---|
| 1 | `G-E61a` byte-identical at both sizes | any sha256 mismatch |
| 2 | `G-E61b` fires; packed `05b` reads **1.15–1.30** | < 1.10 (instrument dead) or > 1.40 |
| 3 | `G-E61c` null; fp32 `05b` reads **0.98–1.04** | ≥ 1.10 — and then prediction 5 is unattributable |
| 4 | `G-E61d` rel L2 in **1e-07 … 5e-06** — at or under E8's packed 3.3e-06, because int8 codes are exactly representable in fp32 and only the summation order moves | > 1.0e-05 |
| 5 | **`G-E61f` reads 1.12–1.22 at `05b` and 1.10–1.20 at `15b`.** The ceiling goes to 145.6 GB/s, so the binder becomes the streaming rate the fp32 arm demonstrates on this box (37.70 / 38.78 GB/s): 37.70÷31.67 = **1.190** and 38.78÷33.09 = **1.172**. In tok/s, `05b` **71.6–77.9** and `15b` **23.5–25.7** | anything outside those bands; ≤1.06 refutes §3 outright |
| 6 | `G-E61e` top-1 ≥ **99.7%** and \|dBPB\| ≤ **1e-04** at both sizes | below 99.0%, or above 5e-04 |
| 7 | **`G-E61g`: the `I8`−`T1` gap does NOT close.** If it is value-dependence (subnormals in a broken model) it survives the fix roughly in proportion; if it had been the chain it would vanish | the gap closing to < 0.5 tok/s at `--mvacc 4` |

**Prediction 5 is the one that can embarrass this brief, and it embarrassed its predecessor
already**: E8 predicted 1.5–1.7× and measured 1.349×, because removing the chain did not hand the
kernel to memory — it handed it to a second kernel-shaped limit at 22.5 GB/s. **That failure mode
is live here.** The next binder below 145.6 GB/s may again not be memory.

## 8. What E61 cannot claim, whatever it reads

- **Nothing about quality.** `G-E61e` compares an arm to itself. E60's `dBPB +0.000066 / +0.001252`
  against fp32 is the quality statement, and E61 does not revisit it.
- **Nothing about the byte ladder's verdict.** `THE-CLIFF-IS-BELOW-ONE-BYTE` stands. This is the
  speed of the 1 B rung, not its position.
- **Nothing measured at 10 B.** E60 §7 item 4 stays owed. If `G-E61f` confirms, the 10 B number is
  re-stated **as a desk model**, on E36's `A10B-K3` cell, with **noise weights** (E36 §0).
- **Nothing about the goal being met.** A confirmed E61 puts `05b` near 76 tok/s and `15b` near 25.
  The target shape is 10 B dense-equivalent, where the faithful arm is desk-modelled at
  34–36 tok/s = 68–71% of the bar. **A ×1.19 on a 0.5 B model is not a 10 B model at 50 tok/s.**
- **No T4 time is asked for or implied.** CPU-only, on the reference box.

## 9. Owed after E61, whatever it reads

1. `CONFIG` must stop printing a per-process flag as if it were a per-kernel fact — print the
   effective accumulator count for the format actually loaded (§4).
2. The 10 B cell at 1 B/weight, **measured**, not desk-modelled (E60 §7 item 4).
3. A third scale for the 0.5 B → 1.5 B dBPB growth segment (E60 §7 item 3).
4. `C/T` measured on the donor engine — decides whether the weights-once chassis in
   `benchmarks/phase60/engine.c` is worth porting (E60 §7 item 6).
5. Channel-granularity mixed precision: 1% of weights at 8 bits costs +1% of bytes (E60 §7 item 7).
6. `G-E55a2`'s replacement interval gate; E56's `OCC_BAR` zero point; E42's void control.
