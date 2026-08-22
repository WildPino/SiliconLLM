# Controller audit — the integrated-MLP measurement and the "Qwen2.5-1.5B clears ≥10 tok/s" claim

**Role:** Controller, independent adversarial review under §7.1 two-key. **Branch** `research/donor-adaptation`.
**Date** 2026-08-21. **READ-ONLY on code.** Nothing in `benchmarks/` or `docs/` was edited; this file is the
only artefact written (`git status` unchanged except for it).

**Subjects:** `docs/research/donor_adaptation/probes/DONOR_STAGE1_ARITHMETIC.md` rev C, `docs/research/DONOR_PROJ_RATE.md` §10,
`benchmarks/donor_adaptation/donor_inventory.py`, `benchmarks/donor_adaptation/gemv_donor_bench.c`.
**Priors:** `CONTROLLER_PROJRATE_AUDIT.md` (C10–C13). That audit confirmed the Builder's *measurement* and
refuted its *inference*; per the brief I have weighted the inference at least as heavily here.

Everything below was rebuilt and re-run by me, in my own binaries, in my own session:

```
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp benchmarks/donor_adaptation/gemv_donor_bench.c -o bin/ctrl3_donor_bench.exe -lm
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp benchmarks/phase60/engine.c              -o bin/ctrl3_engine.exe       -lm
export OMP_PROC_BIND=close OMP_PLACES=cores
```
clang 21.1.8 x86_64-w64-windows-gnu, AMD Ryzen 5 3600X 6C/12T, 80 GB, Windows 11. Machine quiet.

---

## VERDICT SUMMARY

| # | claim under attack | verdict |
|---|---|---|
| F1 | headline 11.97 [10.67..12.68] at n_att=14, fp16 KV, windowed | **PASS — reproduces to the last digit** |
| F2 | interval "fully-correlated, central = midpoint in *time*" | **FLAG — half true; central is not the time-midpoint** |
| F3 | the PASS no longer lives in one cell | **FLAG — it lives in the *windowed* column, declared** |
| **F4** | **converted layers are charged the donor's attention projection bytes** | **BLOCK — undeclared, and it flips the fp16 headline** |
| F5 | integrated MLP = 27.31 ± 0.46 ms at donor D | **PASS — I measure 26.82 ms, 1.8% faster** |
| F6 | known-positive is only ±13.5%, "not the ±5% the proj check achieved" | **PASS, upgraded — the gap is −3.4%, not +13.5%** |
| F7 | published `LUT-MLP 313→207` is the E3.5 `--skip on --exp fast` config | **PASS — confirmed three independent ways** |
| F8 | the systematic 15–25% shortfall vs §1 | **FLAG — uniform across 5 organs, so a global clock factor** |
| F9 | `DENSE_LUT_GBS = 11.398` doubly misattributed, 1.86× | **PASS — both halves verified in `engine.c`** |
| F10 | rebuilt mlpctl controls fire | **PASS — every ladder reproduces verbatim** |
| F11 | no planted control covers any rev-C addition | **FLAG — the terms the verdict now turns on are unguarded** |
| F12 | full-attention desk model, and the "≤2 of 28" conclusion | **FLAG — crossing moves 2 to 11 on the unmeasured constant** |
| F13 | unconverted = 4.70 tok/s, FAIL | **PASS on the FAIL, FLAG on the number** |
| F14 | KV scales with retained attention layers | **PASS — correct and consistently applied** |
| F15 | between-session dispersion of the load-bearing constants | **FLAG — third session, third answer (6.9% spread)** |
| F16 | document/manifest staleness | **FLAG — §5/§7/§8 still rev B and contradict §0** |

**The measurements are sound and, where I can check them, better than the Builder reports. The headline
configuration's verdict still turns on a term the model does not contain.**

---

## F1 — PASS. The headline reproduces exactly, and I re-derived it by hand.

`python donor_inventory.py curve`, my run, n_att=14, ctx=32768, fp16 KV, SWA-128:
**11.97 tok/s [10.67 .. 12.68]** — digit-for-digit the Builder's figure. Lower bound clears 10 by 6.65%.

Re-derived term by term from the config, not from the tool (all µs, `slow / fast / central`):

| term | bytes or form | slow | fast | central | my check |
|---|---|---|---|---|---|
| proj+head+norms | 616 562 688 + 933 494 784 + 579 584 = **1 550 637 056 B** | 41 483.1 | 38 717.5 | 38 892.3 | ok: /37.38, /40.05, /39.87 GB/s |
| ternary MLP | 578 027 520 + 2 179 072 = **580 206 592 B** | 28 261.4 | 26 409.0 | 27 335.2 | ok: /20.53, /21.97; central = time-midpoint |
| KV | 14 × 512 × 2 B × 32768 = **469 762 048 B** | 11 744.1 | 10 676.4 | 11 184.8 | ok: /40, /44, /42 GB/s |
| scan-recur | 46 µs·(2·1536·96·14)/(512·96·6) = 46×14 | 1 288.0 | 322.0 | 644.0 | ok |
| SWA | 65 µs·(1536/256)·14 = 65×6×14 | 10 920.0 | 2 730.0 | 5 460.0 | ok |
| glue | 7 µs·28/6 | 65.3 | 16.3 | 32.7 | ok |
| **total** | | **93 761.9** | **78 871.3** | **83 549.0** | **10.665 / 12.679 / 11.969** |

Byte accounting is exact against `CONTROLLER_STAGE1_AUDIT.md` F1 (my `fullstack` run prints
`1550057472 B -> EXACT MATCH`; the extra 579 584 B is qkv-bias + rmsnorm, correctly in the fp32 organ).
The arithmetic is right. Everything below is about what the arithmetic is *of*.

## F2 — FLAG. The interval is fully-correlated as claimed; the central is not what is claimed.

*"the reported interval is the sum of slows / sum of fasts — a fully-correlated worst case, deliberately
wider than a root-sum-square"* — **TRUE**, verified in `time_model()`: `parts` are summed component-wise.

*"Central time is the midpoint of the time interval, not of the rate interval"* (§1.3) — **FALSE for the two
terms where it matters.** `_t()` honours the rule only when no `rate_central` is passed. It is passed for
proj (39.87 GB/s) and KV (42), and the full-attention branch hard-codes a central of 40 GFLOP/s.

- proj: interval in time is [41 483, 38 718]; midpoint **40 100**; central used **38 892** — 1 208 µs faster.
  Applying the doc's own declared rule moves the headline central **11.97 → 11.80**.
- full attention (F12): interval [293 601, 56 371]; midpoint **174 986**; central used **140 929**.
  Under the declared rule the unconverted figure is **4.05**, not 4.70.

Lower bounds are unaffected, so no gate verdict moves. But the stated method and the implemented method
differ, in the optimistic direction, on the two largest terms in the document.

## F3 — FLAG. It is a 2×2×2 now, and the PASS occupies the windowed half of it.

At n_att=14, ctx 32K, from my `curve`/`grid` runs:

| | 4-bit KV *(unbuilt)* | fp16 KV *(built)* |
|---|---|---|
| integrated MLP, **windowed** | 13.30 [11.77..14.11] PASS | **11.97 [10.67..12.68] PASS** |
| integrated MLP, **full attn** | 7.13 [4.53..10.38] FAIL | 6.73 [4.35..9.59] FAIL |
| kernel-pure MLP, windowed | PASS | PASS |
| kernel-pure MLP, full attn | FAIL | FAIL |

So the prior audit's structural finding survives on a new axis: **every full-attention cell fails.** The
difference from rev B is that this is now *declared* — §0.2 names windowing as a requirement — so it is a
FLAG, not a BLOCK. Two riders the document does not carry:

1. The passing object is not `Qwen2.5-1.5B`. It is a derivative in which 14 layers are SSM and the other
   14 see **128 tokens**. Whether that retains general-purpose capability is S4 / stage-−1 and is unpriced.
2. **The windowed column is internally incoherent, in the conservative direction.** A retained layer is
   charged SWA-128 *compute* and simultaneously a full **32K** KV read (11.18 ms central, 13.4% of the
   token). A window-128 layer reads 128 tokens of KV: 1.8 MB, 0.04 ms. Made coherent, the headline rises to
   **13.81** and the fp16 curve would pass at every `n_att`, dissolving the "18 of 28" budget in §0.2. The
   deliverable crossing point is therefore an artefact of a mixed pricing — favourable to the donor, but it
   should not be quoted as a design constraint.

## F4 — BLOCK. A converted layer is charged the donor's *attention* projection bytes. Nothing says so, and it flips the fp16 headline.

`time_model(ptb, ..., n_att_override=n)` never touches `ptb`. The weight stream is
`row["per_token_bytes_p61"]`, computed from the donor's **28 attention layers** (q,k,v,o), and it is held
fixed while the model simultaneously credits the conversion with:

- removing 14 layers' KV cache (`kv_per_attn_layer(row,ctx,kv) * n_att`), and
- replacing 14 layers' attention with scan-recurrence.

**The conversion is charged its benefits and none of its weights.** The 14 converted layers pay 644 µs of
scan-recurrence while carrying zero bytes of their own.

This is decidable from the repo. `benchmarks/phase60/engine.c:205-208` gives Arch-A's SSM block:
`in_proj 2·DN·D`, `conv_w DN·CONV`, `x_proj (DTR+2N)·DN`, `dt_proj DN·DTR`, `A DN·N`, `out_proj D·DN`,
with `DN = 2D`, `N = 96`, `DTR = 16`. At D=1536 that is **15 151 104 weights/layer against attention's
5 505 024 — 2.75×.** Priced under the tool's own P61/D9 map (projections fp32), 14 converted layers add
**+286 MB/token**:

| n_att = 14, ctx 32K, windowed | 4-bit KV | **fp16 KV (built)** |
|---|---|---|
| as published | 13.30 [11.77..14.11] **PASS** | **11.97 [10.67..12.68] PASS** |
| **+ Arch-A SSM projections, fp32** | 11.27 [10.06..11.86] PASS | **10.30 [9.24..10.83] STRADDLE** |
| + SSM projections **ternary**, attention projections dropped | 13.81 [12.21..14.71] PASS | 12.37 [11.03..13.16] PASS |

**On the only KV precision that is built, the headline PASS becomes a STRADDLE** — the lower bound falls to
9.24 — under an assumption the document never states.

The favourable resolution is not free either: it requires ternary *projections*, which the document's own
F3 / assumption-1 marks **UNMEASURED for projection shapes**, and which P61 rejected on quality
(+0.018–0.022 BPB, about 4σ_seed).

There *is* a defence, and the Builder should make it rather than leave the assumption silent: shape-
preserving conversion schemes (Mamba-in-Llama / MOHAWK class) re-use q,k,v,o as the SSM's input/output
projections and add only small `dt`/`A`/`D` tensors, in which case the byte count really is approximately
preserved. But then the scan-recurrence term — which is priced off Arch-A's `Dn = 2D`, `N = 96` block — is
being charged for a *different* block than the one supplying the bytes. Either way, one of the two terms is
inconsistent with the other, and the resolution decides the verdict.

**BLOCK: the converted-layer weight model must be stated and priced before this configuration may be
reported as clearing the gate.**

## F5 — PASS. I reproduce the integrated MLP measurement, slightly faster than reported.

`bin/ctrl3_donor_bench.exe mlpint --reps 3`, my build, my session:

| config | D | ffn | L | skip | t1 µs/tok | **t6 µs/tok** | cv% | **t6 GB/s** |
|---|---|---|---|---|---|---|---|---|
| sandbox | 256 | 1024 | 6 | 0 | 172.0 | **84.0** | 1.8 | 28.09 |
| sandbox | 256 | 1024 | 6 | 1 | 261.6 | **229.2** | 2.9 | 10.29 |
| **DONOR** | 1536 | 8960 | 28 | **0** | 68 078 | **26 818** | **2.5** | **21.55 / 21.63 (+scales)** |
| DONOR | 1536 | 8960 | 28 | 1 | 79 768 | 62 008 | 0.9 | 9.32 |

**26.82 ms against the Builder's 27.31 ms — 1.8% faster, in the donor's favour.** `mlp_dense_int()` is a
faithful op-for-op copy of `engine.c mlp_dense()` (read line by line: same quant, same `build_lut_t3`, same
`matvec_lut_full`, same per-row scale multiplies, same second quant/LUT before `down`, `rmsnorm` excluded
exactly as `forward_token` excludes it). Layers are allocated separately, 580 MB working set, 5.5% of
aggregate L3, and the rate sits at half the DRAM ceiling — not a warm-buffer artefact. **PASS.**

*Minor, unreported:* the donor is **SiLU**-gated; the harness computes `relu(g)·relu(u)`. The Builder is
careful that the donor cannot use dReLU *sparsity*, then measures its MLP with the dReLU *activation*. The
missing term is about 250 880 `expf` per token; vectorised, 0.25–0.5 ms, i.e. 0.3–0.6% of the token.
Immaterial, but it is the same class of substitution the whole §10 exists to correct, and should be declared.

## F6 — PASS, and the Builder's own hedge was too pessimistic. The +13.5% is a noise artefact.

The brief asks whether +13.5% is acceptable for a number carrying this much weight. **The gap is not real.**
The Builder's harness draws were 89.9 / 103.1 / 97.6 / 121.7 (cv 12.9%) against six engine draws with
sd 5.0%. Both sets were noisy; the comparison of two noisy means produced the 13.5%.

On a quiet machine, interleaved, five pairs, `bin/ctrl3_engine.exe --weights results/phase60/e1_model.bin
--timing --mlp lut --skip {on|off} --exp fast --threads 6`, LUT-MLP bucket, µs/tok:

| config | draws | mean | sd |
|---|---|---|---|
| engine `--skip off` | 90.6, 85.3, 85.7, 85.3, 88.0 | **87.0** | 2.3 (2.6%) |
| engine `--skip on` (E3.5) | 162.1, 163.1, 159.5, 161.8, 165.0 | **162.3** | 2.0 (1.2%) |
| harness `skip=0` (F5) | — | **84.0** | cv 1.8% |

**Harness 84.0 vs live engine 87.0 = −3.4%**, not +13.5% — i.e. *better* than the ±5% the projection
known-positive achieved, and in the conservative direction (the harness is not optimistic; it is 3% fast on
a 2.4 MB working set, and the donor point is 300× larger and 11× tighter).

**Does the verdict survive if the harness were nonetheless 13.5% optimistic at donor D?** I tested it
rather than argued it. Inflating the MLP term by 13.5%: n_att=14, fp16, windowed gives
**11.46 [10.25..12.13]**, lower bound still clears. So the answer to the brief's question is *yes on both
counts*: the gap is not there, and the verdict would survive it if it were. **PASS — this is the strongest
part of the work.**

## F7 — PASS. The E3.5 identification is right, confirmed three independent ways.

1. **The engine's default is E3.5.** `engine.c:893`: `int mlp_lut=1,skip=1,exp_fast=1; // default = the full
   optimized config`. A 64.0(b) decomposition run without flags *is* `--skip on --exp fast`.
2. **The harness's known-positive discriminates.** Against the published 207 µs, `skip=1` lands at
   **229.2 µs (+10.7%)** and `skip=0` at **84.0 µs (−59.4%)**. Only one of those is the published number.
3. **The live engine discriminates the same way.** E3.5 = 162.3 µs, E2 = 87.0 µs. 207 is 1.28× the former
   and 2.38× the latter.

**PASS, and it is a real finding about our own banked data**: `PHASE64_BUDGET.md` §1's `LUT-MLP dense
313→207` is a configuration-conditioned figure and the configuration was never recorded.

**Side-finding, larger than the donor point and outside this brief.** The row-skip path is *slower*, not
faster, at t6 — **1.87× slower in the live engine at claimed-identical output** (`engine.c:612-613` states
E2 and E3 are output-identical; the skip branch at `engine.c:266-269` is a plain serial `for` over
`MLP_HID` with no `OMP_PFOR`, so it does not thread). Whole-token totals from my interleaved runs:
E3.5 **476.3 µs** vs E2+fast **398.2 µs** — a **1.20× engine-wide speedup available by flipping a default
flag**. This should go to the owner independently of the donor programme; I did not verify bit-identity
myself and recommend a `--logits` parity check before acting on it.

## F8 — FLAG. The 15–25% shortfall is uniform across five organs, so it is a machine-state factor, not five findings.

The Builder reports the E3.5 reproduction and does not dwell on the gap. My five-run E3.5 means (µs/tok):

| organ | `PHASE64_BUDGET` §1 | Builder (mean 3) | **mine (mean 5, sd)** | dev |
|---|---|---|---|---|
| scan-recur | 46 | 36.5 | **36.1** (0.5) | **−21.5%** |
| proj-GEMV | 272 | 222.8 | **218.4** (2.4) | **−19.7%** |
| SWA-attn | 65 | 48.5 | **48.3** (0.8) | **−25.7%** |
| LUT-MLP | 207 | 166.3 | **162.3** (2.0) | **−21.6%** |
| head | 8 | 6.3 | **6.2** (0.1) | **−22.5%** |

**−19.7% to −25.7% across five organs with entirely different bottlenecks** (memory-bound GEMV, serial
scalar skip-MLP, elementwise scan, attention, head). That uniformity excludes an organ-specific or
kernel-specific cause and points at a **global scale factor — clock/boost state or background load at
publication time**. It is not evidence that any kernel improved.

*(My first three unpinned draws were 160.6 / 214.9 / 180.7 — one of them straddles the published 207. The
"systematic 15–25% below" reading is only visible on quiet, interleaved runs; on noisy ones it is not even
present. That is a further reason not to build an argument on the magnitude of the gap.)*

Consequence for the donor arithmetic, and it is favourable: `SCAN_US_ANCHOR = 46` and `SWA_US_ANCHOR = 65`
are taken from the *published* (slower) table while proj and MLP are priced on *today's* (faster)
measurements. The model therefore mixes two machine states, over-charging scan and SWA by about 21%.
Direction: **conservative**, worth roughly 1.3 ms of the headline. It should be declared rather than left as
an accident.

## F9 — PASS. `DENSE_LUT_GBS = 11.398` is doubly misattributed. Both halves verified from `engine.c`.

- `11.398 = 2 359 296 B / 207 µs`. The byte count is the **sandbox** stack: 3 × 256 × 1024 × 0.5 B × 6 =
  2 359 296 (codes only; the 55 296 B of per-row scales are excluded). **Wrong `D`** — trivially true.
- **Wrong path** — verified. The 207 µs is the skip-on config (F7), and that path streams *fewer* bytes than
  it is charged: `engine.c:266-269` reads `up_rm` only for rows with `gh[i] > 0` (measured by the harness:
  **2280 of 8960 rows, 25.4%**), and `matvec_lut_tileskip` (`engine.c:120`) visits only the active tiles of
  `down`. Numerator and denominator are both wrong.

**Is it true that a donor cannot use the row-skip path?** **Yes.** The skip is *exact* only because the
activation is `relu(g)·relu(u)`: `gh[i] ≤ 0` implies `h = 0` identically. `Qwen2.5-1.5B` is `silu(g)·u`, and
`silu(g)` is never exactly zero for `g < 0`. A threshold-based approximation is available but is a quality
change — precisely the one P61 / probe-2 price separately. **The Builder is right, and right to refuse to
credit the donor with sparsity it does not have.** 21.25 / 11.398 = **1.864×**, as claimed.

## F10 — PASS. Every rebuilt control fires, verbatim, in my build.

`bin/ctrl3_donor_bench.exe mlpctl`. This is the control that was **dead once**, so I ran all of it:

- M1 accuracy gate: rel 5.462e−08 vs float64, PASS (and the Builder is right that this comparator's own
  5.5e−08 floor is what masked the first version).
- M2 ladder A (bit-exact vs pristine fp32): **N=0 → 0/1536 changed, silent**; **N=1 → 1536/1536, FIRES**;
  monotone through N=768. dReLU left 2280/8960 rows active; scale-setting row 6390.
- M2 ladder B: **δ=0 → 0/1536, silent**; **1 ULP (0.0126 → 0.0126000009) → 1330/1536, FIRES**; δ=1e−7 fires.
- M3 half-tile zeroing of `down`: **1533/1536 changed → CAUGHT.**
- M4 byte accounting: 2 359 296 / 55 296 / 578 027 520 / 2 179 072 — **all four MATCH.**

Both directions on every ladder. **PASS.**

*One honest qualification the §10.3 headline should carry.* "This path resolves one code byte in 6.88
million" is true **only for the deliberately chosen row**. Row 6390 is the row that sets the re-quantisation
scale, which is why a single code flips *all 1536* outputs. A blindly-chosen code has a roughly 75% chance
of landing in a dReLU-dead row and being invisible by construction — which is exactly how version 1 died.
The rebuild swung from the worst case to the best case; neither is the expected case. The control is valid
(it fires on a known positive and is silent on a known negative); the *sensitivity figure* is a best case
and should be labelled as one.

## F11 — FLAG. No planted control covers anything added in rev C.

`python donor_inventory.py control` — my run — is C1 (sandbox byte prediction, 0.0000%), C2 (`top-k` 8→7,
−12.50%), C3 (named refusal, both directions). **All three are byte-accounting controls.** `cmd_control`
contains no call to `time_model`, `kv_per_attn_layer`, or `n_att_override`.

The rev-C additions — the scan-recurrence term, the attention terms, KV-scaling-with-`n_att`, and the whole
`n_att` design curve — are **the terms the verdict now turns on, and not one of them has a control.** Under
`feedback_planted_controls` their nulls do not yet count. A cheap sufficient control would be: at
`n_att = L` with SWA the model must reproduce the tool's own unconverted row exactly; at `n_att = 0` the KV
term must be exactly zero and the scan term exactly `L` times the anchor; and a one-layer perturbation must
move the total by exactly one layer's worth. None is present.

## F12 — FLAG. The full-attention desk model: the *form* is sound, the *constant* is not, and the "≤2 of 28" deliverable rides entirely on it.

**The form is correct.** Per retained layer, decode attention over context `c` costs `QK^T` = `c · n_head ·
head_dim` MACs and `AV` the same, i.e. `4·c·D` flops. At c=32768, D=1536, 28 layers that is
**5.637 GFLOP/token**. I re-derived it independently; the Builder's formula is right, and charging it *on
top of* the separately-budgeted KV traffic is the right structure.

**The constant is not measured and the bracket is 5.2× wide:** `[19.2 .. 100] GFLOP/s`, central 40. The slow
edge is described in-source as "the measured t6 proj-GEMV rate class (memory-bound floor)" — but the memory
traffic is *already* charged in `t_kv`, so the slow edge double-charges memory. The source says as much
("pessimistic here since attention has ~12× the arithmetic intensity"); it is still the edge that sets the
crossing, because PASS is read off the lower bound.

**The "≤2 of 28" conclusion depends on it entirely.** Holding everything else fixed and moving only the slow
edge (my computation, fp16 KV, 32K):

| slow edge GFLOP/s | 19.2 (as shipped) | 30 | 40 | 60 | 100 |
|---|---|---|---|---|---|
| lower bound drops below 10 at n_att = | **3** | 4 | 5 | 7 | **11** |

The defensible statement is *"full attention at 32K is expensive enough that only a handful of layers can
retain it"*. **"At most 2 of 28" is not a deliverable; it is one draw from an unmeasured constant.** §0.2's
crossing-point table *is* labelled "not measured", but §0.2 then says "**The crossing points are the
deliverable**", and §0.3 states the ≤2 bound as a design rule without the label.

**Is the desk-model label carried everywhere it should be?** No. It travels correctly in the tool's `curve`
stdout and in the §0.2 table row. It is **absent** at `DONOR_STAGE1_ARITHMETIC.md:76` and in the T6
`UNCONVERTED` column (`:359`) — the two places the number is actually quotable. See F13.

## F13 — PASS on the FAIL, FLAG on the number. The 4.70 is 66% desk model.

`python donor_inventory.py curve`, n_att=28, full attention, 32K: **4.70 [2.71..7.88]** — exact
reproduction. The FAIL is robust: even at the fast edge, **7.88 < 10**. The claim *"the conversion is the
entire difference between failing and passing"* survives across the whole bracket, and that is the
defensible form of it.

The number should not be. Decomposed (central µs, my run):

| term | µs | share |
|---|---|---|
| **attention (desk model)** | **140 929** | **66.2%** |
| proj+head | 38 892 | 18.3% |
| ternary MLP | 27 335 | 12.8% |
| KV | 5 592 | 2.6% |

**Two-thirds of the most quotable number in the programme is an unmeasured compute term on a 5.2× bracket,
quoted to three significant figures with no desk-model label at either site.** Under the doc's own declared
central rule (F2) it would read **4.05**. Under the retained `full_swaScaled` upper bound it reads 1.75.
Two further points:

- **4.70 is the 4-bit KV row.** On fp16 — the only precision built, and the precision the passing headline
  uses — the unconverted figure is **4.36**. The two halves of "conversion is the entire difference" are
  quoted on different KV precisions.
- Recommended form: *"unconverted, full attention at 32K, the donor fails across the whole bracket
  [2.7 .. 7.9] tok/s, and the term that dominates it is a desk model."*

## F14 — PASS. KV does scale with retained attention layers, and it is applied consistently.

`kv_per_attn_layer()` divides the donor's total KV by its **original** attention-layer count and the callers
multiply by `n_att`. I verified the per-layer figure by hand: `num_key_value_heads=2 × head_dim=128 × 2
(K,V) = 512` elems/token/layer × 2 B × 32768 = **33 554 432 B**, matching the tool exactly; × 28 = 896 MB,
matching T4. It is applied in both `cmd_curve` and `analyze`'s `converted_*` rows (`b / n_orig * n_att_c`),
and `cmd_grid` correctly uses the *unconverted* `n_att` for a 2×2 that is explicitly about the unconverted
model. **Correct and consistent.** The rev-B defect the prior audit named is genuinely fixed. (The
incoherence in F3.2 is a different one: *which* KV a windowed layer should read, not *how many* layers read
it.)

## F15 — FLAG. Third session, third answer. The PASS-critical lower bound is built from within-session error bars.

`bin/ctrl3_donor_bench.exe fullstack --reps 5`, my session: byte accounting EXACT,
**t6 = 40.34 GB/s = 38.425 ms/token, cv 2.16%** — *above* the tool's high edge of 40.05.

| session | donor proj stream | quoted error bar |
|---|---|---|
| Builder | 37.74 | ±0.18 |
| Controller #2 | 39.87 | ±0.09 |
| **me** | **40.34** | cv 2.16% |

**Spread 6.9%, against quoted bars of 0.2–0.5%.** This is the prior audit's C4 FLAG, now with a third data
point confirming it — and the same policy is applied to the MLP constant (`21.25 ± 2×0.36`, a four-run
within-session repeatability). Those two terms are **79% of the token budget** and their bracket edges set
the PASS. Meanwhile the honestly-uncertain terms (scan, SWA, glue) get a ×2 interval. **The uncertainty
policy is inverted: the declared-not-measured terms are widened, the measured-once terms are not.**

For proj the direction happens to favour the donor (my draw is 8% above the low edge). For the MLP it is
untested across sessions. The methodology remains unsound regardless of which way the draws fall.

## F16 — FLAG. Half the document is still rev B and contradicts §0.

`DONOR_STAGE1_ARITHMETIC.md` §0–§2 are rev C; §5, §7 and §8 were not regenerated.

- **§5.1** gives `Qwen2.5-1.5B` as **12.10 [10.17..14.92]**, arithmetic shown as *"ternary MLP ÷
  [11.40..27.64] = 20.99–50.91 ms"* — the bracket §0.4 says was **withdrawn**. §0.3 gives
  13.30 [11.77..14.11] for the same donor. The elimination ledger contradicts the headline.
- **§7** still frames the conclusion as *"What would invalidate the headline (`no donor passes on measured
  rates`)"* — the rev-A headline, twice retracted.
- **§1.2 / §1.3** tables still list proj central **37.74** and dense MLP **[11.40 .. 27.64]**; the tool uses
  **39.87** and **[20.53, 21.97]**.
- **§8 ENV manifest is stale, and that is the serious one.** It records `generated 2026-08-20T11:17:00Z`,
  `sha256 2951bce1…`, `bytes 76043`. The actual tool is **89 633 bytes, sha256 `27a44121…`**, and
  `inventory.json` says `generated_utc 2026-08-21T05:14:45Z`. Under project law 9 (*"a tag is not a
  specification, a content hash is"*) the reproducibility manifest does not identify the artefact that
  produced the tables. `python donor_inventory.py doc` needs re-running before this is published.

---

## VERDICT

**Does `Qwen2.5-1.5B` clear the sealed ≥10 tok/s gate?**

# NOT YET — UNDECIDED on the built precision, PASS only on an unbuilt one

**At what configuration, on which precision, at what confidence:**

| configuration | KV | verdict | confidence |
|---|---|---|---|
| unconverted, full attention @32K, 28/28 | either | **FAIL**, robustly across [2.7..7.9] | **high** |
| retained **full** attention, any `n_att ≥ 3` | either | **FAIL** — but the crossing is a desk-model artefact (2 vs 11) | low on the crossing, high on the direction |
| `n_att = 14`, **windowed**, **4-bit KV** | unbuilt | **PASS**, survives every sensitivity I applied (worst 10.06) | moderate-high — but the precision does not exist |
| **`n_att = 14`, windowed, fp16 KV — the only precision built** | built | **UNDECIDED, trending pass** | — |

The last row is the claim under audit. It reproduces exactly at **11.97 [10.67..12.68]**, and it survives
the two attacks the brief prioritised: the integrated-MLP measurement is corroborated within 1.8% (F5), its
known-positive is **−3.4%**, not +13.5% (F6), and even a hypothetical 13.5% MLP penalty leaves the lower
bound at 10.25 (F6). The E3.5 identification and the double misattribution of `11.398` are both correct
(F7, F9). Every rebuilt control fires (F10).

It does not clear, yet, for one reason: **F4.** The lower bound has 6.65% of headroom, and the model omits
the converted layers' own projection weights while charging them scan-recurrence and crediting them with
removed KV. Priced at the engine's own Arch-A SSM shapes under the tool's own precision map, that term is
worth **14%** — twice the headroom — and the headline becomes **10.30 [9.24..10.83], STRADDLE**. The
resolution that keeps the PASS requires ternary projections, which the document itself marks UNMEASURED and
which P61 rejected on quality.

**Required before this may be reported to the owner as clearing the gate:**

1. **F4 (BLOCK)** — state and price the converted layer's weight model. If the conversion is
   shape-preserving, say so and cite it; then reconcile the scan-recurrence term, which is priced off a
   block that would no longer be the one supplying the bytes.
2. **F16** — regenerate §5, §7, §8 and the ENV manifest. The document currently publishes two different
   headline numbers for the same donor.
3. **F11** — plant controls on the rev-C terms before their nulls count.
4. **F12 / F13** — carry the desk-model label to `:76` and T6 `:359`; retire "at most 2 of 28" as a
   deliverable; requote the unconverted result as a bracket, on a stated KV precision.
5. **F2 / F15** — either implement the declared midpoint-in-time central or restate the rule; requote the
   proj and MLP error bars as within-session repeatability. Three sessions now span 6.9% on proj.

**Not blocking, and it should go to the owner separately:** the engine's default `--skip on` MLP path is
**1.87× slower** than `--skip off` at t6 at claimed-identical output, worth **1.20× engine-wide** on the
8.3M sandbox (F7).

*No commit, no push. Nothing outside this file was written. Binaries were built to `bin/ctrl3_*.exe` so as
not to overwrite the Builder's or Controller #2's.*
