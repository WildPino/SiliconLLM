# E13 — is the LUT kernel's collapse a LAYOUT artifact? Blocked tile-major.

**Pre-registration.** Written and pushed before the arm exists. Bands in §4 are fixed here and are
read off mechanically afterwards.

Donor: `Qwen2.5-Coder-7B` packed, 7.072 B active weights/token. Machine: Ryzen 5 3600X, 6 threads.

---

## 1. Why this, and why now

E10 returned `CORE-BOUND` and withdrew §19.4's bandwidth rows. It named the side but **explicitly
offered no mechanism**, and it left one number unread in its own table:

| E10's sweep, 2048 MB, DRAM-resident, 6 threads | moved GB/s | G-weights/s |
|---|---|---|
| fp32 arm | **38.58** | 9.65 |
| packed arm | **26.48** | 52.95 |

Both are **moved** bytes, same instrument, same run, same threads — the two conventions never meet
in a fraction here, so the comparison is legal. **The packed kernel is running at 69% of a
streaming rate the same sweep demonstrated on the same machine.** E10's verdict is therefore
sharper than "at the ceiling": it is at *a* ceiling that is **1.46× below** the one the hardware
was shown to reach. That 1.46× is measured, not modelled, and it is the largest engine-side number
still on the table.

E11 then measured a kernel that is **1.8× faster per weight than packed inside L3** and 2.5×
slower outside it, and attributed the collapse to **layout**:

> `codes + t*Mpad + base` puts consecutive reads 299 KB apart at donor scale.

**That attribution was asserted, not tested.** E11 §3 is labelled exploratory and it is the only
unexamined mechanism claim this programme currently carries. This experiment tests it by changing
**nothing but the layout**.

## 2. The defect, read in the source

`build_tm` (`donor_engine.c:352`) stores tile-major as `tm[t*Mpad + o]`. `matvec_lut`
(`donor_engine.c:341`) then holds `base` fixed and walks `t`, reading 32 bytes at stride `Mpad`.

At the shapes that matter the stride is not a rounding detail:

| organ, Coder-7B | `Mpad` = stride between consecutive reads | bytes used per 64-byte line |
|---|---|---|
| `gate_proj` / `up_proj` (out 18944) | **18,944 B = 18.5 KB** | 32 of 64 |
| `down_proj` (out 3584) | **3,584 B = 3.5 KB** | 32 of 64 |
| kbench 512 MB cell (`n_in` 3584) | **299 KB** | 32 of 64 |

Every stride above is larger than a 4 KB page except `down_proj`, which is nearly one. The kernel
reads **half of every line it touches** and then leaves the page. The total bytes read are
identical to the packed arm's — **the same matrix, once** — so this is purely an ordering defect.

## 3. The change

Store each 32-row block's bytes **contiguously**: `tm[(base/32)*T*32 + t*32 + r]` instead of
`tm[t*Mpad + r]`. Within a block the kernel then walks memory **linearly**, 32 bytes at a time,
using every byte of every line.

- It is a **pure permutation of the same bytes**. No re-encoding, no numerics.
- The accumulation order per output row is **unchanged** — same `t` sequence, same tile, same
  `acc_add_i8x32`. So the result is **bit-identical**, and the gate is sha256, not parity
  (the E9 standard, which E8 could not meet).
- Behind `--lutblk`, **default off**, so every `--lut` number this programme has published stays
  reproducible byte-for-byte. Adopted as default only if it wins — the E8 order, not the §9 defect.

## 4. Gates and bands, fixed now

| gate | test | band |
|---|---|---|
| **G-M0** | `--logits` sha256, `--lut` vs `--lut --lutblk`, 0.5 B | **must be byte-identical.** A permutation that changes a bit means the index arithmetic is wrong, and nothing below counts. |
| **G-M1** | blocked-lut G-w/s at the **512 MB** cell (the verdict cell E11 named in advance) | **≥65 `LAYOUT-CONFIRMED`** · 30–65 `LAYOUT-PARTIAL` · **≤30 `LAYOUT-REFUTED`** |
| **G-M2** | packed arm, every cell — the known-positive | **47–62 G-w/s and flat (max/min ≤ 1.30)** |
| **G-M3** | end-to-end engine rate, `--lut --lutblk` vs `--lut`, resident shape (0.5 B) | sha256 identical **and** rate must not fall |

**G-M1's boundaries, derived — not chosen.**

- **65** is above the *top* of every reading the packed arm has ever given at the large cells:
  E10 7-rep 50.97 / 52.95, E10 3-rep 57.11 / 60.78, E11 run 2 55.95 / 60.25 → union **50.97–60.78**.
  Clearing 65 means the LUT kernel beats packed at donor scale, which is exactly what E11 refuted.
- **The stream ceiling this instrument has actually demonstrated** at these cells is the fp32 arm's
  37.12 GB/s (512 MB) and 38.58 GB/s (2048 MB). Over the structural factor 0.5 B/weight that is
  **74.2–77.2 G-w/s** — the most a packed-format kernel could reach without moving fewer bytes.
- **30** is E11's measured 21.90 inflated by the dispersion this instrument is known to carry:
  E10's 4 MB cell spread **30.6%** (the G-K3 lesson), and 21.90 × 1.30 = 28.5. **Below 30 the
  reading is indistinguishable from E11's, and layout is refuted as the mechanism.**

**G-M2's band is drawn from EVERY reading of the known-positive**, which is E11's law, written
after I drew a band from one of E10's two sweeps and ignored the other. Union 48.14–60.78, widened
to 47–62. A packed arm outside it, or with a slope, voids the run — as it correctly did in E11 run 1.

## 5. Prediction, on the record

**65–77 G-w/s at the 512 MB cell, i.e. `LAYOUT-CONFIRMED`.**

Derived, per E8 §3's corollary, from a measured quantity over a structural factor: the fp32 arm's
demonstrated 37.12–38.58 GB/s divided by 0.5 B/weight, floored at the top of the packed arm's
observed range. The reasoning is that the blocked kernel streams the same bytes in the same order
as packed while issuing **fewer instructions per weight** — counted from the source, not from a
port model:

| per 64 weights | packed (`--mvacc 4`) | `matvec_lut` |
|---|---|---|
| vector instructions | ~36 | ~12 |

**This is an instruction count from the intrinsics, not a Zen 2 port model.** I do not have a
citable issue/port table for this core and will not build one; the count bounds nothing on its own
and is offered only as the reason the prediction points up rather than down.

## 6. What a `LAYOUT-REFUTED` would mean

That **E11 §3's mechanism paragraph is wrong** and must be retracted, and that the LUT collapse is
something else — most likely the table walk or the int8 accumulate tree. That outcome is worth as
much as the other one and is why the gate is symmetric.

## 7. What this cannot claim

- It is a **microbench**. Phase 61's law: a compute-bound microbench does not compose to a
  memory-bound engine. G-M3 exists because of it, and no engine claim is made without G-M3.
- Phase 60's law: kernel-bit-exact does not compose to system correctness. G-M0 is bit-exactness of
  the *kernel arm*; the engine claim still rides on the end-to-end run.
- **It does not touch activation quantization.** `--lut` quantizes activations to int8 and E11
  measured that cost as **1.40e-01** whole-vector, **3.10e-02** at G=32 — not a rounding error, and
  not addressed here. **A speed verdict on `--lutblk` is not a licence to ship `--lut`.**
- Even a full `LAYOUT-CONFIRMED` at the top of its band is **1.46×**, which takes 6.79 tok/s to
  roughly 9.3. **It does not reach 50 and is not claimed to.** The gap remains a property of the
  model, per §19.3 and E10.

## 8. Contention

Speed requires an idle machine. Another checkout (`SiliconLLM_private`) is streaming the 30 GB fp32
donor on six threads as this is written, and it voided E11 run 1. **G-M0 is deterministic and is run
now; G-M1/G-M2/G-M3 are timings and wait for a quiet machine.** G-M2 is the witness that decides
whether the wait was long enough.
