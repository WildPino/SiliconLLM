# P1 — Nibble packing: 2 bits/weight with the pshufb path intact

**Builder** · run 2026-08-29 · brief `docs/research/donor_adaptation/briefs/BRIEF_P1_NIBBLE_PACKING.md`
(pre-registered `3f885c5`, Amendment 1 `e73420e`, repo tip at run `565b4dd`).

**Stage 1 (correctness) and Stage 2 (timing) are separate sections below. Stage 1 stands on its own.**

---

# STAGE 1 — CORRECTNESS

**Verdict: C1 PASS · C2 PASS · C3 PASS · C4 reported. The 2x is available and bit-exact.**

No timing number appears anywhere in this section, by construction: the Stage 1 harness contains no clock.

## S1.0 What was built

Two arms coexist; **nothing on the shipped path was replaced or deleted.**

| file | status | what it is |
|---|---|---|
| `benchmarks/donor_adaptation/nibble_pack.h` | **new** | the whole nibble arm: 2 encoders, 3 AVX2 kernels, 1 scalar accessor. Single source of truth, included by both the test harness and the engine, so C1 and C3 exercise the same code. |
| `benchmarks/donor_adaptation/nibble_pack_test.c` | **new** | Stage 1 harness: C1, C2, C4. No clock. |
| `benchmarks/phase60/engine.c` | **modified**, +50 / -19 | adds `--pack byte|nibble`. `byte` is the shipped path, untouched. |

**Layout.** Code for `t` in the low nibble, code for `t+1` in the high nibble, `m` contiguous exactly as today:

```
tile-major (streamed):   codes2[(size_t)(t/2)*Mpad + m]        Tp = (T+1)/2 planes
row-major  (skip path):  codes2[(size_t)m*Tp + (t/2)]
```

**Inner loop, per byte-plane:** 1 load, `vpand`, `vpsrlw`+`vpand`, 2 broadcasts, **2 `pshufb`**, 2 accumulates.
Per-weight `pshufb` count is unchanged from the byte arm. One 32-byte load removed per 4 trits.

`_mm256_srli_epi16(b,4)` pulls bits across the byte boundary inside each 16-bit lane; masking with `0x0F`
discards them, so the per-byte high nibble is recovered exactly. Masking also clears bit 7, so the `pshufb`
lane-zeroing behaviour is never triggered.

## S1.1 The two traps from brief section 2 — derived for the new layout, not inherited

### Padding

`c=0` decodes to `(w0,w1) = (-1,-1)`. The neutral code is `4`. Today `bc_tm` writes `0` into rows
`m in [M,Mpad)`. **That is not why it is safe**, and the reason had to be re-derived because the plane count
changed from `T` to `Tp`:

1. **Padding lives on the OUTPUT axis `m`. Nibble packing changes only the REDUCTION axis `t`.** The
   `m`-blocking is untouched: still 32 rows per vector, still `Mpad = (M+31)&~31`.
2. The store guard `for(r=0;r<32 && base+r<M;r++)` is on the output axis and is **unchanged**, so every
   accumulator lane fed by a padding row is discarded before it is written.
3. `acc_add_i8x32` is strictly lane-wise (sign-extend `int8` to `int32`, add). **A padding lane cannot leak
   into a live lane.** So padding content is arithmetically irrelevant in both arms. We still write `0` so
   the array is fully initialised and the encoding is deterministic.
4. **The part that genuinely needed re-deriving is the allocation bound.** Allocation is `Tp*Mpad`; the last
   32-byte load is at `(Tp-1)*Mpad + base_max` with `base_max = 32*((M-1)/32)`, and `base_max+32 <= Mpad`
   because `Mpad` is a multiple of 32. In bounds. **`Mpad` must stay a multiple of 32 for this to hold.**

**This is not left as an argument.** Control **C2b** corrupts every padding byte to `0x88` — code 8 twice,
`(w0,w1)=(+1,+1)`, the least neutral value available — and requires C1 to still pass. It does, on 3 shapes
that actually have padding (7967, 2048 and 31 padding bytes). Shapes where `M == Mpad` are reported as
`n/a`, **not** as a pass.

### Odd `T`

When `T` is odd the final plane `tb = Tp-1` carries a real code only in its low nibble. The kernel runs full
pairs `tb in [0, T/2)` and then handles the leftover plane with the **low nibble only**:

> **The high nibble of the final plane is never loaded into a shuffle.** It cannot contaminate the
> accumulator because no instruction ever reads it — not because its value happens to be benign.

The encoder writes `0` there so the byte is defined. Control **C2c** sets that nibble to `8` for every `m`
and requires C1 to still pass. It does, on 3 odd-`T` shapes (`T` = 513, 4095, 7).

**`K` parity:** `T = K/2` truncates, exactly as the existing `bc_tm`/`bc_rm` do. Odd `K` silently drops the
last weight in **both** arms. Pre-existing property of the byte path, not introduced here.

## S1.2 Control table

| # | control | verdict | the actual numbers |
|---|---|---|---|
| **C1** | nibble vs `ref_t3` scalar **and** vs today's byte kernel, identical `int32` | **PASS** | **523,395 exact int32 comparisons, 15 shapes, 0 mismatches.** Covers `matvec_lut_full`, `matvec_lut_tileskip`, `matvec_lut_rows` and the row-major scalar skip path. |
| **C2a** | planted hi/lo nibble swap must **break** C1 | **PASS — fires** | **18/18 plants fired**, 3 sites x 6 shapes. Each changed **exactly 1 row**, at the planted `m`, by **exactly the predicted delta** (e.g. `+58`, `-42`, `-109`, `+119`, `-143`). |
| **C2b** | padding-discard demonstration | **PASS** | every padding byte set to `0x88`; **0 rows differ**, on the 3 shapes that have padding. |
| **C2c** | odd-`T` unread-nibble demonstration | **PASS** | high nibble of the last plane set to `8`; **0 rows differ**, on the 3 odd-`T` shapes. |
| **C3** | real engine, real weights, real prompt: token stream and BPB identical | **PASS** | **4 configurations, 2 models. fp32 logit streams SHA-256-identical, 0/2048 (and 0/1024) tokens differ, BPB identical to all 6 printed digits.** |
| **C4** | bytes streamed per matvec, both arms, read from the allocation | **reported** | **exactly 2.000000x** at every engine and donor shape with even `T`. In-engine: dense 3.000 to 1.500 MiB, MoE 9.000 to 4.500 MiB. |

### C1 / C4 — shapes swept, ACHIEVED

Every value printed from the objects themselves. `T`, `Tp`, `Mpad` are ACHIEVED, not requested.

| shape | M | K | T | Tp | Mpad | C1 | alloc byte (B) | alloc nib (B) | ratio |
|---|---|---|---|---|---|---|---|---|---|
| eng_dense_gate_up | 1024 | 256 | 128 | 64 | 1024 | PASS | 131,072 | 65,536 | 2.0000x |
| eng_dense_down | 256 | 1024 | 512 | 256 | 256 | PASS | 131,072 | 65,536 | 2.0000x |
| eng_moe_gate_up | 4096 | 256 | 128 | 64 | 4096 | PASS | 524,288 | 262,144 | 2.0000x |
| eng_moe_down | 256 | 128 | 64 | 32 | 256 | PASS | 16,384 | 8,192 | 2.0000x |
| **donor70b_kv** | 1024 | 8192 | 4096 | 2048 | 1024 | PASS | 4,194,304 | 2,097,152 | 2.0000x |
| **donor70b_qo** | 8192 | 8192 | 4096 | 2048 | 8192 | PASS | 33,554,432 | 16,777,216 | 2.0000x |
| **donor70b_gate_up** | 28672 | 8192 | 4096 | 2048 | 28672 | PASS | 117,440,512 | 58,720,256 | 2.0000x |
| **donor70b_down** | 8192 | 28672 | 14336 | 7168 | 8192 | PASS | 117,440,512 | 58,720,256 | 2.0000x |
| odd_T3_M37 | 37 | 6 | 3 | 2 | 64 | PASS | 192 | 128 | 1.5000x |
| odd_T513_M97 | 97 | 1026 | 513 | 257 | 128 | PASS | 65,664 | 32,896 | 1.9961x |
| odd_T4095_M8191 | 8191 | 8190 | 4095 | 2048 | 8192 | PASS | 33,546,240 | 16,777,216 | 1.9995x |
| odd_T1_M1 | 1 | 2 | 1 | 1 | 32 | PASS | 32 | 32 | **1.0000x** |
| M31_T2 | 31 | 4 | 2 | 1 | 32 | PASS | 64 | 32 | 2.0000x |
| M33_T2 | 33 | 4 | 2 | 1 | 64 | PASS | 128 | 64 | 2.0000x |
| M32_T7odd | 32 | 14 | 7 | 4 | 32 | PASS | 224 | 128 | 1.7500x |

Donor width = Llama-3-70B class, `d_model` 8192, `d_ffn` 28672, matching the organ shapes already used in
`gemv_donor_bench.c`: `q/o` 8192x8192, `k/v` 1024x8192, `gate/up` 28672x8192, `down` 8192x28672.

### C4 — the byte convention, stated once, applied identically to both arms

> **Bytes streamed per matvec** = bytes of the **weight-code array** fetched by the kernel, counted as
> **32 B per `_mm256_loadu_si256` from `codes`**, summed over one complete matvec.
> **Excluded from BOTH arms:** the 16 B/tile LUT broadcasts (activation-derived, **identical count in both
> arms**, L1-resident), the `y` output store, the fp32 per-row scales.
> **Padding rows are charged to both arms alike.** `bits/weight = 8 * bytes / (M*K)`.

Two independent readings are printed and they agree to the byte at every shape:

1. **Allocation** — `_msize()` of the pointer `malloc` returned. Not `T*Mpad` recomputed.
2. **Counted loads** — a runtime counter incremented inside the kernel, structurally identical in both arms.

In-engine, summed over every ternary code array via `_msize`:

| model | pack=byte | pack=nibble | ratio |
|---|---|---|---|
| E1M1 dense 8.3M | 3,145,728 B (3.000 MiB) | 1,572,864 B (1.500 MiB) | **2.000000x** |
| E4M1 MoE | 9,437,184 B (9.000 MiB) | 4,718,592 B (4.500 MiB) | **2.000000x** |

### C3 — end-to-end, the Phase-60 law

Standing law: *kernel-bit-exact does NOT compose to system-correctness.* C1 does not discharge this.

Instrument: the raw fp32 logit stream (`--dumplogits`, this project's parity-acceptance instrument), its
greedy argmax token stream, and the `--bpb` line. Identity = SHA-256 over the whole stream. Logit-stream
identity subsumes token identity for **any** deterministic decoder, and the argmax stream is compared
explicitly on top of it.

| config | model | logits SHA-256 identical | tokens differing | BPB byte | BPB nibble |
|---|---|---|---|---|---|
| `--mlp lut --skip on --exp fast` (production) | E1M1 dense | **yes** `4277910243ad4cde...` | **0 / 2048** | 0.907629 | 0.907629 |
| `--mlp lut --skip off --exp exact` (E2) | E1M1 dense | **yes** `647020cb1739d7d4...` | **0 / 2048** | 0.907629 | 0.907629 |
| `--mlp fp32 --exp exact` (LUT path unused) | E1M1 dense | **yes** `352e201a7d86a080...` | **0 / 2048** | 0.907550 | 0.907550 |
| `--mlp lut --exp fast` | E4M1 MoE | **yes** `b54998bcaec7405d...` | **0 / 1024** | 0.874376 | 0.874376 |

**Regression guard.** A fourth binary was compiled from `git show HEAD:benchmarks/phase60/engine.c`, i.e. the
engine *before* my edits, and run on the same configs. Its logit streams and BPB are **identical to the
patched binary under `--pack byte`** in all three configs where it applies. **My edits did not perturb the
shipped path.**

`--skip on` matters here: it is the only config that exercises the **row-major scalar** nibble accessor
(`np_rm_code`) and `matvec_lut_tileskip_n`. Both are covered.

**Refused rather than faked:** `--g3c` (the layer-major weight-once block-decode path) has no nibble variant
because `matvec_lut_full_K` was not converted. The engine now **hard-errors** on `--g3c --pack nibble`
instead of silently mixing layouts.

## S1.3 Things that contradict, or qualify, the brief

The brief's author asked for these explicitly.

1. **"Exactly 2x fewer bytes" is exactly right only when `T` is even.** The achieved factor is
   **`2T / (T+1)`**. At `T=1` it is **1.0x — no saving at all**; at `T=7`, 1.75x; at `T=513`, 1.9961x.
   Every engine and donor shape has even `T` (128, 512, 4096, 14336), so the practical factor **is** exactly
   2.000000x — but the claim should be stated as `2T/(T+1)`, not as an unconditional 2x. This is the odd-`T`
   tail plane being half empty, and it is unavoidable in this layout.
2. **`Mpad` rounding dominates both arms at small `M`, and the nibble arm does not fix it.** At `M=1, K=2`
   both arms stream 32 B for 2 weights: **128 bits/weight**, and the ratio is 1.0x. At `M=33` both arms pay
   the 64-row `Mpad`: 7.76 vs 3.88 bits/weight — still 2x, but both far above the nominal 4 and 2. The
   nibble lever halves whatever the padding geometry hands it; **it does not attack the padding geometry**,
   which is a separate and, at donor width, negligible term (`Mpad == M` at every donor organ shape).
3. **The brief's per-byte cost table (section 1) is arithmetically right but understates a structural
   change.** It lists the added cost as one shift and two masks, purely additive. That is what I
   implemented. But the two `pshufb` now carry a **serial dependency on the mask ops** rather than each
   depending on an independent load, and the mask constant occupies a vector register across the inner
   loop. Whether that costs anything is a Stage 2 question — flagging it because the brief presents the
   added cost as purely additive when it also changes the dependency shape.
4. **The padding argument silently depends on `codes` never holding a value above 15.** `bc_tm` writes
   literal `0` into padding, so it holds. Had any encoder ever written `>15`, the byte arm would already be
   relying on the `pshufb` bit-7 zeroing while the nibble arm's mask would silently change behaviour. It
   does not happen, but this is a real coupling between the two arms that the brief does not mention.
5. **`--g3c` is now unavailable under nibble packing.** The block-decode weight-once path would need
   `matvec_lut_full_K` converting before P1 could be adopted engine-wide. That is unbuilt work the brief
   does not scope.

## S1.4 Allocation inventory (O(input))

Printed from `_msize` of the returned block, per shape. Largest swept shape, `donor70b_gate_up`
(M=28672, K=8192):

| tag | requested B | `_msize` B |
|---|---|---|
| `Wt` (raw ternary int8) | 234,881,024 | 234,881,024 |
| `xq` | 8,192 | 8,192 |
| `lut` | 65,536 | 65,536 |
| `codes_byte` | 117,440,512 | 117,440,512 |
| `codes_nib` | 58,720,256 | 58,720,256 |
| `rowm_byte` | 117,440,512 | 117,440,512 |
| `rowm_nib` | 58,720,256 | 58,720,256 |
| `y_ref` / `y_byte` / `y_nib` | 114,688 each | 114,688 each |
| `act` | 16,384 | 16,384 |
| **TOTAL** | | **587,636,736 B (560.41 MiB)** |

All O(M*K) or O(T*Mpad); every one is freed before the next shape, so peak harness footprint is one shape's
total, 560 MiB. In-engine the only new O(input) allocation is the code arrays themselves, which **shrink**
by 2x under `--pack nibble`.

## S1.5 Reproducibility manifest — Stage 1

**Compiler.** `clang version 21.1.8` (llvm-mingw, msvcrt), target `x86_64-w64-windows-gnu`,
`InstalledDir: .../llvm-mingw-20251216-msvcrt-x86_64/bin`.

**Flags.** `-O3 -mavx2 -mfma -march=znver2 -Wall`. **No `-ffast-math`** — standing law, and meaningless here
anyway since the kernels are integer.

**Machine.** AMD Ryzen 5 3600X, 6 cores / 12 threads, 3793 MHz max, L3 16 MiB per CCX.
Windows 11 Pro 10.0.26200. 80 GB RAM (81,868 MB visible).

**Machine state during the Stage 1 runs — recorded, not assumed.** Stage 1 is bit-exact and therefore immune
to contention, but it is recorded because Stage 2 is not. At the time of the C3 runs `llama-server.exe` was
resident at **17,602 MB working set / 32,940 MB private**, plus two other Builders' python jobs (pids 14000,
26112) at 1,787 and 1,192 MB. **This is exactly why Stage 2 was withheld at that point.** See
`benchmarks/donor_adaptation/p1/results/c3/machine_state.txt`.

**Repo state.** `565b4ddd8e2a2fb5014b2942b8bf4069aa4118cc`, branch `research/donor-adaptation`. Not committed.

**Source files touched (left in the working tree, uncommitted):**

- `benchmarks/donor_adaptation/nibble_pack.h` — new
- `benchmarks/donor_adaptation/nibble_pack_test.c` — new
- `benchmarks/phase60/engine.c` — modified, +50 / -19

**Exact commands.**

```sh
# --- build ---
clang -O3 -mavx2 -mfma -march=znver2 -Wall \
      benchmarks/donor_adaptation/nibble_pack_test.c -o bin/nibble_pack_test.exe -lm
clang -O3 -mavx2 -mfma -march=znver2 -Wall \
      benchmarks/phase60/engine.c -o bin/engine_p1.exe -lm
# regression-guard reference: the engine BEFORE these edits
git show HEAD:benchmarks/phase60/engine.c > engine_head.c
clang -O3 -mavx2 -mfma -march=znver2 engine_head.c -o bin/engine_p1_ref.exe -lm

# --- C1 + C2 + C4 (this binary contains no clock) ---
./bin/nibble_pack_test.exe          # exit 0 = every control behaved as required
#   -> benchmarks/donor_adaptation/p1/results/c1_c2_c4.log

# --- C3, per arm; repeat with --pack byte / --pack nibble / the HEAD binary ---
./bin/engine_p1.exe --weights results/phase60/e1_model.bin --pack nibble \
    --mlp lut --skip on --exp fast --dumplogits OUT.logits --ntok 2048 --bpb --eval-tok 10240
./bin/engine_p1.exe --weights results/phase60/e4_model.bin --pack nibble \
    --mlp lut --exp fast           --dumplogits OUT.logits --ntok 1024 --bpb --eval-tok 5120
# identity = sha256 over the .logits files + the printed BPB line
#   -> benchmarks/donor_adaptation/p1/results/c3/C3_SUMMARY.txt
```

**Raw artefacts.** `benchmarks/donor_adaptation/p1/results/c1_c2_c4.log`,
`benchmarks/donor_adaptation/p1/results/c3/` (12 `.out` logs, 11 `.logits` streams, `C3_SUMMARY.txt`,
`machine_state.txt`).

---

# STAGE 2 — TIMING

**Verdict, in one line: P1 buys 1.58–1.91x matvecs/s at 6 threads on donor shapes once cache-set aliasing is
removed, while moved GB/s falls only to 0.79–0.96x — an INTERMEDIATE result, jointly limited, leaning
bandwidth. And the stride arm found something bigger than P1: the donor-width "fall past L3" is largely
CACHE-SET ALIASING, worth 1.43–1.68x for the price of an allocation offset, on the shipped byte arm alone.**

## S2.0 Machine condition — from my own sampler, not from any banner

Amendment 1 A1.6 forbids citing a quiescence banner. This harness prints none. An independent PowerShell
process sampled the machine every 1.5 s for the whole run and every sample is on disk
(`results/stage2/machine_samples.csv`).

| | |
|---|---|
| run span | **16:39:39 to 16:55:54**, 494 samples, no gaps |
| largest non-self process, every single sample | **`Memory Compression`**, max 1354 MB (a kernel-managed store, not a CPU competitor) |
| samples where `llama-server` / `python` / `pythonw` was the top process | **0 of 494** |
| free physical memory | 59,708–65,269 MB throughout (no pressure; pool peak ~560 MB) |
| processor queue length | **0 in 80.0% of samples**, mean 2.19 |

**One thing the sampler shows that I must not launder: the processor queue spiked above 6 in 21 of 494
samples (4.3%)**, between 16:47 and 16:55, peaking at 143 on the final sample as the run tore down. Those
spikes coincide with the 6-thread cells. **This is the honest bound on the t6 rows** and it is consistent
with their large within-invocation dispersion. It is recorded rather than dismissed.

`llama-server.exe` returned to the machine at **19:34:40 — 2 h 39 min after the run ended at 16:55:54** — so
it cannot have touched this data. **No new timing number was taken after that point**, and none should be.

## S2.1 The byte convention — restated, and a note that matters

> **MOVED bytes are the reporting convention.** Moved = bytes of the weight-code array the kernel's loads
> actually pull from memory in one matvec, **counted at runtime as 32 B per `_mm256_loadu_si256` issued from
> `codes`**, single-threaded, outside every timed window.
> **CHARGED bytes** = bytes attributable to the `M*K` useful weights only (`M*K/2` byte arm, `M*K/4` nibble
> arm). Carried in **its own labelled column**. **The two never meet in one ratio.**

**In this sweep the two columns happen to be numerically equal**, because all four donor organs have
`Mpad == M` (1024, 8192, 28672, 8192 are all multiples of 32) so there are no padding rows, and the +64
stride pad occupies exactly one 64 B line per plane that is **never requested**. Both columns are printed
anyway, so a reader can verify that no ratio crosses them.

Requested == moved here because accesses are contiguous 32 B within 64 B-aligned streams and `M` is a
multiple of 32 at every shape measured, so every byte of every touched line is requested.

## S2.2 Protocol, ACHIEVED

- **Primary estimator: per-rep MEAN** inside each invocation. **Min-of-reps is never computed** — a minimum
  over reps is a maximum order statistic and biases upward by construction.
- **5 invocations** (separate process launches, seeds `0x1001`–`0x1005`), aggregated afterwards; the tables
  below report the **mean of the invocation means**, the **between-invocation CV**, and the mean
  **within-invocation CV**.
- **200 timed cells**, `stage2_raw.err` empty, 0 `P1S2ERR` rows, 5 `P1S2DONE` markers.
- **Every cell is bit-exactness-guarded before it is timed**: the kernel is run once against `ref_t3` and the
  harness `exit(2)`s on any mismatch. This also proves the **+64 stride arm is still bit-exact** — it never
  fired.
- Threads ACHIEVED read back from `omp_get_num_threads()` inside a probe region, not from the request. Every
  cell achieved what it asked for (1 and 6).
- Replica pool ~512 MiB per cell, **equal footprint in bytes for both arms** (so the nibble arm simply gets
  2x the replicas). Streamed cells rotate replica per rep.
- Reps sized to stream ~20 GB per cell: 170 (`gate/up`, `down`) to 9536 (`k/v` nibble).

## S2.3 d5cd re-fired at donor shape — the meter is honest

Before quoting any rate: does the instrument distinguish a block that fits L3 from the same block streamed
from a pool >> L3? Donor shape `k/v` 1024x8192.

| arm | thr | matvecs/s resident | matvecs/s streamed | **resident / streamed** | moved GB/s resident | moved GB/s streamed |
|---|---|---|---|---|---|---|
| byte | 1 | 3099.6 | 1503.5 | **2.06x** | 13.00 | 6.31 |
| byte | 6 | 9627.5 | 4445.9 | **2.17x** | 40.38 | 18.65 |
| nibble | 1 | 3666.3 | 2318.8 | **1.58x** | 7.69 | 4.86 |
| nibble | 6 | 10531.5 | 6842.2 | **1.54x** | 22.09 | 14.35 |

**The meter fires.** Resident is 1.54–2.17x faster than streamed on the same kernel and the same shape.
Rates quoted below are therefore measuring something real. Note the nibble arm's resident/streamed gap is
*smaller* (1.54–1.58x vs 2.06–2.17x) — its 2 MiB block is half as costly to refill, which is itself
consistent with the residency story.

## S2.4 THE BIG FINDING — A1.5 stride-conflict separation

**Padding each row stride by 64 bytes is a shape change, not a kernel change.** Applied identically to both
arms. Same `M`, same `K`, same kernel, same bit-exact output; only the plane stride moves.

| shape | thr | arm | matvecs/s +0 | matvecs/s +64 | **+64 / +0** | moved GB/s +0 | moved GB/s +64 |
|---|---|---|---|---|---|---|---|
| `q/o` 8192x8192 | 1 | byte | 66.2 | 103.4 | **1.562x** | 2.22 | 3.47 |
| `q/o` 8192x8192 | 6 | byte | 224.2 | 375.6 | **1.675x** | 7.52 | 12.60 |
| `gate/up` 28672x8192 | 1 | byte | 17.3 | 24.7 | **1.429x** | 2.03 | 2.90 |
| `gate/up` 28672x8192 | 6 | byte | 61.5 | 103.4 | **1.681x** | 7.22 | 12.14 |
| `down` 8192x28672 | 1 | byte | 17.1 | 25.8 | **1.506x** | 2.01 | 3.03 |
| `down` 8192x28672 | 6 | byte | 62.2 | 101.6 | **1.634x** | 7.30 | 11.93 |
| **`k/v` 1024x8192** | 1 | byte | 1473.5 | 1370.0 | **0.930x** | 6.18 | 5.75 |
| **`k/v` 1024x8192** | 6 | byte | 4207.2 | 3999.8 | **0.951x** | 17.65 | 16.78 |
| `q/o` | 1 | nibble | 135.0 | 339.7 | **2.517x** | 2.26 | 5.70 |
| `q/o` | 6 | nibble | 489.6 | 718.0 | **1.466x** | 8.21 | 12.05 |
| `gate/up` | 1 | nibble | 71.3 | 67.8 | 0.951x | 4.19 | 3.98 |
| `gate/up` | 6 | nibble | 173.3 | 179.2 | 1.034x | 10.17 | 10.52 |
| `down` | 1 | nibble | 22.7 | 38.0 | **1.671x** | 1.33 | 2.23 |
| `down` | 6 | nibble | 79.7 | 160.7 | **2.016x** | 4.68 | 9.43 |
| `k/v` | 1 | nibble | 2312.0 | 2213.7 | 0.958x | 4.85 | 4.64 |
| `k/v` | 6 | nibble | 6605.0 | 6752.5 | 1.022x | 13.85 | 14.16 |

**Padding the stride by 64 B recovers 1.43–1.68x on the shipped byte arm at three of the four donor organs,
and `k/v` — the one shape it does not help — is exactly the shape the mechanism predicts it should not
help.**

### The mechanism, derived and then checked against which shapes moved

Zen2 L1d is 32 KB, 8-way, 64 B lines = **64 sets**; set index = `(addr >> 6) & 63`. The tile-major walk holds
`base` fixed and steps `t`, so consecutive loads are **one plane stride apart**. The set index therefore
advances by `(stride/64) mod 64` per load:

| shape | `Mpad` | `Mpad/64` | **set advance mod 64** | distinct L1 sets reachable | recovered by +64? |
|---|---|---|---|---|---|
| `q/o`, `down` | 8192 | 128 | **0** | **1** | **yes, 1.51–1.68x** |
| `gate/up` | 28672 | 448 | **0** | **1** | **yes, 1.43–1.68x** |
| `k/v` | 1024 | 16 | 16 | 4 | **no, 0.93–0.95x** |

**Every plane load in a `q/o`, `gate/up` or `down` matvec lands in the SAME L1 set.** Eight ways, thousands
of planes: the walk evicts itself on every access. `k/v` alone is not a multiple of 4096, already spreads
over 4 sets — and it is the only shape that does not recover. Adding 64 B makes the advance
`(Mpad/64 + 1) mod 64 = 1`, which walks all 64 sets. The same arithmetic at L2 (512 KB, 8-way, 1024 sets)
gives 8 reachable sets for `q/o` and `down`, 16 for `gate/up`, 64 for `k/v`, and
`gcd(stride/64 + 1, 1024) = 1` for all of them after the pad.

> **Which shapes move and which does not was predicted from the stride arithmetic, and the measurement
> matches 4 for 4.**

### What this means for the programme

**D5's donor-width "fall past L3" is substantially cache-set aliasing, not capacity.** The Controller's
suspicion in A1.5 is confirmed: *every* `Mpad` in the existing sweep is a power of two or a multiple of 4096,
so set-conflict eviction was never separated from capacity, and the two rounds of work that tried to settle
compute-vs-memory at donor width were both partly measuring a layout artefact.

**It is worth 1.63–1.68x at 6 threads on the shipped byte packing, with no packing change at all** — the
whole intervention is an allocation offset and one extra argument. That is a larger, cheaper win than P1.

## S2.5 The A1.3 discriminator — matvecs/s and moved GB/s, both arms

A1.3 registers: bandwidth-limited => matvecs/s roughly doubles, moved GB/s roughly invariant. Compute- or
port-limited => matvecs/s roughly invariant, moved GB/s roughly halves.

**Read the `+64` rows.** The `+0` rows are contaminated by the aliasing above, and — critically — **the
contamination is not symmetric between arms**: the nibble arm has half the planes, hence half the
set-conflicting loads, so it suffers less from the artefact. That is why `gate/up` at `+0` reports a
physically impossible **4.127x** matvecs/s ratio. **A 2x byte reduction cannot buy 4x. That number is an
aliasing artefact and appears here only so it is not mistaken for a result.**

### Streamed, stride +64 (aliasing removed) — the readable cells

| shape | thr | **matvecs/s ratio (nib/byte)** | range over 5 inv | **moved GB/s ratio** | range over 5 inv | reading |
|---|---|---|---|---|---|---|
| `q/o` 8192x8192 | 6 | **1.912x** | 1.209–2.925 | **0.956x** | 0.604–1.462 | **bandwidth-limited** |
| `gate/up` 28672x8192 | 6 | **1.734x** | 1.257–2.164 | **0.867x** | 0.629–1.082 | **jointly limited** |
| `k/v` 1024x8192 | 6 | **1.688x** | 0.806–4.397 | **0.844x** | 0.403–2.199 | jointly limited *(dispersion too large to lean on — see S2.6)* |
| `down` 8192x28672 | 6 | **1.581x** | 1.220–2.329 | **0.791x** | 0.610–1.164 | **jointly limited** |
| `gate/up` | 1 | 2.747x | 2.183–3.311 | 1.373x | 1.091–1.655 | **OFF-TABLE** (see below) |
| `q/o` | 1 | 3.286x | 2.575–4.100 | 1.643x | 1.287–2.050 | **OFF-TABLE** |
| `k/v` | 1 | 1.616x | 1.241–2.613 | 0.808x | 0.620–1.306 | jointly limited |
| `down` | 1 | 1.474x | 1.206–2.054 | 0.737x | 0.603–1.028 | jointly limited |

**At 6 threads with aliasing removed: matvecs/s 1.58–1.91x, moved GB/s 0.79–0.96x.**

> **This is an INTERMEDIATE result and it is reported as one.** It is not rounded to a pole. Halving the
> weight stream buys most, but not all, of a doubling, and the memory system moves slightly fewer bytes per
> second than before. **The path is jointly limited, leaning bandwidth.**

### A third outcome A1.3's table does not contain, and what it means

Four cells show a **moved GB/s ratio above 1.15** — the nibble arm moves *more* GB/s than the byte arm.

> **You cannot beat a byte ceiling by moving fewer bytes if bytes were the binding constraint.** A ratio
> above 1 falsifies "bandwidth-limited" for those cells outright. It means the byte arm was limited
> **per-ACCESS**, not per-byte.

This exposes a **confound in P1's own design that the brief does not acknowledge, and that I should flag
plainly: P1 halves the bytes AND halves the load count, simultaneously and inseparably.** A contiguous
2 bits/weight layout necessarily issues half as many 32 B loads. So "matvecs/s roughly doubles" is predicted
by *two* different hypotheses — bandwidth-limited, and load-issue/latency-limited — and P1 cannot tell them
apart on the matvecs/s column alone.

**The moved GB/s column is what saves the experiment**, and it is the reason A1.4's convention was worth
insisting on. Where that column sits near 1.0 with matvecs/s near 2 (`q/o` t6), bandwidth is the honest
reading. Where it climbs above 1 (t1 on the large shapes), the byte arm was access-limited and A1.3's table
simply does not apply.

**P1 is therefore a weaker discriminator than Amendment 1 hoped.** It is still the cleanest one this
programme has run — and it found the aliasing, which neither prior round could have.

## S2.6 Dispersion, stated rather than buried

| cell class | between-invocation CV | within-invocation CV | trust |
|---|---|---|---|
| large shapes (`q/o`, `gate/up`, `down`), **+64**, t1 | **8.8–14.6%** | **9.0–18.3%** | **best rows in the sweep** |
| large shapes, **+64**, t6 | 8.7–19.9% | 10.1–42.2% | good |
| large shapes, **+0** | 9.2–28.1% | 9.1–25.7% | usable but aliasing-contaminated |
| `k/v` t1 (any stride) | 6.5–19.5% | 13.1–31.8% | usable |
| **`k/v` t6 (any stride)** | **17.8–33.0%** | **56.5–136.7%** | **do not lean on these** |

`k/v` at 6 threads has a within-invocation CV above 100%: the per-rep distribution is wildly skewed, because
a `k/v` matvec is only ~100–240 us and OpenMP fork/join jitter is a large fraction of that. Those cells are
reported for completeness and **excluded from every conclusion drawn above**. The cells the verdict rests on
(`q/o`, `gate/up`, `down` at +64) are the tightest in the sweep.

The wide ratio ranges (e.g. `k/v` t6 `+64`, 0.806–4.397) are the product of two noisy cells divided, not
evidence of a 4x effect; the mean-of-means is the estimator and the range is disclosure.

## S2.7 Prior art — no novelty is claimed

**`bitnet.cpp`'s TL1 already uses this alphabet and this density.** arXiv:2410.16144v2, Table 2: the same
`(w0+1)*3+(w1+1)` two-trit code at **2.0 bits/weight**. **This is a defect in our engine that we are fixing,
not an invention, and no novelty claim enters any document from this work.** Per the brief's own section 7,
that is a perfectly good outcome — fixing it is worth the same either way.

**The brief's "2 trits/nibble is the `pshufb` optimum" phrasing is withdrawn (Amendment 2) and is not
repeated here.** `bitnet.cpp`'s TL2 reaches **1.667 bits/weight** using a sign bit plus a 4-bit index, so
2.0 bits/weight is not an optimum of anything. **P1 is a catch-up, not a frontier.**

## S2.8 Recommendations, in priority order

1. **Adopt the +64 stride pad on the byte arm now.** 1.63–1.68x at 6 threads on three of four donor organs,
   bit-exact, no packing change, one allocation offset and one argument. **It is independent of P1 and
   larger than P1.** It also means every donor-width bandwidth number this programme has taken should be
   re-read as partly an aliasing measurement.
2. **Then adopt nibble packing.** Stage 1 is green and unconditional: the 2x is bit-exact and banked.
   Stage 2 says it buys a further 1.58–1.91x matvecs/s once aliasing is out of the way.
3. **Before engine-wide adoption, convert `matvec_lut_full_K`.** The block-decode weight-once path has no
   nibble variant; the engine currently hard-errors on `--g3c --pack nibble` rather than mixing layouts.
4. **Do not re-litigate compute-vs-memory with another shape sweep.** Every `Mpad` in the existing evidence
   base is a power of two or a multiple of 4096, and the set-index arithmetic above shows what that does.

## S2.9 Amendment 2 A2.4 — the `int16` vs `int8` accumulator, analysed not asserted

**I have not read TL1's kernel.** What follows is the analysis of *our* accumulator and of what an
int16-first variant would cost, so the comparison can be closed cheaply by whoever does read it. No timing
number was taken for this: the machine has been contaminated since 19:34:40 (S2.0).

**What ours does.** `acc_add_i8x32` takes the `pshufb` result — 32 `int8` partial products — and
sign-extends **straight to `int32`**: 2 lane extracts, **4x `_mm256_cvtepi8_epi32`**, **4x
`_mm256_add_epi32`**. That is ~10 vector ops per `pshufb`, and it runs **once per tile `t`**, i.e. 4096
times per `k/v` matvec.

**Saturation headroom, derived.** LUT entries are `s = w0*x0 + w1*x1` with `w in {-1,0,1}` and `|x| <= AQ =
63`, so `|s| <= 126` — which is why the `int8` LUT is safe in the first place. An `int16` accumulator holds
`32767`, so it can absorb `floor(32767/126) = 260` tiles before it can possibly overflow. **An int16-first
scheme is therefore viable on our data with a widening flush every 256 tiles** — a power-of-two flush
interval with 4 tiles of margin.

**What that would buy.** Per tile it replaces 4 `cvtepi8_epi32` + 4 `add_epi32` with 2 `cvtepi8_epi16` + 2
`add_epi16`, deferring the int32 widening to once per 256 tiles: **roughly a halving of the accumulate-side
op count**, amortised.

**Why this matters for reading Stage 2 rather than for P1.** Our accumulator is on the *arithmetic* side of
the compute-vs-memory balance. **S2.5's "jointly limited, leaning bandwidth" verdict was measured with this
heavier accumulator in both arms.** A lighter one would move the balance *toward* bandwidth-limited, not
away from it — so it cannot rescue a compute-bound reading, and it does not threaten the S2.4 aliasing
finding at all (that is a pure address-arithmetic effect, identical under any accumulator).

**It is an unexploited lever that is independent of P1**, sits alongside the +64 stride pad as a
packing-neutral win, and should be costed with the same protocol rather than folded into this probe.

## S2.10 Reproducibility manifest — Stage 2

**Compiler.** `clang 21.1.8` (llvm-mingw, msvcrt), target `x86_64-w64-windows-gnu`.
**Flags.** `-O3 -mavx2 -mfma -march=znver2 -fopenmp -Wall`. **No `-ffast-math`.**
**Machine.** AMD Ryzen 5 3600X, 6C/12T, 3793 MHz, L1d 32 KB 8-way, L2 512 KB 8-way, L3 16 MiB per CCX.
Windows 11 Pro 10.0.26200, 80 GB RAM. **Condition during the run: section S2.0, from my own sampler.**
**Repo state.** `565b4ddd8e2a2fb5014b2942b8bf4069aa4118cc`, branch `research/donor-adaptation`. Not committed.

**Files added for Stage 2 (working tree, uncommitted):**

- `benchmarks/donor_adaptation/nibble_pack_bench.c` — the timed harness (contains the bit-exactness guard)
- `benchmarks/donor_adaptation/p1/sample_machine.ps1` — the independent machine sampler
- `benchmarks/donor_adaptation/p1/run_stage2.sh` — driver: starts sampler, 5 invocations, stops sampler
- `benchmarks/donor_adaptation/p1/aggregate_stage2.py` — aggregation, mean-of-invocation-means

**Exact commands.**

```sh
clang -O3 -mavx2 -mfma -march=znver2 -fopenmp -Wall \
      benchmarks/donor_adaptation/nibble_pack_bench.c -o bin/nibble_pack_bench.exe -lm

sh benchmarks/donor_adaptation/p1/run_stage2.sh      # sampler + 5 invocations
#   inner loop, per invocation inv = 1..5:
#   ./bin/nibble_pack_bench.exe --inv INV --target-gb 20 --pool-mb 512 --seed 0x100INV

python benchmarks/donor_adaptation/p1/aggregate_stage2.py \
     > benchmarks/donor_adaptation/p1/results/stage2/STAGE2_AGGREGATE.md
```

**Raw artefacts.** `benchmarks/donor_adaptation/p1/results/stage2/stage2_raw.csv` (200 timed cells, one row
per cell per invocation), `machine_samples.csv` (494 samples), `stage2_raw.err` (empty),
`STAGE2_AGGREGATE.md` (the full aggregation, including every cell not reproduced above).
