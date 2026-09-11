# E26 — what does an ACTIVATED weight cost?

**Pre-registered.** This file is pushed before the runner exists and before any timing is taken.
Nothing below §3 is written after a number is known; §3 is filled in from the runs.

Predecessors: `probes/E19_DOES_THE_CARVE_RANK.md`, `probes/E23_A_REAL_ROUTER.md`,
`probes/E25_WHAT_THE_RANK_COSTS.md`, `SPEED_LEDGER.md` §31 and §37.

---

## 0. Why this exists, in one paragraph

E25 measured the goal's shape for the first time: `T10` — 10.60 G active ternary weights per
token, the dimensions of the goal's "es 10B" — runs at **4.70 tok/s** on this box, against a
target of 50. E25 also found the invariant that makes the rest of the programme legible: four
`T10` arms whose rates differ by 15% deliver **charged throughput inside a 1.54% band**
(49.59–50.36 G active weights/s). The engine converts active weights into time at a rate that
does not care how they are *arranged* — dense or factored, one matvec call or two. **Every
budget table in this programme rests on that invariant holding for the next lever too, and the
next lever is the carve.** The FFN is `8.4557 G` of `T10`, **79.7%** of the model and nine tenths
of the remaining gap, and E19's carve and E23/E24's router are both aimed at exactly it.

But E19 measured the carve with a PyTorch forward-pre-hook that **zeroes** the non-kept
activations. Zeroed weights are still read, still multiplied, still streamed. **No carve has ever
skipped a byte in this engine, and `engine.c` has neither a router nor a gated FFN.** So the
load-bearing, never-tested assumption is:

> **an ACTIVATED weight costs what a DENSE one costs.**

That assumption is not obviously true. A dense FFN is three long sequential streams. A carved
FFN is a *gather*: scattered rows of `gate`/`up`, and — worse — scattered **columns** of `down`,
which in the `[D, F]` packed layout are half a byte each inside a 64-byte line. If gathering
costs more per weight than streaming, every `k` in every budget table here is wrong in the
optimistic direction, including E23 §7's `k = 139…156`.

---

## 1. The format obstacle, measured before it was designed around

D0c's partition (`density/results/d0c_labels/labels_E256.npz`, `E = 256` at Qwen2.5-1.5B) reads:

```
group sizes: min 35 max 35 mean 35.0  n=256
odd-sized groups: 256 of 256
contiguous groups: 0/256
first 40 labels: [175 220 249  81   4  26   6 235  40  43 214 170 201  20 118  29 126 160 ...]
```

**Every group is 35 neurons and not one of the 256 is a contiguous index range.** Two consequences:

1. `gate`/`up` are `[F, D]`, so a kept neuron is a whole **row** — contiguous regardless of
   scattering. A 35-neuron group is 35 rows at scattered addresses, each `D/2` bytes
   (768 B at `S15`, 2048 B at `T10`). Skipping rows is easy and the only question is what
   scattering costs the prefetcher.
2. `down` is `[D, F]`, so a kept neuron is a **column** — half a byte, strided by `F/2`. A
   scattered carve saves *nothing at all* on `down`: every kept neuron still pulls a full
   64-byte line that carries 128 neighbours. And a 35-neuron group straddles packed byte
   boundaries, because `down` packs two trits per byte **along `F`**.

Both are fixed by two changes that are **exact** — no approximation, no re-quantization:

- **Permute the neuron order** so every group is a contiguous run. A permutation of the `F` axis
  is a permutation of the rows of `gate`/`up` and the columns of `down`; the model it computes is
  unchanged to the last bit. The permutation is baked into the export and the engine never sees
  a label.
- **Store `down` transposed** as `downT [F, D]`, packed two trits per byte **along `D`**, with
  its `fp32` scale array still indexed by `d` (length `D`) because `down` is ternarized per
  output row and transposition does not move a scale. A kept neuron is then a whole contiguous
  row of `D/2` bytes, the odd-35 problem disappears (parity now falls on `D`, always even), and
  `downT` is **exactly the same size** as `down`: `F·(D/2) + 4·D` either way.

Transposing changes the accumulation: `y[d] = scale[d] · Σ_f codeT[f][d] · h[f]` is an
accumulate over kept rows into a `D`-float accumulator, not `D` independent dot products. Same
trits, same FMA count, different summation partition. Phase 60's law applies: **this is gated on
end-to-end parity, never on sha256.**

---

## 2. What will be built (part A)

### 2.1 `donor_engine.c`

- **`quant == 4`** — tagged-v2. Byte-identical to `quant == 3` (E25's tagged container, every
  matrix carrying its own `int32` kind) **except** that each layer's FFN is preceded by an
  `int32 ffn_kind`. `quant == 3` files are frozen and must load unchanged.
  - `FK_DENSE = 0`: `gate [F,D]`, `up [F,D]`, `down [D,F]`, exactly as today.
  - `FK_CARVED = 1`: `int32 E`, `int32 k_default`, tagged `router [E,D]`, then
    `gate [F,D]`, `up [F,D]` in **group-major permuted** row order, then `downT` as a new
    matrix kind.
- **`MK_PACKED_T = 3`** — a packed matrix stored transposed: `code[in][out/2]`, `scale[out]`,
  logically `[out, in]`. Used only for `downT`.
- **`matvec_rows(m, x, bias, y, rows, nr)`** — the existing packed kernel with its `o` loop
  walking a row list instead of `0…n_out`. One OpenMP region, as before.
- **`matvec_colacc(m, h, rows, nr, y)`** — the transposed-`down` kernel. One OpenMP region with
  a manual split of the `D/2` accumulator range across threads (each thread owns a private,
  contiguous slice, so there is no reduction), the `t` loop *inside* the region so a carve opens
  one region per call and not `nr` of them.
- **top-`k` router**: `r = W_r · x` over `E` groups, partial selection sort, `k ≥ E` short-circuits
  to "keep everything". Ties resolve to the lower index, and the reference mirrors that.
- **`--carve-k K`** — runtime override of `k_default`, so **one artifact serves every `k` arm**
  and no cell in the sweep can be confounded by a different file.
- **`--carve-dump`** — print the selected group ids per layer, so the selection can be gated
  directly instead of only through logits.

### 2.2 Exporters and the size model

`synth_export.py` gains `--carve E` (writes `quant == 4` with `FK_CARVED`, a random permutation,
a random router); `qwen_export.py` gains `--carve-labels <npz>` (the D0c partition, turned into a
group-major permutation) and `--router <npz|random>`. `e1_bpb_through_engine.py` gains
`nbytes_tagged("packedT", …)` and a `layout_bytes_v4(...)` derived **from the format description
and never from the writer** — that separation is the whole value of the size gate.

---

## 3. Result of part A — every gate fires, and part B is BLOCKED on an idle box

| gate | result |
|---|---|
| `G-E26a` | sha256 `c638214ef1661fcc…` on a `quant == 2` file and `7130ac5ea0346955…` on E25's `quant == 3` `QO512-TB`, **identical on the pre- and post-patch engines**. |
| `G-E26b` / `G-E26L` | `GATE V3` on all five synthetic arms; `GATE E26-L` on the real export, `1,597,287,796` bytes, against E1's independent `layout_bytes_v4`. |
| `G-E26c` | `layout OK: consumed exactly 1597287796 bytes`. |
| **`G-E26P`** | worst relative l2 **`4.481e-06`** (bar `1e-4`), top-1 **`1.0000`** on 8/8. **FIRES.** |
| **`G-E26S`** | **224 / 224 selections IDENTICAL** (8 positions × 28 layers). **FIRES.** |
| **`G-E26R`** | worst relative l2 **`1.156e-06`** (bar `2e-3`), top-1 **`1.0000`**. **FIRES.** |

The artifacts are Qwen2.5-1.5B at the pinned revision, `--rule R0 --fold none`, exported twice:
once dense (`quant == 2`, `1,591,752,756` bytes) and once carved (`quant == 4`, `E = 256` from
D0c's labels, `1,597,287,796` bytes). The reference dequantizes through **`qwen_export.quantize`
itself** and builds the router through **`carve_common.router_weights`**, the same function the
exporter called — neither side reads the artifact under test.

**Disclosure on `G-E26R`**: with a *random* router at `k = 64` the model is destroyed and both
sides predict `<|endoftext|>` at 7 of 8 positions, so top-1 agreement is cheap there. The
load-bearing numbers are the relative l2 over the full 151,936-dim vector and `G-E26S`, which
compares sets and cannot be passed by agreeing on garbage.

**Two layout decisions were forced by measurement, and both are recorded as tuning, not as
findings.** The first cut stored the transposed `down` row-major and read the byte-neutral
control at **−17%**; block-major storage (`[(D/2)/64][F][64]`, the move E13 made on the LUT
path) recovered most of it. `PT_BLK = 64` was chosen against 32 on `S15` (20.8 vs 17.6 tok/s at
`k = E`, 45.2 vs 36.6 at `k = 64`) and is now a **format** constant.

### 3.1 Part B ran, and the run is VOID

The first part B attempt completed all 12 arms × 3 reps in 363 s and **none of it counts**.
`S15-DENSE` read **16.21 tok/s** where E25 read **29.50** on the same shape, `T10-DENSE` read
**2.55** where E25 read **4.70**, and the dispersions were 20–39%. The box was running a game:
**5.82 of 12 logical cores were busy.** A contended timing is not a timing, and this one is
kept only as a witness, at `engine/results/e26_carve_cost_contended.json`.

So idleness is now an **instrument** rather than an intention. `e26_carve_cost.py` samples
system-wide CPU before the run and after every repetition (`Win32_PerfFormattedData_PerfOS_Processor`,
chosen because the `Get-Counter` path is *localized* and does not exist on this box's Italian
Windows), **refuses to start above 12%**, and stamps any record whose witness peaks above the
bar `VOID_AS_A_TIMING`. It fired on its known-positive the moment it existed: `63% busy →
REFUSING`. Part B is **owed and unrun**; §§4–8 stand exactly as pre-registered.

---

## 4. Gates on the construction

| gate | what it holds | bar |
|---|---|---|
| `G-E26a` | legacy paths unmoved: the pre-patch and post-patch engines give **bit-identical** logits on a `quant == 2` file **and** on E25's `quant == 3` `QO512-TB` artifact | sha256 equal |
| `G-E26b` | the writer's byte count equals `e1_bpb_through_engine`'s independent model, on every arm | exact |
| `G-E26c` | the reader consumes exactly the file | `layout OK: consumed exactly N bytes` |
| **`G-E26P`** | **`--carve-k E` equals dense.** The carved artifact with every group kept, against the DENSE artifact built from the same weights, end to end on the real donor. This validates the permutation, the transposed packing, the transposed kernel and the row-selection machinery at full width, all at once | rel l2 `< 1e-4`, top-1 `1.0` on 10/10 |
| **`G-E26S`** | **the selection is faithful.** `--carve-dump` at `k < E` against a PyTorch top-`k` on the same router scores | selected sets **identical**, every layer, every position |
| **`G-E26R`** | **the carve is faithful.** Engine at `k < E` against a PyTorch reference running the same permutation, the same ternarization and the same carve | rel l2 `< 2e-3`, top-1 `1.0` on 10/10 |

`G-E26P` is the planted control's **correctness** half and `G-E26S` is its **selection** half.
Both must fire before any timing counts (`feedback_planted_controls`).

---

## 5. Part B — the arms

Synthetic shapes (`synth_export.py`), `E = 256`, `q/o` left **dense** so the FFN term is the only
thing moving. Weights are noise: E3 §4's Gate V1 authorises that for **time** and for nothing
else. `F / E` is exact at both shapes — group size 35 at `S15`, 56 at `T10`.

| arm | `k` | % of `F` | active/token | byte prediction vs its dense control |
|---|---|---|---|---|
| `S15-PACKED` | — | 100% | 1.5436 G | untagged-container continuity control |
| `S15-DENSE` | — | 100% | 1.5436 G | — |
| **`S15-K256`** | 256 | 100% | 1.5546 G | **−0.71%** — the planted control |
| `S15-K128` | 128 | 50% | 0.9766 G | +58.06% |
| `S15-K64` | 64 | 25% | 0.6875 G | +124.51% |
| `S15-K16` | 16 | 6.2% | 0.4708 G | +227.88% |
| `S15-K4` | 4 | 1.6% | 0.4166 G | +270.53% |
| `T10-DENSE` | — | 100% | 10.6032 G | — |
| **`T10-K256`** | 256 | 100% | 10.6535 G | **−0.47%** — the planted control |
| `T10-K128` | 128 | 50% | 6.4257 G | +65.01% |
| `T10-K64` | 64 | 25% | 4.3117 G | +145.91% |
| `T10-K16` | 16 | 6.2% | 2.7263 G | +288.92% |
| `T10-K4` | 4 | 1.6% | 2.3299 G | +355.09% |

`active = (q + k + v + o + E·D + 3·D·(F/E)·k)·L + V·D`. The router is charged at `E·D` per layer
— `50.3 M` at `T10` — and that is why the `k = E` control is **not** exactly byte-neutral but
`−0.47%` / `−0.71%`: **the registered expectation for the control is that offset, and whatever it
loses BELOW it is the carve machinery's own cost** (the router matvec, the selection, the row
list, the gather, the transposed kernel's different access pattern).

---

## 6. Protocol

Idle box, `--threads 6`, `--fuse` **off** everywhere (a carved FFN cannot fuse `gate|up` into one
matrix and leaving it on for the dense control would compare different region counts), ≥ 3
repetitions **interleaved by rep**, dispersion printed with every rate. A contended timing is not
a timing. The `k` sweep runs off **one artifact per shape** via `--carve-k`, so within a shape the
only thing that changes between cells is an integer.

Every absolute tok/s carries ±5%; ratios inside one shape do not. `GB/s` figures are **charged**
bytes at the packed format's exactly 0.500000 B/weight, never moved bytes.

---

## 7. Predictions — fixed here, before the run

1. **`G-E26P` fires.** The permutation and the transposition are exact; if this misses, the
   kernel is wrong, not the idea.
2. **The planted control reads within ±3 pt of its `−0.47%` / `−0.71%` byte prediction at `T10`,
   and I am less sure at `S15`.** Registered alternative: if `k = E` loses more than **8%**
   against dense, the carve machinery is not free and every carved reading must be corrected by
   that offset before it is called a saving.
3. **The headline.** Charged throughput at `T10` stays on E25's line — every carved arm inside
   **45 – 55 G active weights/s**. Registered alternative, and the one that would matter most:
   if the deep arms (`K16`, `K4`) fall **below 40 G weights/s**, then a gathered weight costs
   more than a streamed one, the invariant is arrangement-dependent after all, and **every `k`
   in every budget table here — E18 §31, E19, E23 §7 — is optimistic.**
4. **`T10` saturates around 22 tok/s and the carve alone cannot reach 50.** With `q/o` dense the
   non-FFN floor at `T10` is `2.1978 G` (`k/v` `0.4027 G`, `q/o` `1.6106 G`, router `0.0503 G`,
   head `0.1342 G`), so at 49.9 G weights/s the `k → 0` asymptote is **22.7 tok/s**. Registered
   numbers: `T10-K64` ≈ **11.6**, `T10-K16` ≈ **18.3**, `T10-K4` ≈ **21.4** tok/s, each ±10%.
   This is E19's "FFN-only carving cannot reach the target" turned into a curve that can miss.
5. **The `S15` deep arms will disappoint relative to `T10`.** At `S15` a kept row is 768 B and at
   `T10` it is 2048 B; if scattering costs anything, it costs more where the runs are shorter.
   Registered as a direction, not a number.

---

## 8. What E26 will NOT be able to claim

- **Nothing about quality.** Part B's weights are noise and its router is random. Which `k` a
  real model survives is E19's and E24's question, and **E24 is still unrun**.
- **`T10` is a shape, not a model**, and it carries Mistral's 32,768 vocabulary, so its head
  (134 M) is the favourable case; a Qwen-vocabulary 10 B carries 622 M.
- **Nothing about the router's own accuracy.** The router here is charged and executed, never
  evaluated.
- **Nothing about training.** A permuted neuron order and a carved FFN are an inference-time
  layout; whether a donor tolerates the carve at all is the H0/T4 question, still waiting.
- **`6.79 tok/s` stays what it is** — the real 7.072 B packed donor, a different artifact.
