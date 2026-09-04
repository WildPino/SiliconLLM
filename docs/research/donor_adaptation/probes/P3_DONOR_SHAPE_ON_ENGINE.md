# P3 — The first donor shapes to touch engine.c

**Date: 2026-09-04. Machine: 3600X reference, generic `x86-64-v3`, `OMP_PLACES=cores`,
`OMP_PROC_BIND=close`.**
**Raw: `benchmarks/donor_adaptation/speed/results/p3_donor_shape.log`.**
**Apparatus: `benchmarks/phase60/engine.c`, `--donor-shape <model>` (new flag; no existing path
modified; kernel selftest green, 73,024 checks, worst 0).**

> **Not audited.** Written by the figure that built the arm. A Controller pass is owed.
> **This is not a port.** See §4 before quoting anything from it.

---

## 1. What was measured, and the prediction it was tested against

Every number in `SPEED_LEDGER.md` is arithmetic. This is the first measurement in which a **donor's
actual shapes** are driven through **`engine.c`'s actual kernels** (`build_lut_t3` +
`matvec_lut_full`), with the **entire ternary-packed weight stack resident in DRAM** — 0.72 GB for
Qwen2.5-1.5B, far beyond the 32 MB aggregate L3, so the traffic is real streaming and not an L3
replay.

**The prediction was written into the bench's own source and into the log header before the first
run**, so this tests the ledger instead of confirming it:

> `SPEED_LEDGER.md` predicts **24.0 tok/s** for Qwen2.5-1.5B at 4-bit, of which 33.60 ms is FFN,
> 5.24 ms attn+head, 2.80 ms KV.

## 2. The result

Shapes read from each donor's own `config.json`. All seven matrices per layer (q, k, v, o, gate, up,
down) at every layer, plus the output head. 4-bit codes, 0.5 B/weight.

| donor | packed stack | t1 tok/s | **t6 tok/s** | t6 GB/s | head share | t1→t6 |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B** | 0.23 GB | 28.8 | **81.0** | 20.01 | 25.5% | 2.81× |
| **Qwen2.5-1.5B** | 0.72 GB | 10.7 | **30.7** | 23.69 | 16.4% | 2.87× |
| **Qwen2.5-3B** | 1.44 GB | 4.8 | **15.1** | 23.23 | 11.0% | 3.15× |

**Qwen2.5-1.5B: 32.57 ms of weight-matvec per token at t6.** Adding the ledger's KV term (2.80 ms at
ctx 4096) gives **35.37 ms = 28.3 tok/s**, against a predicted 24.0.

> **The ledger was 18% CONSERVATIVE, and the reason is diagnostic.**

## 3. Why the ledger was wrong, and in which direction

`SPEED_LEDGER.md` priced the FFN at the **expert path's** kernel-pure rate — 17.0 GB/s, measured in
64.1b on an **i.i.d. gather** over a 512 MB pool of 48 KB experts. But a **dense donor FFN is read
contiguously and sequentially**, not gathered. P2's own arm S already showed sequential beats random
(2.690 vs 2.857 µs/expert); P3 shows the effect is larger at donor shapes, where the contiguous runs
are megabytes rather than 48 KB.

Measured aggregate: **23.2–23.7 GB/s at t6** on the two larger donors, stable across a 2× size
change — above the 17.0 GB/s gather rate, well below the 40–44 GB/s DRAM ceiling, which is where a
kernel with a real compute floor (P2: ~60% of the expert path is arithmetic) should sit.

**Correction owed to the ledger:** using a *gather* rate to price a *dense contiguous* organ is
wrong, and it is wrong in the safe direction. A carved MoE donor **would** pay the gather rate; a
dense one does not. The two must not share a row.

## 4. ⚠ What this is NOT — read before quoting

This measures **weight streaming and multiplication**, the dominant per-token term, and **nothing
else**. Absent: attention itself (scores, softmax, the KV read), RoPE, RMSNorm, residuals, the SwiGLU
non-linearity, sampling, and detokenisation. The weights are **pseudo-random, not the donor's**, so
no output is meaningful and no BPB is implied — the checksum column exists only to stop the compiler
eliminating the work.

**`engine.c` still cannot execute a transformer.** It has no attention organ, no RoPE, no KV cache.
This bench drives its matvec kernels at a transformer's dimensions; it does not run one. **The
correct reading is: "this is the floor a real port would build on", not "the donor runs at 30.7
tok/s".** A real port is strictly slower.

## 5. What it changes

**5.1 Qwen2.5-0.5B already clears the "good" bar on the matvec term** — 81.0 tok/s, dense, no carve,
no quality damage, with today's 4-bit packing. That is a real datum about a real pretrained model,
and it is the first time this programme has had one.

**5.2 The head is a first-class term and it grows as donors shrink** — 25.5% of the per-token time
at 0.5B, 16.4% at 1.5B, 11.0% at 3B. `SPEED_LEDGER.md` §2 flagged that no probe has ever touched the
output head; this measures the bill. At 0.5B, **a quarter of every token is spent projecting to a
151,936-word vocabulary.**

**5.3 Thread scaling is 2.81×–3.15×**, above the ~2.4× that 63.T recorded as "the memory-bound
ceiling" for the engine's own model. Consistent with P2: there is more arithmetic in this mix than in
a pure streaming workload, so more of it parallelises.

**5.4 The extrapolation to the goal, stated as arithmetic and marked as such.** t6 aggregate is
~23.5 GB/s and stable across 1.5B→3B. A **dense** 10B at 0.5 B/weight is 5 GB/token → ~213 ms →
**~4.7 tok/s**. A 10B at **10% activation** (the ratio Qwen3-30B-A3B ships with) is ~0.5 GB/token →
~21 ms → **~47 tok/s** if read contiguously, and ~34 tok/s at P2's gather rate, which is what a
routed MoE actually pays. **Neither is measured. Both are the same division this document just found
the ledger doing wrong**, and the honest statement is that the goal sits within a factor of ~1.5 of
this machine's measured throughput for an already-sparse 10B donor, and roughly 10× away for a dense
one.

## 6. Reproduce it

```bash
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp benchmarks/phase60/engine.c -o bin/engine_p3.exe -lm
OMP_PLACES=cores OMP_PROC_BIND=close ./bin/engine_p3.exe --kselftest
OMP_PLACES=cores OMP_PROC_BIND=close ./bin/engine_p3.exe --donor-shape qwen2.5-1.5b
```

`--donor-shape` accepts `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`; it defaults to 1.5B. Synthetic
weights, no model download, ~1 GB of RAM for the 3B stack, seconds of wall clock. rep 0 is a warm-up
and is discarded; reps 1–2 are averaged.

## 7. Owed

- **A Controller audit.** Built and read by the same figure.
- `SPEED_LEDGER.md` must stop pricing dense FFNs at the gather rate (§3). Amendment pending.
- The **proj-path decomposition** P2 left owed is now more pressing: P3's attn+head figures come from
  the same kernel and this bench can host that arm.
- The real gap remains what §4 says it is: **attention, RoPE, and the KV cache do not exist in
  `engine.c`.** Until they do, no donor runs — only its shapes.
