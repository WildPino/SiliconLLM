# BRIEF E48 — WHERE, INSIDE THE ENGINE, DOES THE CONTEXT SLOPE LIVE?

**Status: PART 1 IS POST-HOC DESK ANALYSIS OF DATA ALREADY IN THE RECORD. IT CARRIES NO GATE.**
E14 §6 forbids promoting a post-hoc metric to a gate, and this brief does not. Part 1 produces a
decomposition and a mechanism; part 2 registers the run that can refute it, with its gates and my
prediction written before any cell exists.

## 1. Why this question, and why it is answerable today

E46 measured that per-token cost is **linear in context position** and that the 10B target is
**context-limited at `C50 = 575` tokens**. E47 measured that the marginal cost is **MIXED** — both
query heads and KV heads move the slope. Neither says *which part of the engine* the slope is in,
and without that there is no lever.

Two things make it answerable without new data:

1. **A structural fact, read out of the source** (`donor_engine.c`). Every `pos`-dependent loop in
   the entire forward pass is inside the attention block, lines 1099–1176: the Q·K loops
   (1099–1135), the softmax passes (1148–1154), and the A·V loops (1164–1176). Everything else
   loops over *dimensions* — `T`, `H`, `nr`, `M`, `SL` — never over position. The KV cache write is
   a `memcpy` of `KVO*4` bytes (1071–1072), constant in `pos`. **So 100% of the context slope is
   the attention organ, by construction and not by measurement.** This is E46 §5's owed reading,
   done.

2. **E5 run 6 already measured the attention organ at two context lengths.** `--bench 300` (mean
   position 150.5) and `--bench 800` (mean position 400.5), five runs each, on `T10`
   (`L=48, NH=32, NKV=8, HD=128`, ~10.6B). It decomposed the organ into `X` (Q·K), `S` (softmax),
   `Y` (A·V) and `P` (residual), all bit-identical (G2 PASS, `166667.1361128952`) and all
   validated by a 3× out-of-sample test that E5 could have failed nine times and did not (worst
   error −1.15% against a 3% tolerance fixed before run 1).

E5 read those two cells as *levels*. **E46's linearity result licenses reading them as a slope** —
and that composition is the whole of part 1. Each term is also linear in `pos` by construction
(`X`, `S` and `Y` are `O(pos)` loops; `P`'s thread-imbalance component is a multiple of them).

## 2. The decomposition (post-hoc, no gate)

Source: `benchmarks/donor_adaptation/results/e5/analysis6.txt`, `T10` @300 and @800, medians over
the runs that passed G0. `P′ = P − X`, because E5's `P` contains `X` by construction (`R = S+Y+P`
with `X` reported separately).

| term | @300 (ms) | @800 (ms) | slope (ms/pos) | **share of the organ's slope** |
|---|---|---|---|---|
| `X` — the Q·K loop | 0.713 | 2.253 | 6.1600e-03 | **16.9%** |
| `S` — the softmax pass | 1.048 | 2.953 | 7.6200e-03 | **20.9%** |
| `Y` — the A·V loop | 0.781 | 2.406 | 6.5000e-03 | **17.9%** |
| `P′` — the non-loop residual | 1.530 | 4.064 | 1.0136e-02 | **27.9%** |
| the organ outside `R` | 0.728 | 2.290 | 6.2480e-03 | **17.2%** |
| **organ (`none`)** | 4.834 | 13.931 | **3.6388e-02** | 100% |
| sum of the five parts | | | 3.6664e-02 | 100.8% |

**The parts close on the whole to 0.8%**, which is inside E5's own run-to-run band on every term.

### 2.1 There is no single fix

**No part of the engine owns more than 28% of the context slope.** Deleting the softmax entirely
buys 20.9% of it; deleting both data loops buys 34.8%. Anyone promising to fix context cost with
one kernel is promising at most a third.

### 2.2 Three consistency checks that could have failed and did not

None of these was arranged; all three are checks on the decomposition, not inputs to it.

* **`Y/X = 1.055`.** The Q·K loop and the A·V loop read the same number of bytes (`HD` floats per
  head per position) and do the same number of FMAs. They must cost the same. They do, to 5.5%.
* **The `expf` count.** `L·NH = 1,536` per position → 615,168 per token at mean position 400.5.
  E5 §14.8 reports **615,168**, arrived at independently from the disassembly.
* **Charged versus moved bytes, which is the law that killed the "39.7% anomaly".** `X` reads
  `L·NH·HD·4 = 786,432` **charged** bytes per position, which at 6.16e-03 ms is **127.7 GB/s** —
  **3.52× above `BW-CEIL = 36.30 GB/s` (E30)**. That is impossible from DRAM, so the loop must be
  getting ≥3.52× reuse. **GQA supplies exactly 4.00×** (`NH/NKV = 32/8`): each KV head's slice is
  re-read once per query head that shares it. Crediting it, `X` **moves** 196,608 B/pos =
  **31.9 GB/s = 88% of `BW-CEIL`**.

### 2.3 What that buys: which levers are bandwidth and which are not

**`X` and `Y` are bandwidth-saturated and `S` and `P′` are not.** At 88% of the measured ceiling,
the two data loops cannot be made faster by better code — only by **moving fewer bytes**: fewer KV
heads, or a narrower KV dtype (the cache is **fp32**, `xmalloc(L*maxseq*KVO*4)`, line 965–966 — no
quantization anywhere in it). Together they are **34.8%** of the slope.

The other **65.2%** — softmax, the non-loop residual, and the organ outside `R` — is not bandwidth
and will not move for any KV compression. This is the answer to E46 §5, and it is the reason E47
came back MIXED rather than KV-BOUND.

### 2.4 The two named, sized, attackable levers

* **`S`, 20.9% of the slope, and it is calling the wrong function.** E5 §14.8: `expf` compiles on
  this toolchain to `vcvtss2sd → callq exp → vcvtsd2ss` — **double-precision, scalar, with a
  `vzeroupper` in front of each call**, 615,168 of them per token at @800. A vectorised
  single-precision exponential is standard work with a known shape. E5 declined to run it because,
  unlike every arm in E5, **it is not value-preserving** and therefore needs its own parity gate
  rather than bit-identity. That is a cost, not an objection.
* **`P′`, 27.9% of the slope — the largest single piece — and nobody has named its parts.** E5
  §14.6 priced the one named candidate (the OpenMP fork/join: **1.4% of `P`**) and then wrote:
  *"E5 establishes its size and refuses to name its parts. Decomposing `P` is a new experiment with
  its own arms, and it is owed, not answered here."* It is still owed. From the source, the
  candidates are the 32-heads-over-6-threads imbalance, the per-head address arithmetic, the
  `out[]` initialisation and the `mx` reduction (lines 1086–1180).

### 2.5 A candidate for E47's superlinearity — offered as a candidate, not a finding

E47 left `q = 2.8–4.0` unexplained against the 2.00 that doubling query FLOPs predicts. This
decomposition narrows where it can live but does **not** close it:

* `S` scales with `NH` and not with `NKV` (one `expf` per head per position) → exactly ×2. Not it.
* `X` and `Y` charge `NH·HD` but move `NKV·HD`. Doubling `NH` doubles *charged* and leaves *moved*
  unchanged, so they should rise by **less** than 2×, not more. Not it, and it pushes the wrong way.
* `P′` is the only term whose per-head overheads could rise faster than `NH`, and it is also the
  term nobody has decomposed.

**So the superlinearity, if it is real, is most likely inside the one term that has never been
opened.** That is a reason to open it, and it is not evidence about what is in it.

### 2.6 Four limits on part 1, stated here and not in a footnote

1. **It is post-hoc.** The cells were run for another question. No gate in this brief is applied to
   part 1, and part 1 may not be cited as a gated result.
2. **Two points per term is a slope with zero degrees of freedom.** Linearity is licensed by E46
   and by the loop structure, not by these data.
3. **The shape is `T10`, not the target.** `T10` is `L=48, NH=32, NKV=8, HD=128` (GQA 4); the A10B
   R128 headline arm is `L=16, NH=32, NKV=2, HD=128` (GQA 16). The shares are `T10`'s. The GQA
   factor in §2.2 is 4.00× at `T10` and would be 16× at R128 — the *argument* travels, the
   *number* does not.
4. **Absolute ms/pos carry E5's own warning.** E5 §14.7 refused to publish absolute tok/s because
   run 6 drifted **+15.5%** on the organ against E4. Slopes here are taken **within run 6**, where
   that drift is common to both cells and largely divides out of a difference — but the ±5% law
   still applies to every absolute in the table. **The shares are the result; the milliseconds are
   the arithmetic.**

---

## 3. PART 2 — the confirming run, registered before it exists

Part 1 is a composition of two experiments neither of which was designed for it. Part 2 measures
the same decomposition **directly, at more than two context points, on the target shape**, with the
harness E46 already validated.

### 3.1 Method

`e48_slope_decomposition.py`. For each arm in `{none, qk1, qk2, sm1, sm2, av1, av2}`, sweep the
context windows `{40, 160, 640}` with `1280` held out, `REPS = 5`, `--threads 6`, arms interleaved
at **rep level** (E45 run 4's schedule) so that any machine drift hits every arm equally.

The sub-organ cost is the difference between the 2× and the 1× wrapper, never against `none`:
`X = dt(qk2) − dt(qk1)`, `S = dt(sm2) − dt(sm1)`, `Y = dt(av2) − dt(av1)`. E5 §13.4 establishes
these wrappers are bit-identical to `none` and that the 1× form exists precisely so the wrapper's
own code shape is not charged to the loop.

Two shapes: **`E47-BASE`** (cheap, already built, GQA 8) and the **A10B R128 `--carve-k 3`**
headline arm (GQA 16, the shape `C50 = 575` was measured on).

### 3.2 Gates

* **`G-E48a` — the planted control. The instrument must fire before its nulls count.** Each of
  `X`, `S`, `Y` must be **positive at every window** and must **rise with context** by more than
  1.05× from the 40-window to the 640-window. A sub-organ probe that cannot see context in a loop
  that is `O(pos)` is not measuring the loop. If `G-E48a` does not fire, E48 part 2 reports no
  decomposition.
* **`G-E48b` — CLOSURE, and this is the gate that matters.** Fit each of `X`, `S`, `Y` and the
  residual against mean position. Their slopes must sum to the **independently measured** total
  slope `b` for that shape — for A10B R128 k3, E46's `b = 2.0307e-05 s/tok/pos`, obtained by a
  completely different route (a whole-token fit, no wrappers, on another day). **Tolerance ±15%.**
  The parts must add up to a whole that was measured without them. If they do not, the
  decomposition is wrong and part 1 goes with it.
* **`G-E48c` — the shares, which is the deliverable.** Report each term's share of the total slope
  with its dispersion across reps. **No threshold and no verdict**: this is a SCORE, and per E14 §3
  its RANK partner is `G-E48d`.
* **`G-E48d` — the RANK partner.** The *ordering* of the four terms by share must be identical
  between the two shapes, or the ordering is a property of the shape and not of the engine, and no
  lever may be prioritised from it.

The ±15% on `G-E48b` is set against **measured** dispersion, per the law this programme broke twice
in one day: E43 measured 9–22% intrinsic per-cell oscillation on this engine, and `G-E48b` is a sum
of four such quantities compared against a fifth. A tolerance tighter than that could never fire;
one looser than ~25% would be passed by noise. It is **not** set to my expected answer.

### 3.3 My prediction, registered

Written before any cell of part 2 exists, so it can be scored:

| quantity | prediction |
|---|---|
| `G-E48a` | **fires** on all three terms, both shapes |
| `G-E48b` closure at A10B R128 k3 | **passes**, within 10% |
| largest share at R128 | **`P′`**, 25–35% |
| `S` share at R128 | 15–25% |
| `X + Y` share at R128 | **lower than `T10`'s 34.8%** — GQA 16 instead of 4 means four times the reuse on the same moved bytes |
| `G-E48d` ordering | **holds** |

My last three registered predictions scored half-right, one-of-three and zero-of-three. This one is
recorded on the same terms.

### 3.4 What part 2 cannot do

It cannot open `P′`. It measures `P′` as a residual, exactly as E5 did, and inherits E5's refusal to
name its parts. **Decomposing `P′` is E49 and it is owed.** Part 2's only claim on `P′` is a number
and a rank.
