# E12 — does the ternarization cost shrink with donor scale?

**Pre-registered. Pushed before the runner exists.** The oldest open item in this programme, and
after E10 the only one that can still move the goal.

---

## 1. Why this, and why now

E10 closed the engine's weight path: the packed kernel is on its ceiling at every footprint, the
engine's FFN organs are already there, and ledger §19.4's 37 and 42 GB/s budget rows are withdrawn.
**The remaining 7.4× is a property of the model.** The only route to a 10 B at 50 tok/s that this
programme has ever had is **fewer active weights per token** — sparsity/MoE — at **ternary
precision**.

That route rests on a quality number, and **that number has only ever been measured at one shape.**
`probes/T2_TERNARIZATION_RULE.md:4` reads *"Donor: Qwen2.5-1.5B"*. The whole R0→R5 rule
investigation, the +3.309 → **+1.260 BPB** FFN result, and T2b's whole-model **+2.466** are one
point on a curve nobody has drawn.

**If the cost is flat or grows with scale, the strategy's quality premise is false and we would be
building a 10 B whose ternary damage is at least as bad as the 1.5 B's.** If it shrinks, the road is
supported by evidence rather than by hope. Either way it is the cheapest decisive thing left,
**and it is deterministic — it can be measured on a contended machine**, which the speed work
cannot.

## 2. What is measured

For each donor **D ∈ {Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B}**, all present offline and pinned to
the commit hashes already in the local cache (1.5 B's is the **same** hash `8faed761…` this
programme has used throughout, so the middle point is the historical one, not a re-derivation):

**ΔBPB(D) = BPB(D ternarized) − BPB(D fp32)**, on the same held-out corpus, same sequence count,
same organ set, and the **same conversion oracle** — `t1_ternarize.ternarize()`, which is the single
definition the exporter and the parity gate both call.

**ΔBPB is the comparable quantity; absolute BPB is not.** A 3 B is a better model than a 0.5 B and
will score lower on both arms; the question is only what the format costs each of them.

## 3. Controls

**Planted control (read first).** The **1.5 B cell must reproduce the published number at that
shape.** If a sweep cannot re-derive the point it already has, nothing else in it counts. This is
G-L2's discipline applied deliberately rather than after a void — E11's run 1 was caught exactly
this way, by a number carried in from outside the run.

**One variable.** Corpus, sequence count, rule, organ set and dtype are identical in every cell; the
donor is the only thing that changes. Each donor gets **its own fp32 baseline**, computed in the
same run — `t1_ternarize.py`'s `BASELINE_STANDING` constant is a **1.5 B** number and **must not**
be reused for the other two.

**No anchoring.** The runner prints ΔBPB per cell without a verdict; the bands below decide
mechanically.

## 4. Bands, fixed here

Let `r = ΔBPB(3B) / ΔBPB(0.5B)`.

| r | verdict |
|---|---|
| **≤ 0.75** | **COST-SHRINKS** — the format gets cheaper as the donor grows; the MoE road's premise is supported and the 1.5 B number is pessimistic |
| 0.75 – 1.25 | **COST-FLAT** — the 1.5 B number generalises; **no scale relief**, and a 10 B inherits the damage |
| **≥ 1.25** | **COST-GROWS** — the premise is false and the road needs re-planning before anything is built on it |

The middle cell is not decorative: **COST-SHRINKS or COST-GROWS also requires 1.5 B to sit between
0.5 B and 3 B in the same direction.** If it does not, the verdict is **NON-MONOTONE** and three
points were not enough — reported as such, not smoothed into a trend.

## 5. What this expressly does NOT license

**Three points across 6× do not extrapolate to 10 B**, which is 3.3× beyond the largest cell. E12
can establish a **direction** and nothing more. This programme has already been burned by a
two-point fit that a third point refused (E7 §8.2), and the rule written down then applies here:
**below the band is not a phenomenon.** A `COST-SHRINKS` verdict would license *building the 10 B
experiment*, not *claiming the 10 B number*.

**It does not test activation quantization** (`--lut`'s separate 1.40e-01 / 3.10e-02 cost), and it
does not test MoE routing. It tests the format's damage to a dense donor's weights, which is the
input every later step assumes.

## 6. Honest ceiling

No outcome of E12 makes anything faster. Coder-7B is **7.4× short of 50 tok/s** and stays there.
What E12 decides is whether the one remaining strategy is standing on a measurement or on a
single-shape assumption — which, after E10 removed the engine's cover, is the question that
determines what gets built next.
