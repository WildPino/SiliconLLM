
# BRIEF E37 ADDENDUM B — is the fitted router load-bearing at all?

**Pushed before the control runs. The verdict cell is already measured, recorded and committed
(`S15-K3 = 4.029398`, `e4c03cd` + the quality run), and this control CANNOT change it.**

---

## 1. The gap this closes, which is a gap in my own gate

`G-E37D` fired convincingly: the fitted router recovers **0.5075** of the oracle's top-3 groups
against **0.0102** for a seed-matched random one, and captures **0.7132** of the oracle's mass
against **0.0608**, on 28/28 layers.

**That is a statement about predicting group mass. It is not a statement about BPB.** E14 §3
says a SCORE needs a RANK partner; this is the mirror failure — I have a rank result
(`G-E37D`) standing in for a score result it does not imply. Nothing measured so far shows that
routing *well* produces a better model than routing *badly*, and the sweep's own shape raises
the suspicion sharply: BPB is **flat at 3.99–4.03 from `k=16` all the way down to `k=1`**, a
40× change in activation that moves quality by less than the noise between neighbouring `k`.

If a 40× change in *how many* neurons survive does nothing, it is entirely possible that *which*
neurons survive also does nothing — in which case `G-E37D` is measuring a real improvement in a
quantity the model does not care about.

## 2. The control

Export a second carved artifact **identical in every respect** — same donor, same revision,
same `--rule R3`, same `--fold none`, same labels, same `--carve-k 256` in the file — except
that the router is the **synthetic** one (`--carve-seed 26`, `carve_common.router_weights`),
which is what every carved artifact before E37 has used.

Sweep `k ∈ {64, 32, 16, 3, 1}` under E1's protocol, unchanged.

## 3. The rule, fixed before the run

* **The verdict cell does not move.** `S15-K3 = 4.029398` and its band `SPARSITY-DEGRADES` are
  measured and committed. This control may not re-open them.
* Its **only** output is whether the fitted router is load-bearing on BPB.
* **`FITTED-MATTERS`**: fitted beats synthetic by **> 0.05 BPB** at the verdict `k`. Then routing
  quality is a real lever and the remaining gap is about how much better a router can get.
* **`ROUTER-IS-NOT-THE-CONSTRAINT`**: the two are within **0.05 BPB**. Then a router that finds
  71% of the oracle's mass and one that finds 6% produce the same model, the damage is about
  **how few** neurons survive rather than **which**, and `G-E37D`'s win is real but inert.

## 4. Predictions, registered now

1. **`ROUTER-IS-NOT-THE-CONSTRAINT`.** I expect the two curves to sit inside 0.05 BPB of each
   other at every `k` below 64. I am registering this against my own `G-E37D`, which I wrote
   and which fired: I do not think it buys a model.
2. **At `k=64` the fitted router WILL show an advantage**, because 25% activation is the only
   arm where enough of the FFN survives for the choice to matter.
3. **The synthetic curve will also be non-monotone**, with its worst point in the middle of the
   range rather than at the bottom. If instead only the fitted curve is non-monotone, the
   anomaly is about the fitted router and not about partial FFNs, and that changes the reading.

## 5. The non-monotonicity, registered as an observation before it is explained

The measured curve is worst at `k=32` (**4.074431**, the only point OVER the 4.069819 chance
line) and *improves* as activation falls further, reaching **3.989839** at `k=1` (0.39%).

**Hypothesis, stated now so it cannot be fitted afterwards:** a partial FFN is worse than
almost no FFN. `down_proj` sums contributions from the neurons that survive; dropping most of
them does not shrink the update toward zero, it produces a **systematically wrong** update of
roughly the original magnitude. As `k → 0` the FFN contribution vanishes and the model falls
back on attention plus the residual stream; in the middle it is loudly wrong.

**This addendum does not test that hypothesis** — it would need a zero-FFN arm, which the
engine cannot express (`ffn_carved` clamps `k` to at least 1). It is registered as the reading
to beat, and named in E37 §7 as owed work.
