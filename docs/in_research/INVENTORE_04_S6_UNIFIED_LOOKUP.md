# The Inventor — 04: S6 design note — one "knowing" operator (recall ∪ router)

**Status: v2 DESIGN NOTE — no v1 impact, no code.** The frozen spec ships both mechanisms as designed;
this note records the unification for the v2/engine-v2 design gate and states what would have to be
measured before any adoption.

## 1. The observation (they are the same operator)

Both "knowing" mechanisms in the architecture are instances of

> **`lookup(q; K, V, k)`** — project the state to a query, score against a key set, take top-k, return a
> weighted sum of values.

| | recall slot (D4, IN v1) | MoE router (D1, IN v1) |
|---|---|---|
| query | `q(x_t)`, dim 128, L2-norm | `router(x_t)`, dim E |
| keys | past positions' `k(x_p)` (dynamic, grows) | E expert identities (static rows of the router matrix) |
| top-k | 16 of ≤128K | 8 of 32-256 |
| values | `v(x_p)` (128-d vectors) | expert weight blocks (streamed, computed through) |
| index | Hadamard-IVF + 4-bit ADC + exact rerank (64.3: 29µs @128K) | full-scan softmax (fine at E≤256) |
| trained by | InfoNCE (mined positives) | CE + Switch aux |

The differences are parameters of one operator, not different operators: key set dynamics
(append-only vs static), value semantics (state residual vs weights-to-run), and index regime
(sub-linear vs full-scan).

## 2. What the unification buys (concretely, engine-v2)

1. **One C code-path.** The 64.3 recall probe machinery (Hadamard partition + ADC shortlist + rerank)
   IS a router the day E outgrows full-scan: at E=2048+ fine-grained experts (the DeepSeek-granularity
   direction the E-dial points at), `argtop8(router·x)` becomes exactly the ANN problem the recall tier
   already solved on this silicon. Building router-scan and recall-scan as one kernel with two
   configurations halves the engine surface and gives the router a measured sub-linear path *for free*.
2. **One scaling law.** Both mechanisms' cost curves (µs vs N-entries at fixed k) are the same measured
   curve; the design gate for E-growth reads the 64.3 curve instead of commissioning a new probe.
3. **One training pressure family.** InfoNCE (recall) and load-balanced CE (router) are both
   contrastive selection losses over the key set; a v2 experiment can share the mined-positive
   machinery ("which expert SHOULD have fired" ≈ "which past position SHOULD have been read").

## 3. The dream extension (fast weights) — stated, bounded

If values can be *weight deltas* instead of state residuals, the context writes temporarily into the
knowing: `retrieved = low-rank ΔW applied to the MLP for this token`. That is product-key-memory
territory with a Zen2-native twist (the delta must stay in the resident budget: rank-1..4 × 128-d =
KB-scale, ρ-safe). **Bound:** this is a research program, not a design item — it enters only through
its own pre-registered probe chain, after v1 ships. Recorded here so the door has a handle.

## 4. What would have to be measured (pre-conditions, so this note is falsifiable)

- E at which full-scan routing exceeds the recall slot's query cost — **MEASURED 2026-07-18
  (`s6_router_crossover.py`, desk-level, torch 1-thread): crossover at E ≈ 2048** (58.9µs vs the 52.4µs
  t1 ANN reference; E=256 costs ~25µs → v1 untouched, as predicted). Above E≈2K the unified lookup is a
  speed argument; the router matrix bytes (4 MB fp32 at E=4096 vs 1.69 MB searchable recall @128K) push
  the same direction. C-engine confirmation behind the kernel interface = engine-v2 item.
- Router-as-ANN recall quality: top-8-of-ADC vs exact top-8 agreement ≥ the dispatch-exactness bar the
  engine already enforces (E4 gate 2 lineage) — routing tolerates NO approximation slack until measured.
- For fast weights: does a rank-1 ΔW read from context beat the same bytes spent on the recall residual?
  (matched-bytes A/B at sandbox scale.)
