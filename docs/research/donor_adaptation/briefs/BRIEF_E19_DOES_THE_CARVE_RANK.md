# E19 — does the carve rank? And how deep would it have to cut?

**Pre-registration. Part B has not been run. Nothing in §§3–6 has been measured.**
Part A (§2) **has** been run and is carried here as a measured *input*, the way E18 §2 carried its
own part A — its numbers are derivation over the ledger, not new timing.

---

## 0. The question

E18 closed the conversion route on two independent axes and left one structural escape: **50 tok/s
is a ~10%-activation budget**, and the only lever that reaches an activation budget is *conditional
activation* — carving. This programme has built exactly one kind of carve (D0, D0c, Probe-4) and it
carves **the FFN and only the FFN**.

Two questions, in the order that decides:

1. **How deep would an FFN carve have to cut to reach the budget?** (§2, already answered.)
2. **Does a carve rank at all?** D0 and D0c scored every carve arm **in BPB only** — the exact
   omission E18 caught in T2b, where five arms with `2.4` BPB of spread all turned out to be at the
   agreement floor. **No carve of this donor has ever been read in generation, at any granularity.**

## 1. Why this before healing

E18 §9 item 1 named healing the load-bearing owed item, and it still is *for the quality axis*.
But healing cannot move the speed axis: **a perfectly healed all-ternary 7 B still runs at
7.21 tok/s** (E18 §31.4, weight path). If no carve can rank, then healing has to repair two
catastrophic failures at once and the donor route is closed on evidence; if a carve *does* rank,
healing acquires a well-defined target. **Sequencing E19 first costs hours of CPU and decides what
the expensive experiment should even aim at.**

## 2. Part A — the carve depth the budget requires. MEASURED, and it constrains §3

`benchmarks/donor_adaptation/engine/e19_carve_budget.py`, derivation over measured quantities only,
importing `qwen_shapes` and the bytes-per-weight verification from `e18_ladder_bandwidth.py` rather
than restating them. **No timing taken.** Every weight is charged as **ternary** (`0.500000` B,
recomputed from the artifact) — the friendliest possible assumption, since it grants full
ternarization for free.

| model | active/token | FFN | **attn + head** | tok/s with the **FFN at ZERO** |
|---|---|---|---|---|
| Qwen2.5-Coder-7B | `7.070 G` | `5.703 G` (80.7%) | **`1.367 G`** | **35.9 – 38.8** |
| Qwen2.5-1.5B (the D0/D0c donor) | `1.544 G` | `1.156 G` (74.9%) | `0.388 G` (25.1%) | 126.7 – 136.8 |

**On the 7 B, `attn + head` alone (`1.367 G`) exceed the entire 50 tok/s budget (`0.982–1.060 G`).
So 50 tok/s is unreachable by carving the FFN alone at ANY depth — including deleting it outright.**

Projected to the goal's size, carrying each donor's **measured** attn+head share across to a 10 B of
the same proportions (a projection, labelled as one, no config invented): a 10 B at the 7 B's 19.3%
share has a **`1.934 G` uncarvable floor = `1.82×` the entire budget**, i.e. **25.4–27.4 tok/s with
its FFN carved to zero**. At the 1.5 B's 25.1% share it is `2.37×` over and 19.6–21.1 tok/s.

**Consequence for §3, fixed before any arm runs:** at 1.5 B — the only donor with carve labels on
disk — 50 tok/s needs an FFN active fraction of **`0.5144 – 0.5817`**, not 10%. The verdict cell is
therefore **~52% activation**, and the 25% cells D0/D0c already scored are *more* aggressive than
the 1.5 B budget requires. This is registered now so that the arm list cannot be tuned afterwards.

## 3. Arms

All arms use the **ORACLE top-k router** — it selects experts by reading the true squared-activation
mass *after* computing it, so it is an **upper bound on any trainable router**. If the ceiling
cannot rank, nothing below it can. `install()`, the partitions, and the B3 seeding repair are
**imported from `d0c_granularity.py` / `d0_coactivation.py`**, not re-derived, exactly as E18
imported T2b's `ARM_ORGANS`. Label caches `results/d0c_labels/labels_E*.npz` already exist.

| tag | E | k | active | role | BPB already measured |
|---|---|---|---|---|---|
| `base` | — | — | 100% | **control**, must reproduce `results/e6/ref.json` | `0.767595` |
| `FULL` | 256 | 256 | 100% | **identity control for the carve path** — the hook runs and removes nothing | — |
| `V52` | 256 | 133 | **51.95%** | **the verdict cell**: the 1.5 B's own 50 tok/s requirement (§2) | — |
| `S1` | 256 | 64 | 25% | finest carve D0c built; **best BPB of any carve** | `1.383868` |
| `A0` | 32 | 8 | 25% | **D0's headline cell** — the one `+1.09062 BPB` was read from | `1.858218` |
| `N0` | 32 | 8 | 25% | **null partition** — same expert sizes, random assignment | `2.578731` |
| `D10` | 256 | 26 | 10.16% | brackets the depth a larger model would need | — |

`base` and `FULL` are the two controls; `N0` is the honest comparator that asks whether
co-activation buys **ranking**, not just BPB (D0c's `G32` gap was BPB-only).

## 4. Gates

- **`G-C0`** — `base` reproduces E6's reference at `160/160`. **If it does not, E19 is VOID.**
- **`G-C1`** — `FULL` is **token-identical** to `base`. This is the carve path's identity control:
  it proves the hook, the partition load and the oracle machinery remove nothing when `k = E`.
  **If `FULL` differs from `base`, every carve reading is uninterpretable and E19 is VOID.**
- **`G-C2`** — replication: each arm with a published BPB reproduces it to `< 1e-6`
  (`base`, `S1`, `A0`, `N0`). Ties this harness to D0c's numbers before its new cells are read.
- **`G-C3`** — the verdict. For each carve arm, greedy agreement against the donor's own
  continuations over the frozen 5 prompts × 32 new tokens = **160 positions**.
- **`G-C4`** — achieved activation is recorded per arm and must match nominal within `0.002`
  (D0c §3.2's own tolerance), so "25%" means 25%.

## 5. Bands — fixed here, before the run

Identical to E18's, which were themselves derived from measured quantities and have now been used
twice:

- **floor = 12/160** — the best constant-token predictor on the 1.5 B reference (E18 part A).
- **margin = 2** — E17's own: a change that rewrote 49% of the output moved 2 correct tokens.
- **`AT-FLOOR` ≤ 14/160**; **`RANKS` ≥ 80/160**; between them, `PARTIAL`.

**No band is one-sided** (E14's law): a carve arm scoring *above* the reference is as reportable as
one below, and `PARTIAL` is a real outcome with a real reading, not a gap.

## 6. Predictions — and the alternative, registered

E18's law: registering the alternative outcome is worth more than getting the direction right.
**Both readings are written here before the data exists.**

1. `G-C0` fires at `160/160` and `G-C1` is token-identical. *(High confidence — E18's harness did
   both, and this adds one hook.)*
2. `G-C2` replicates all four published BPBs to `< 1e-6`.
3. **`V52` (52% active) → `RANKS`.** Reasoning: the carve is a *different kind of damage* from
   ternarization. Ternarization perturbs every weight; a carve leaves every surviving weight
   **bit-exact** and computes a *subset* of the true function. Half the FFN mass, oracle-selected,
   should preserve the argmax far more often than a 12/160 ternary arm does.
4. **`S1`/`A0` (25%) → `PARTIAL`**, `S1` above `A0` (it is `0.474` BPB better).
5. **`D10` (10%) → `AT-FLOOR`.**
6. **`N0` scores below `A0`** — co-activation buys ranking as well as BPB.

**THE REGISTERED ALTERNATIVE.** *If instead `V52`, `S1`, `A0` and `D10` all come back `AT-FLOOR`,
then the carve behaves exactly like every conversion this programme has tried, and the reading is
much stronger and much worse: **the donor's argmax does not survive ANY structural modification —
not precision, not sparsity — and the failure is not a property of ternarization but of post-hoc
modification of a pretrained dense model as such.** Combined with §2 (which already shows FFN-only
carving cannot reach the target at 7 B or above), that closes the donor route on both axes with no
remaining branch except healing, and makes healing a repair of TWO failures rather than one.*

**And the third possibility, also registered:** if `V52` ranks but `D10` does not, the carve has a
usable depth at 1.5 B and an unusable one at the depth larger models need — in which case §2's
attn+head floor is the binding constraint and the next experiment is **carving attention and the
head**, which nothing in this programme has ever attempted.

## 7. What E19 will not be able to claim

- **The oracle router is not a model.** It reads the answer to choose the experts. Every number here
  is a **ceiling**; a real router is worse. A `RANKS` result would therefore be *permission to try*,
  never a working system.
- **No healing.** Same scope limit as E18, unchanged.
- **One donor (1.5 B), one corpus, 160 positions, 5 prompts**, and the carve labels are D0's,
  fit on D0's mask at `p = 10%`. A different partitioner might carve better.
- **Part A is arithmetic, not a measurement.** No timing is taken and **`6.79 tok/s` stays exact**.
- **Nothing here measures the engine.** The carve has no engine implementation; a ranked carve would
  still need one, and Phase 61's law forbids carrying a PyTorch result into an engine rate.

## 8. Cost and stop rule

Seven arms × ~160 greedy positions at 1.5 B, plus BPB for the three new cells. Estimated **40–70
minutes**, CPU only, machine idle, one job. **If `G-C0` or `G-C1` fails, the run stops and E19 is
filed VOID** rather than reported with a broken instrument — E12's law.
