# Controller audit — D0 co-activation, Part III (sections 12–15)

**Auditor:** the Controller (independent adversarial review; read-only on the probe report and on all repo code)
**Target:** `docs/research/donor_adaptation/probes/D0_COACTIVATION.md`, **Part III, lines 1518–1778**, commit `7c49a1b`
**Date:** 2026-09-03
**Predecessor:** `CONTROLLER_D0_AUDIT.md` (Parts I and II, 2026-08-29), whose findings section 14 claims to resolve
**Method:** every number re-derived from the rawest artefact on disk, never from a JSON field that also
reports it. The paired standard errors and z-scores were recomputed from `d0_carved_arms/*.npy` plus the
pinned slice's per-sequence byte counts, and cross-checked against **two independent estimators** the
Builder did not use (a delete-one jackknife over the 24 sequences, and a linearised ratio/delta-method
SE). The carve partitions were **recomputed in my own process** and compared bitwise against the stored
labels. The 210-cell counts were recomputed from `d0_coactivation.analyse.json` under the Builder's rule
and under two alternative sd-imputation rules.

**Standing law applied.** *"Three probes audited, three times the tables were perfect and the prose was
not."* I therefore checked every summary sentence in 12.3, 12.4, 13.1, 13.3, 13.4, 14 and 15 against the
row it names. **The tables in Part III are again very good.** Section 13.2, the coact-vs-null table, the
12.3 correction table, the 12.2 ARI table and the 13.4 cross-process table all reproduce exactly. Every
finding below is in the prose, in section 14's bookkeeping, or in a test whose construction was never
checked against the regime it was applied to.

---

## 0. Verdict summary

| # | tag | subject |
|---|---|---|
| B1 | **BLOCK** | The B3 repair is **not in the repo**. `d0_layout.py` still draws the sketch off the global RNG; the fix exists only inside two uncommitted scratchpad scripts. "B3 is closed on reproducibility" is false of the pipeline. |
| B2 | **BLOCK** | "Every co-activation number in Parts I and II is now re-derivable from its seed" is false, and I disproved it: the main run's values differ from the seed-7 sweep and fall outside the whole 8-seed range in 4 of 15 cells. |
| B3 | **BLOCK** | **190/210 is an artefact of the `p = 0.10` → all-`p` sd extrapolation.** 19 of the 20 "failures" have a threshold larger than the entire treatment statistic at that cell — the test cannot be passed there by any amount of structure. Density-aware sds give 199/210 or 209/210. The honest count is **70/70 at the one density where the sd was measured**. |
| B4 | **BLOCK** | Section 14 claims B1's sentence "is withdrawn" and B2's column "stays withdrawn". **Neither withdrawal exists in the document.** §11.12 line 1438 still asserts the false 0.5097; §11.7b line 1235 is unannotated. The only two occurrences of "withdrawn" in 1778 lines are inside section 14's own table. |
| F1 | FLAG | "18 of the 20 at block ≥ 32" contradicts the list printed two lines below it. It is **20 of 20**. |
| F2 | FLAG | "The treatment's own sd is 1.4× to 200× the null's" is wrong at both ends. Measured range **0.96× to 4545×**; at L7 bs16 the treatment sd is *smaller* than the null's. |
| F3 | FLAG | "the damage compounds with depth rather than accumulating linearly" **inverts its own number**: 28 layers cost 16.8× one layer, and 16.8 < 28 is sub-linear. |
| F4 | FLAG | "the marginal slice SE … would swamp every effect below" is false for two of the four effects, and calls two different quantities "the same comparison". |
| F5 | FLAG | "No trainable router can beat it" / "no routing improvement can move it" is not established. The oracle is top-k by **retained activation energy**, per token, per layer, greedily — an oracle on a proxy, not on BPB. |
| F6 | FLAG | F8 is not "PARTLY ANSWERED". The audit asked for a **positive** control; section 13 supplies a **negative** one on a different metric. Separately, **the carve hook has no control of any kind** — the baseline arm installs no hook, so the hook path is never run against a known answer. |
| F7 | FLAG | Section 14 silently omits **7 of the previous audit's 15 findings** (F2, F3, F7, F9, F10, F11, F12). F10 (provenance) is live again: both carve runs record `git_dirty_at_launch: true`, and every Part III script is uncommitted. |
| F8 | FLAG | "section 13 selects nothing" is false. `E=32, k=8` is the **argmax of Part I's 120-row fidelity table** (relerr 0.0243, the single lowest). The selection is in the conservative direction, but F6 is not moot. |
| F9 | FLAG | 13.4's "free determinism check" did **not** test the partitioning — the measuring process loaded labels from `d0_carved_labels_E32.npz`. I supply the missing evidence below; the claim survives, the check as cited does not establish it. |
| F10 | FLAG | "matches … S1's 0.767594952 to 8 decimal places" — it matches to **7**. (C1's match to 8 dp is correct.) |
| F11 | FLAG | §15 point 4 drops the denominator: "the structure is worth 39.8% of the damage". 39.8% is of the *null's* damage; of the co-activation carve's own damage it is 66%. |
| F12 | FLAG | §15 sets a **2.04%-budget** fidelity number beside a **25%-budget** BPB number without saying they are different operating points. |
| F13 | FLAG | The 46.5 / 50.5 / 71.7% scale curve — which carries §15's entire "what stays open is scale" — is **uncited in Part III and uncited in its source**. |
| F14 | FLAG | z(token) = −60.8 is quoted as a second confirmation of z(seq) = −6.3. It assumes token independence inside a sequence and is anti-conservative by ~10×; it confirms nothing the sequence z does not. |
| P1 | **PASS** | Every BPB value, every paired SE, every z and every `frac_tokens_worse` reproduces from the raw `.npy`. Two independent SE estimators agree with the bootstrap to ≤3%. Section 13.2 is clean. |
| P2 | **PASS** | 12.3's table reproduces **exactly** — all 12 margins, all null sds, all treatment sds, all ratios, and the 140 → 41.6 correction. This is the strongest thing in Part III and it is right. |
| P3 | **PASS** | 209/210 and 195/210 reproduce. §15's "clears its null in 209 of 210" is correct: the single failing cell fails both legs. |
| P4 | **PASS** | 13.4's cross-process table reproduces (max dev 3.73e-8). I additionally verified partition determinism **bitwise**, which the report asserts but never tested. |
| P5 | **PASS** | 12.2's ARI table reproduces exactly, and the finding it carries — reproducible statistic, unidentified partition — is correct and is the most useful negative in Part III. |
| P6 | **PASS** | The B3 defect is real and the repair is load-bearing. I ran the planted control the Builder did not: without `manual_seed`, the same call in the same process returns a different partition (ARI 0.34–0.38). |

**Direction of the lean.** Part III leans the **same way Parts I and II did**. B3, F1, F3 and F5 each make
the negative look wider or more final than the artefacts support. The 12.3 correction is honest and
correct; the 12.4 re-count built on it over-corrects, and the over-correction is the number that reached
section 15 and the handoff.

---

## B1 — BLOCK. The B3 repair is not in the repo

§12.1 states the repair and calls B3 "closed on reproducibility". The repair is a single line,
`torch.manual_seed(seed)`, and it lives **only** inside two scripts in a session-temporary directory:

```
C:/Users/giosa/AppData/Local/Temp/claude/D---THINGS-Progetti-SiliconLLM/
   1f268f4b-0a3c-40c3-a2f9-63f38f70de0c/scratchpad/b3_sweep.py
   .../scratchpad/carved_bpb_paired.py   (and carved_bpb.py)
```

The repo pipeline is untouched:

```
$ grep -rn "manual_seed" benchmarks/donor_adaptation/density/*.py
d2_basis.py:187 / d2b_actbasis.py:134 / d4_reconstruction.py:161      # d2, d2b, d4 only
```

`d0_layout.py:143` (`clustered_order`) and `d0_layout.py:182` (`balanced_labels`) still call
`torch.svd_lowrank(X, q=dim, niter=4)` off the **global** RNG, and neither `d0_coactivation.py` nor
`common.py` seeds it. `git log` on those two files shows no commit since `30efdd9`, the commit that
published Parts I and II.

So: run `d0_coactivation.py` today and it is still non-reproducible. What §12.1 demonstrates is that a
*wrapper written outside the repo* can be made deterministic. That is not the same statement, and the
scripts carrying it are in a directory that is garbage-collected without notice — **the artefacts of
Part III currently have no committed producer at all.**

**Reproduction:** the grep above; `sed -n '139,190p' benchmarks/donor_adaptation/density/d0_layout.py`;
`git log --oneline -- benchmarks/donor_adaptation/density/d0_layout.py`.

**What would close it:** move the seeding into `d0_layout.py` / `d0_coactivation.py`, commit
`b3_sweep.py` and `carved_bpb_paired.py` into `benchmarks/donor_adaptation/density/`, and re-state §12.1
as "the repair is committed at `<sha>`".

---

## B2 — BLOCK. "Every co-activation number in Parts I and II is now re-derivable from its seed" is false

This is §12.1's closing sentence and it is the load-bearing consequence the repair is claimed to have.
It is false twice over, and the second half is disprovable from artefacts already on disk.

**First:** by B1 the pipeline is unrepaired, so nothing is re-derivable from it.

**Second, and independent of B1:** Parts I and II were produced by an **unseeded** draw. Seeding the RNG
now produces a *different* partition from the one that produced those numbers, and the global RNG state
of the original run was never recorded. It is unrecoverable in principle.

I checked this rather than argued it. `d0_b3_treatment_spread.json` contains the seed-7 value of exactly
the statistic Parts I/II report as `coactivation_best`, at `p = 0.10`, at every deep layer and block. It
does not match the main run at any cell I checked, and in 4 of 15 it falls **outside the entire 8-seed
range**:

| cell | main run (Parts I/II) | seed-7 sweep | 8-seed range | main run inside range? |
|---|---|---|---|---|
| L1 bs12 | 0.394092 | 0.392766 | [0.386931, 0.393278] | **no** |
| L1 bs64 | 0.028601 | 0.026399 | [0.024802, 0.028448] | **no** |
| L7 bs140 | 0.000121 | 0.000061 | [0.000043, 0.000104] | **no** |
| L27 bs64 | 0.159288 | 0.157006 | [0.148529, 0.158226] | **no** |
| L27 bs12 | 0.521374 | 0.528909 | [0.519873, 0.531704] | yes |

(4 of 15 outside the range of 8 draws is what chance predicts for a fresh draw — 2/9 = 22% — so this is
not evidence of a *different distribution*. It is direct evidence that the specific numbers in Parts I
and II are a draw nobody can reproduce.)

This also means **§12.3's table pairs quantities from two different RNG populations**: the `margin`
column comes from the unseeded main run, the `treatment sd` column from the seeded 8-seed sweep. That
is defensible as an error bar on the *statistic*, and I do not block on it — but the sentence "every
number … is now re-derivable" must go.

**The correct statement:** the repair makes future runs reproducible. Parts I and II remain a single
unrecoverable draw whose spread is now, for the first time, characterised.

**Reproduction:**
```python
A=json.load(open('.../d0_coactivation.analyse.json')); S=json.load(open('.../d0_b3_treatment_spread.json'))
A['deep_layers']['1']['densities']['0.10']['verdict_by_block']['12']['coactivation_best']  # 0.394092
S['layers']['1']['per_seed']['7']['best_s']['12']                                          # 0.392766
[S['layers']['1']['per_seed'][s]['best_s']['12'] for s in ('7','11','22','33','44','55','66','77')]
```

---

## B3 — BLOCK. 190/210 is an artefact of the density extrapolation, and the caveat does not cover it

§12.4 reports "margin > 2× measured treatment sd → **190/210**" and adds a *"Stated limit on this test"*:
the sd was measured at `p = 0.10` only and applied unchanged to `p = 0.05` and `p = 0.20`. **The caveat
names the defect and then reports the number anyway** — in the 12.4 table, in §15 point 3, and onward
into the handoff. It is not adequate. The extrapolation does not add noise to the count; it *is* the
count.

The arithmetic reproduces exactly (190/210, and the failing-cell list is identical to the one printed).
Split it by density and the whole result is one cell of the table:

| density | cells clearing 2× the (imported) treatment sd |
|---|---|
| p = 0.05 | 70 / 70 |
| **p = 0.10** — the only density where the sd was measured | **70 / 70** |
| p = 0.20 | **50 / 70** |

Every one of the 20 failures is at the density where the sd is imported. Now look at what the imported
threshold actually is there. At `p = 0.20` the skippable statistic is one to three orders of magnitude
smaller than at `p = 0.10`, while the threshold is carried over at full size:

| failing cell | `coactivation_best` at the cell | threshold `2×sd` | ratio |
|---|---|---|---|
| L27 p=0.20 bs128 | 0.000133 | 0.010401 | **78×** |
| L27 p=0.20 bs140 | 0.000075 | 0.010360 | **138×** |
| L21 p=0.20 bs64 | 0.000095 | 0.001131 | 12× |
| L14 p=0.20 bs128 | 0.0000003 | 0.000105 | 350× |
| L1 p=0.20 bs128 | 0.000002 | 0.000587 | 294× |

**In 19 of the 20 "failures" the threshold is larger than the entire treatment statistic at that cell.**
The margin cannot exceed `coactivation_best`, so no amount of structure — not a perfect partition, not an
oracle — could pass. Those 19 cells are not measurements that came out negative. They are cells where the
test is undefined. §12.4's sentence *"indistinguishable from noise only where there is nothing left to
skip"* asserts a measurement that was not made; the accurate word is "untested", not "indistinguishable".

Re-run the same test with a density-aware sd and the count moves:

| sd rule | count |
|---|---|
| Part III: absolute p=0.10 sd, same block | **190 / 210** |
| within-layer magnitude match (use the p=0.10 block whose mean `best_s` is closest to this cell's) | 199 / 210 |
| proportional (relative sd `sd/mean` at p=0.10, scaled to this cell's `coactivation_best`) | 209 / 210 |

The count is a free parameter of the imputation rule. It should not have been reported as a third
column beside 209/210 and 195/210, which are measurements.

Note the p = 0.05 column too: it passes 70/70, but for the mirror-image reason. There the statistic is
*larger* than at p = 0.10, so the imported sd is too small and the test is anti-conservative. Only the
p = 0.10 row is a measurement. The 190 is 70 real passes, 70 anti-conservative passes, and 70 cells where
the test mostly cannot be run.

**What the artefacts support:** *"At `p = 0.10`, the one density where the treatment's own spread was
measured, **70 of 70** cells clear 2× that spread. At `p = 0.05` and `p = 0.20` the treatment sd was
never measured; the test is not defined there."* That is a stronger and cleaner statement than 190/210,
and it is the one the data licenses.

**Note on direction.** The defect makes the structure look weaker than the evidence supports. It is the
same lean the previous audit found in Parts I and II — the over-statements all widen the negative.

**Reproduction:** for each of the 210 cells in `d0_coactivation.analyse.json`, compare
`coactivation_best − random_max` against `2 × d0_b3_treatment_spread.json['layers'][L]['treatment_stats'][bs]['sd']`,
then tabulate by `p`; and for the 20 failures compare `2×sd` against `coactivation_best` itself.

---

## B4 — BLOCK. Section 14 claims two withdrawals that were never made

| section 14 row | claim | state of the document at commit `7c49a1b` |
|---|---|---|
| B1 | "the 11.12 sentence **is withdrawn**" | line 1438 still reads: *"**`FFN_active` never goes below 0.5097** anywhere in 210 cells"* — unmarked, unedited, unqualified |
| B2 | "the column **stays withdrawn** pending a rewrite" | §11.7b (line 1235) carries no withdrawal marker of any kind |

```
$ grep -n -i "withdrawn" docs/research/donor_adaptation/probes/D0_COACTIVATION.md
1732:| B1 | ... | **UPHELD** - the 11.12 sentence is wrong and is withdrawn; ...
1733:| B2 | ... | **UPHELD, still open** - ... the column stays withdrawn pending a rewrite |
```

Both hits are inside section 14's own table. **The withdrawal is asserted in the errata and never
performed on the text.** A reader who reaches §11.12 — 300 lines before section 14 — reads a claim the
author knows is false, with nothing on the page to warn them. The document is the deliverable; a status
table that describes a document state that does not exist is a claim about an artefact contradicted by
the artefact.

"Withdrawn" is also the wrong word for what §14 does even in intent: an appended note 300 lines later is
an erratum, not a withdrawal.

**What would close it:** strike or annotate line 1438 in place ("*withdrawn — see 14/B1; the true
minimum is 0.3911*") and the same at §11.7b, then leave section 14's row as the pointer.

---

## F1 — FLAG. "18 of the 20 at block ≥ 32" is 20 of 20, and the list two lines below says so

§12.4: *"**All 20 sit at `p = 0.20`**, and 18 of the 20 at block ≥ 32"*, followed by the list:

```
L1  bs 64,128,140   L7  bs 42,64,128,140   L14 bs 32,42,64,128,140
L21 bs 32,42,64,128,140   L27 bs 64,128,140
```

3+4+5+5+3 = 20, and every block named is ≥ 32. My recount agrees exactly with the list. Blocks below 32
contribute **zero** failures. The sentence understates the cleanliness of its own pattern — a small
error, in the same direction as B3.

**Reproduction:** `Counter(int(bs) for (L,p,bs) in failures)` → `{32:2, 42:3, 64:5, 128:5, 140:5}` = 20.

---

## F2 — FLAG. "1.4× to 200× the null's" is wrong at both ends, and "always larger" is false

§12.3: *"The treatment's own sd is 1.4× to 200× the null's, depending on the cell"*, restated in §15
point 3 as *"its own error bar is 1.4–200× the one the document had been quoting"*.

Over all 70 (layer, block) cells at `p = 0.10`:

| | value | cell |
|---|---|---|
| minimum ratio | **0.96×** | L7 bs16 (treatment 0.000630, null 0.000656) |
| maximum ratio | **4544.6×** | L27 bs140 (treatment 0.005180, null 0.00000114) |

Restricted to the 12 rows §12.3 actually prints, the range is **1.33× to 4544.6×**. Neither endpoint of
"1.4 to 200" appears in either reading.

The low end matters more than the high end. At L7 bs16 the treatment sd is *below* the null's, so the old
error bar was **conservative** there — which contradicts the section's framing that the null's sd was
uniformly the flattering choice. Nine cells sit under 1.4×.

**Reproduction:** ratio `treatment_stats[bs]['sd'] / verdict_by_block[bs]['random_std']` over all
`deep_layers × blocks` at `p = 0.10`.

---

## F3 — FLAG. "the damage compounds with depth" inverts the number it is drawn from

§13.3: *"Carving all 28 costs 1.09062, which is **16.8×** the single-layer cost across 28 layers: the
damage **compounds with depth rather than accumulating linearly**, and there is no depth at which it is
free."*

1.090623 / 0.064988 = **16.78**, across **28** layers. Linear accumulation of L27-sized damage would
predict 28×. 16.8 < 28 is **sub**-linear — the opposite of "compounds … rather than accumulating
linearly". The evidence is two points; nothing in Part III can distinguish super- from sub-additivity in
general, because no other single layer was carved alone. What the two points do show is that the total is
**less** than a naive 28× extrapolation of the deepest and most damaging layer.

The trailing clause — "there is no depth at which it is free" — is supported and should carry the
sentence on its own.

**Reproduction:** `paired_vs_baseline.all_coact.delta_bpb / paired_vs_baseline.L27_coact.delta_bpb`.

---

## F4 — FLAG. "would swamp every effect below" is false for half the effects

§13.1: *"the marginal slice SE on the baseline is 0.0622, which would swamp every effect below; the
paired SE on the same comparison is 0.0042."*

| effect | delta | marginal two-sample SE `√(se_a²+se_b²)` | marginal z |
|---|---|---|---|
| L27_coact | 0.0650 | 0.0904 | 0.72 — swamped |
| L27_null | 0.1466 | 0.0946 | 1.55 — swamped |
| **all_coact** | **1.0906** | 0.1103 | **9.9 — not swamped** |
| **all_null** | **1.8111** | 0.1367 | **13.2 — not swamped** |

The paired design is a real and correct improvement — it is what makes the L27 arms and the
coact-vs-null comparison readable. But "every effect below" is two of four. Separately, "the paired SE on
the same comparison" is not the same comparison: 0.0622 is the marginal SE of the *baseline BPB level*,
0.0042 is the paired SE of the *L27_coact − baseline delta*.

**Reproduction:** `marginal_slice_se` and `paired_vs_baseline` in `d0_carved_bpb_paired.json`.

---

## F5 — FLAG. The oracle is an oracle on activation energy, not on BPB

§13.1: *"**The router is an oracle.** It selects the `k` experts by true activation. **No trainable router
can beat it.** Everything below is a **ceiling**."* §13.3: *"no routing improvement can move it."* §15:
*"the only thing that could change that verdict is a different point on the scale curve, **not a better
router — the router here is already an oracle**."*

The hook (`carved_bpb_paired.py`, `install()`):

```python
sel = ((flat ** 2) @ oh).topk(k, dim=1).indices     # top-k experts by summed squared activation
```

That is a **greedy, per-token, per-layer selection maximising retained activation energy**. It is an
oracle on that objective. It is not an oracle on BPB. A router optimising the decision metric could
(a) trade energy for downstream effect, and (b) account for interaction across the 28 carved layers,
which a per-layer greedy rule cannot. Nothing here bounds how much that is worth. I inspected the hook
line by line and it correctly implements what it says — `keep[:, lab]` maps expert-keep to neuron-keep in
the right direction, and `oh` is a correct one-hot — so this is not an implementation defect. It is an
over-claim about what the construction bounds.

The verdict does not turn on it: +1.09 BPB is 15.6 paired SEs and 218 σ_seed, and a router improvement
would have to close ~1.0 BPB. But "no routing improvement can move it" is the sentence that licenses
§15's directive to leave the joint brief shut, and it is stronger than the construction supports. The
defensible form: *"a router that maximises retained activation energy cannot be beaten at that objective;
we have not bounded the gap between that objective and BPB."*

---

## F6 — FLAG. F8 is answered with a negative control against a demand for a positive one, and the carve harness has no control at all

Section 14: *"F8 — **PARTLY ANSWERED** — section 13's random-null arm is a genuine negative control on
the *decision metric*, and the co-activation arm clears it at z = −6.3."*

What F8 said: *"**No positive control for `routed_error`** — the whole of Part I's 120-row fidelity table
… **No control of any kind for `moe_geometry` or `duplication`** … **Standing rule on this programme is
that an instrument must be shown firing on a known positive before its nulls count.** That rule is
satisfied for the permutation/skippable path (C2) and **not** satisfied for the fidelity, geometry or
duplication paths."*

Part III supplies a **negative** control (a shuffled partition) on a **new** path (BPB). The three paths
F8 named — fidelity, geometry, duplication — are untouched. Substituting a different control type on a
different metric is not a partial answer to that demand; under this project's own planted-control law it
is the exact move the law exists to catch. The correct status is **UPHELD, still open**, with a note that
section 13 adds a good negative control on a fourth path.

**And the new path has no control either.** The `baseline` arm runs with `layer_ids = None`, which
installs **no hook at all**. So the hook code — the thing that produces every carved number — is never
executed in a condition with a known answer. The cheapest missing control is one line: install the hook
at `k = E = 32` (keep everything) and assert the BPB equals the uncarved baseline. Two further graded
positives would cost one arm each: carve at `k = 16` and `k = 24` and check the damage is monotone in
`k`. None was run.

One further construction note on the null: `labels_null` is drawn once, from
`np.random.default_rng(1000 + E)`, i.e. **a single shuffle, with the same permutation reused at every
layer**. Part II's null was a max over 8 shuffles and had a spread; section 13's has n = 1. The paired
sequence bootstrap therefore prices sequence-sampling noise but carries **no variance component for the
draw of the null partition**. z = −6.3 is comfortable enough that this is unlikely to matter, but the
error bar is not what a reader assumes it is.

Two things do partly discharge the burden and I credit them: the null arm is a real negative control on
the decision metric, and the baseline reproducing C1 and S1 to 7–8 dp shows the *unhooked* forward path
is sound. Neither exercises the hook.

---

## F7 — FLAG. Section 14 answers 8 of the previous audit's 15 findings and does not say so

The previous audit issued B1, B2, B3 and F1–F12. Section 14's table covers **B1, B2, B3, F1, F4, F5, F6,
F8** and is silent on **F2, F3, F7, F9, F10, F11, F12** — presented as a complete status table with no
statement that seven findings are out of scope.

Two of the omissions bear on Part III directly:

- **F10 (provenance) is live again.** `d0_carved_bpb.json` records `git_revision e79f548`,
  `git_dirty_at_launch: true`; `d0_carved_bpb_paired.json` records `684de6e`, `git_dirty_at_launch: true`.
  Combined with B1 — the producing scripts are uncommitted scratchpad files — **no Part III number has a
  reproducible provenance chain.**
- **F2** ("§11.7a's 79 of 80 is 72 of 80") is an uncorrected count in the same class as B1, in a document
  Part III declares closed in section 15.

**Reproduction:** section 14's table (lines 1731–1741) against `## 0. Verdict summary` of
`CONTROLLER_D0_AUDIT.md`; `env.git_dirty_at_launch` in both carve JSONs.

---

## F8 — FLAG. "section 13 selects nothing" — it selects the argmax of the fidelity table

Section 14, F6: *"**UPHELD, and now moot** — section 13 selects nothing; it carves at a fixed `E`, `k`
and measures BPB."*

`carved_bpb.py` line 13: `E_CARVE, K_CARVE = 32, 8   # 25.1% of FFN bytes; Part I's best-fidelity cell`.
I checked the claim in the comment against the artefact: across all 120 rows of `fidelity`, `E=32, k=8`
at L1 is the **single lowest `relerr` in the table** (0.02426; next is 0.02993).

So the operating point *was* selected, from the table F6 was about. **The direction is conservative** —
it is the kindest cell in the probe, and the carve still costs +1.09 BPB, which strengthens the negative.
That is a good design choice and it should be stated as one. But "selects nothing" is false, and F6 is not
"moot": a fidelity-optimal cell is not necessarily a BPB-optimal cell, and no other cell was measured in
BPB, so nothing here bounds how much of the +1.09 is the choice of `(E, k)`.

---

## F9 — FLAG. The "free determinism check" did not test the partitioning. I tested it; it holds

§13.4: *"The whole forward path, **the partitioning**, the oracle router and the BPB accounting reproduce
across processes and across days."*

The measuring process did not recompute the partition. `d0_carved_bpb_paired.log` line 4:
`partitions loaded from cache (28 layers)` — it read `d0_carved_labels_E32.npz`, written at 10:44 on
09-03 by an earlier attempt (`d0_b3.wallclock.txt`: `PAIRED_BPB_RESTART=10:42:19`, then
`RESTART2=10:50:58 pid=24252`). Calling a check "free" and then listing among its outputs a stage the run
loaded off disk is the framing the handoff warned about.

**I ran the check the report should have.** In my own process, on a different day, from the mask files:

```python
torch.manual_seed(7)
lab = np.asarray(D0.balanced_labels(torch.from_numpy(B), 32, 7))
np.array_equal(lab, np.load('d0_carved_labels_E32.npz')[f'c{L}'])
```

| layer | bitwise identical to the stored labels | ARI |
|---|---|---|
| L0 | **True** | 1.0000 |
| L14 | **True** | 1.0000 |
| L27 | **True** | 1.0000 |

**The claim is true.** It is now established, by a third process on a fourth day, rather than asserted.
Combined with the fact that the labels were computed on 09-03 at 10:42 in a process distinct from the
08-30 one, and that a different partition would move BPB by ~1e-2 rather than ~1e-8, the cross-process
partition claim is sound. Only its stated basis was wrong.

---

## F10 — FLAG. The S1 baseline matches to 7 decimal places, not 8

§13.4: *"The baseline also matches C1's independently-measured 0.7675949601 … and S1's 0.767594952 to 8
decimal places."*

| source | artefact | value | vs paired baseline 0.7675949641 | agrees to |
|---|---|---|---|---|
| C1 | `d0_coactivation.analyse.json` → `control_C1_losslessness.bpb_identity` | 0.7675949601147849 | 4.0e-9 | **8 dp** ✓ |
| S1 | `benchmarks/donor_adaptation/s1/results/s1_run_qwen2.5-1.5b.json` → `baseline_bpb` | 0.7675949524755323 | 1.17e-8 | **7 dp** |

At 8 dp: 0.76759496 (Part III) vs 0.76759495 (S1). One clause covers two agreements of different
tightness and reports the better one for both. The agreement is still excellent and the point it makes
survives intact; the number attached to it does not.

*(Minor, not scored: "three days apart" is 08-30 15:04 → 09-03 10:50, i.e. 3 d 19.8 h — four calendar
days.)*

---

## F11 — FLAG. §15 point 4 drops the denominator

| where | sentence | denominator |
|---|---|---|
| §13.3 | "recovers 0.7205 BPB **of the 1.8111 a random carve costs** — 39.8% of the damage" | stated, correct: 0.7205/1.8111 = 0.3978 |
| §15 pt 4 | "The carve costs +1.09 BPB … **The structure is worth 39.8% of the damage.**" | unstated, and the nearest antecedent in the same sentence is 1.09 |

Against the carve's own damage the recovery is **66%** (0.7205/1.0906), not 39.8%. §13.3 is right; §15,
which is the paragraph that gets quoted onward, is ambiguous and reads wrong. A ceiling is a
denominator, and this is where the wrong one hides.

---

## F12 — FLAG. §15 puts a 2.04%-budget number beside a 25%-budget number

§15 point 1: *"**Fidelity (Part I):** with an oracle router, the carve loses 59.5% to 94.7% of the FFN
output norm **at the budget the adapter needs**."*
§15 point 4: *"**BPB (Part III, §13):** the carve costs **+1.09 BPB = 218 σ_seed** with an oracle router."*

They are different carves. Point 1 traces to line 846, which reads *"`E=128, k=2` reads **2.04%** of FFN
weight bytes … At that operating point the oracle-routed carve loses 59.5% (L27), 83.7% (L21), 94.7%
(L14) and 90.8% (L7)"*. Point 4 is `E=32, k=8` = **25%** of FFN weight bytes — a budget **12× more
generous** than the one point 1 calls "the budget the adapter needs", and the single kindest cell in the
fidelity table (F8).

Nothing in §13 or §15 states the byte budget of the measured carve. A reader takes points 1 and 4 as the
same object. The direction is conservative — at 2% the BPB cost would be far worse than +1.09 — so this
does not weaken the negative. It does mean **+1.09 BPB is a floor on the damage at the budget that
matters, and §15 never says so**, which is the more useful sentence and the one that should replace it.

*(Also: "59.5% to 94.7%" is a range over four of the five deep layers; L1 is absent from line 846's list.)*

---

## F13 — FLAG. The scale curve that carries the whole "what stays open" paragraph is uncited

§15: *"FFN sparsity is known to rise with model size (**46.5% / 50.5% / 71.7% at 0.5B / 1.5B / 14B**). We
are measuring at the bottom of that curve."* This is the sole evidential basis for the paragraph that
keeps the axis alive, for S1's mandatory scale arm, and for the framing of what remains open.

Part III gives no source. Tracing it: the identical triple appears in
`docs/internal/OWNER_DECISION_NOTE_03.md:66-70`, introduced as *"C'è una misura pubblicata, sulla nostra
identica architettura"* — with **no paper, no table, no line reference**. The repo's own prior-art
dossiers (`DENSITY_PRIOR_ART.md`, `ACTIVATION_SPARSITY_PRIOR_ART.md`), which do carry `[TABLE]`-marked
citations, contain no such triple; the `71.7` hits there are unrelated benchmark averages.

This programme's standing rule is that a literature number does not enter a decision until it has been
read in its own table, and that marking it `[L?]` is not enough. Here it is unmarked and it is
load-bearing for a decision (the S1 scale arm was made mandatory on it, commit `f90ac3d`). The claim may
well be right — activation sparsity rising with width is a documented effect — but the three specific
percentages must be sourced to a table or restated qualitatively before they carry a gate.

**This is the one item I would put in front of the Principal first**, because unlike everything else here
it points forward rather than backward: it is the premise of the next probe, not a defect in a closed one.

---

## F14 — FLAG. z(token) = −60.8 is not a second confirmation

§13.3: *"at z = −6.3 paired on sequences **and −60.8 on tokens**."*

`paired_se_per_token` divides the per-token sd by `√12264`, treating 12,264 tokens inside 24 documents as
independent. They are not — the per-sequence deltas are the source of nearly all the variance, which is
exactly why the sequence bootstrap gives an SE ~10× larger. The token z is anti-conservative by that
factor and adds no information the sequence z does not already carry. Quoting both, in that order, reads
as corroboration.

The sequence figure is the right one and it is comfortable: z = −6.28, and my jackknife gives −6.09 on 23
degrees of freedom. The finding survives; the second number should be dropped or labelled.

---

## P1 — PASS. Section 13.2 and the coact-vs-null table reproduce, under three estimators

Recomputed from `d0_carved_arms/*.npy` (5 × 24 × 511 float64 per-token nats) and the pinned slice's
per-sequence byte counts, with no reference to the JSON's own statistics.

**BPB**, as `nats.sum() / ln2 / 51870`:

| arm | recomputed | reported | diff |
|---|---|---|---|
| baseline | 0.7675949641 | 0.7675949641 | −1.1e-16 |
| L27_coact | 0.8325832592 | 0.8325832592 | 0 |
| L27_null | 0.9142022100 | 0.9142022100 | 0 |
| all_coact | 1.8582178498 | 1.8582178498 | −2.2e-16 |
| all_null | 2.5787313454 | 2.5787313454 | −4.4e-16 |

**Paired errors.** The Builder used a ratio bootstrap over sequences (20 000 draws, seed 7). I reproduced
it and added two estimators he did not use — a delete-one jackknife over the 24 sequences and a
linearised (delta-method) ratio SE:

| comparison | delta | SE boot | SE jackknife | SE delta-method | z boot | z jack |
|---|---|---|---|---|---|---|
| L27_coact − baseline | +0.064988 | 0.00423 | 0.00429 | 0.00429 | 15.37 | 15.16 |
| L27_null − baseline | +0.146607 | 0.00943 | 0.00968 | 0.00966 | 15.55 | 15.14 |
| all_coact − baseline | +1.090623 | 0.06981 | 0.07130 | 0.07130 | 15.62 | 15.30 |
| all_null − baseline | +1.811136 | 0.14000 | 0.14343 | 0.14304 | 12.94 | 12.63 |
| **L27_coact − L27_null** | **−0.081619** | 0.00612 | 0.00631 | 0.00629 | **−13.34** | −12.94 |
| **all_coact − all_null** | **−0.720513** | 0.11475 | 0.11837 | 0.11789 | **−6.28** | −6.09 |

Every `frac_tokens_worse` matches to 4 dp (0.5635 / 0.5537 / 0.8621 / 0.9194). The independent
estimators sit 2–3% above the bootstrap in every row — the expected direction for a small-n jackknife —
so the reported SEs are, if anything, marginally optimistic and nothing turns on it. `delta/σ_seed` =
13.0 / 29.3 / 218.1 / 362.2 checks. The 39.8% recovery checks (0.39783). **The tables in §13.2 are
clean.**

The pinned slice on disk carries `ids_sha256 = a1a48dc9…d52f65`, 24 × 512, 51,870 bytes,
12,264 predicted tokens, matching every stated value. The partitions are exactly size-balanced
(280/280 neurons per expert at E = 32), so `k/E = 0.25` really is a byte fraction and not an average —
no charged-vs-moved-bytes defect fires anywhere in section 13.

---

## P2 — PASS. §12.3's correction table reproduces exactly, and it is the best work in Part III

All twelve rows check: margins (which are `coactivation_best − random_max` — I verified the definition
by exhaustion, `random_mean` does not reproduce them), null sds (`random_std`), treatment sds
(`treatment_stats[bs].sd`, which I re-derived from the eight per-seed `best_s` values with `ddof=1`), and
every ratio. 0.11051/0.00266 = 41.6; 140/41.6 = 3.4. The headline correction — *the 11.3 error bar was
the null's, the honest figure is 41.6 sd not 140 sd* — is right, is properly derived, and is exactly the
kind of self-correction this programme needs. The direction-survives / precision-did-not framing is
accurate.

One construction note, not a finding: `best_s` is a max over `E ∈ {16,32,64,128}` (the `best_E` column
varies by seed at coarse blocks), i.e. the same F6-tainted quantity as `coactivation_best`. The margin
and the sd are therefore the same class of object, which is what makes the ratio meaningful. §12.2's
setup line labels the arm "`E = 64`", which is true of the ARI only, not of the sd.

---

## P3 — PASS. 209/210 and 195/210, and §15's phrasing of them

Both counts reproduce over the 5 layers × 3 densities × 14 blocks = 210 cells. Broken out:
`coact_beats_identity` = 209, `coact_clears_random_spread` = 209, conjunction = **209** — the single
failing cell (L14, p=0.20, bs128) fails **both** legs. So §15 point 2's *"clears its null in 209 of 210
cells"* is correct, not a conjunction quietly relabelled. `margin > generalisation_gap_in_minus_out` =
**195/210** under either margin definition. §12.4's "Part II's two counts are reproduced here unchanged,
and the Controller's F1 is upheld" is accurate.

---

## P4 — PASS. The cross-process reproduction is real (see F9 for what its cited basis did not cover)

| arm | 08-30 `d0_carved_bpb.json` | 09-03 `d0_carved_bpb_paired.json` | diff |
|---|---|---|---|
| baseline | 0.7675949601 | 0.7675949641 | +4.01e-9 |
| L27_coact | 0.8325832415 | 0.8325832592 | +1.77e-8 |
| L27_null | 0.9142022104 | 0.9142022100 | −4.4e-10 |
| all_coact | 1.8582178125 | 1.8582178498 | +3.73e-8 |
| all_null | 2.5787313710 | 2.5787313454 | −2.56e-8 |

Max |dev| 3.73e-8 (reported 3.8e-8), i.e. 7.5e-6 σ_seed. This is a genuinely strong check and stronger
than the report claims for a reason it does not mention: the two runs used **different BPB code paths**
(`C.bpb(..., batch=1)` on 08-30, a hand-rolled per-sequence loop in the paired script), so the agreement
is cross-implementation as well as cross-process. Both ran at 6 torch threads, which is what makes
fp32 reduction order comparable; had thread counts differed the check would prove more.

---

## P5 — PASS. §12.2's ARI table, and the finding it carries

| layer | mean ARI over the 28 seed-pairs | min | max |
|---|---|---|---|
| L1 | 0.4930 | 0.4541 | 0.5322 |
| L7 | 0.5286 | 0.4997 | 0.5650 |
| L14 | 0.4242 | 0.3864 | 0.4752 |
| L21 | 0.5144 | 0.4826 | 0.5722 |
| L27 | 0.4485 | 0.4141 | 0.4723 |

Reproduces to every printed digit. The finding — *"the statistic is reproducible; the expert assignment
is not"*, and that this is fatal to shipping one specific layout — is correct, is properly separated from
the determinism claim, and is the most transferable result in Part III. It is also the honest disclosure
that a lesser report would have buried, and I credit it.

---

## P6 — PASS. The B3 defect is real and the repair is load-bearing — the control the Builder did not run

§12.1's determinism check calls `seeded_clustered_order` twice back-to-back **in the same process**,
which is close to a tautology once the seed is set. It shows the repair works; it does not show the
defect was there. The planted control that does is the negative one, and it was not run. I ran it:

| layer | `manual_seed(7)` then `balanced_labels` | no `manual_seed`, same call, same process |
|---|---|---|
| L0 | identical to stored, ARI 1.0000 | **different**, ARI 0.3828 |
| L14 | identical to stored, ARI 1.0000 | **different**, ARI 0.3627 |
| L27 | identical to stored, ARI 1.0000 | **different**, ARI 0.3393 |

So B3 was a real defect of the size the previous audit said (~0.34–0.43 ARI between identical calls), and
the one-line repair is exactly sufficient. Everything in B1 and B2 above is about where that line lives
and what it can retroactively fix — not about whether it works. It works.

---

## What the artefacts do and do not support, in one place

**Supported, and I would defend these against a hostile reader:**

- The carve at `E=32, k=8` with an energy-oracle router costs **+1.09062 BPB**, paired SE 0.0698,
  z(seq) = 15.6, on the pinned 24 × 512 heldout slice. Reproduced from raw per-token nats under three
  estimators, and across two processes and two implementations to 3.7e-8.
- Co-activation ordering beats a matched random partition **end to end on BPB** by −0.72051,
  z(seq) = −6.28. First time on the decision metric. Part II's direction is confirmed.
- The correction of §11.3 from "140 sd" to "41.6 sd" is right.
- The partition is deterministic once seeded and **not identified** across seeds (ARI 0.42–0.53).
- +1.09 BPB is 218 σ_seed, measured at the kindest cell in the probe and at a budget 12× more generous
  than the target. **The negative verdict on FFN co-activation carving at 1.5B survives this audit
  intact.** Nothing below changes it.

**Not supported as written:**

- "B3 is closed" and "every number in Parts I and II is re-derivable" (B1, B2).
- "190 of 210" as a measurement (B3) — the defensible count is 70/70 at `p = 0.10`.
- "the 11.12 sentence is withdrawn" (B4).
- "no routing improvement can move it" (F5), and "F8 partly answered" (F6).
- "the damage compounds with depth" (F3); "1.4× to 200×" (F2); "18 of the 20" (F1).

---

## For the Principal — what I judge needs a decision, in priority order

1. **F13, and it is the only forward-looking item.** The 46.5 / 50.5 / 71.7% scale curve is uncited in
   Part III and uncited in `OWNER_DECISION_NOTE_03.md`. It is the premise of §15's "what stays open is
   scale", and the S1 scale arm was made mandatory on it (`f90ac3d`). Under this programme's own rule
   this number cannot carry a gate until it is read in a table. **Decision: source it, or restate the
   scale argument qualitatively and re-derive the S1 amendment's justification.**
2. **B1 — no Part III number has a committed producer.** `b3_sweep.py` and `carved_bpb_paired.py` are in
   a session-temp directory that is deleted without warning, and the repo pipeline is still unrepaired.
   **Decision: commit both scripts and move the seeding into `d0_layout.py`, before the temp directory
   is collected.** This is time-sensitive in a way the others are not.
3. **B3 — how the 190/210 is retired.** My recommendation is to replace it with "70/70 at `p = 0.10`,
   undefined elsewhere", not to re-report it with a better imputation. It is not a decision I should take.
4. **B4 — whether §11.12 and §11.7b get annotated in place.** The document is closed as of §15; if it is
   closed with a known-false sentence in its body and the erratum 300 lines later, that is what a future
   reader inherits.
5. **F6 — the carve harness has no control.** One extra arm (`k = E = 32`, expect BPB ≡ baseline) closes
   the identity control for ~6 minutes of CPU. Two more (`k = 16`, `k = 24`) would give a graded positive.
   Cheap, and it is the same class of gap the previous audit flagged and this one finds again.

**Not a decision, an observation.** The previous audit found three over-statements all pushing the same
way — widening the negative. Part III corrects one of them honestly (§12.3) and then introduces four more
pushing the same way (B3, F1, F3, F5). The failure mode is stable across authors and across parts: the
tables are right, and the sentence that summarises them reaches further than they do.
