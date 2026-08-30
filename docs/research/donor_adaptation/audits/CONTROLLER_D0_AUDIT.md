# Controller audit — D0 co-activation, Parts I and II

**Auditor:** the Controller (independent adversarial review) · **Date:** 2026-08-29
**Target:** `docs/research/donor_adaptation/probes/D0_COACTIVATION.md` (Parts I and II)
**Instrument:** `benchmarks/donor_adaptation/density/d0_coactivation.py` (+ `d0_layout.py`, reused)
**Artefacts:** `benchmarks/donor_adaptation/density/results/d0_coactivation.{controls,model,analyse}.json`,
`d0_coactivation.json`, `d0_{controls,model,analyse}.log`, `d0_analyse.wallclock.txt`,
`d0_peak_rss.txt`, `d0_masks/` (56 npz, 4.6 GB)
**Brief:** `briefs/BRIEF_D0_COACTIVATION_PERMUTATION.md` + Amendment 1
**Cross-referenced:** `probes/F1_GATE_PREDICTOR.md`

**Role limits.** I do not implement fixes, commit, push, or adjudicate direction. Verdicts below are
BLOCK / FLAG / PASS with a reproduction path for each.

**Machine state I observe myself**, 2026-08-29 ~21:25 local, before and after my own work:
`llama-server.exe` pid 31092, 20.62 GB working set, started 19:34:40; `League of Legends` 1.78 GB;
`LeagueClient` 1.12 GB; `Memory Compression` 1.74 GB. **I took no timing numbers.** My only compute was
three repeat `clustered_order` calls (≈5 s each, 4 threads) over mask files already on disk, plus JSON
and log parsing. Nothing I ran can have perturbed another Builder's measurement beyond four thread-seconds.

---

## 0. Verdict summary

| # | finding | verdict |
|---|---|---|
| B1 | `FFN_active` "never goes below 0.5097 anywhere in 210 cells" — the true minimum is **0.3911** | **BLOCK** |
| B2 | §11.7b's "per-token bytes, lossless" column is not a per-token byte count; two of its five entries are below a hard analytic bound | **BLOCK** |
| B3 | The co-activation arm is **not reproducible at fixed seed** — the rank-64 sketch feeding k-means is drawn from an unseeded RNG | **BLOCK** |
| F1 | §11.12 attributes the 195/210 count to the wrong test (the section-6 row-1 count is 209/210) | FLAG |
| F2 | §11.7a "79 of 80" saturate — it is 72 of 80 | FLAG |
| F3 | The block-size axis mixes two incompatible layout models; only block 64 is self-consistent for this donor | FLAG |
| F4 | The 0.3333 floor is unattainable at these densities; the real bound is `FFN_active ≥ (1+2p)/3` and it is never stated | FLAG |
| F5 | §11.5's "the structure vanishes at block 140" is an absolute-margin artefact; in ratio the advantage rises monotonically | FLAG |
| F6 | `coactivation_best` is a max over four arms selected on the held-out half; the gap floor is blind to it | FLAG |
| F7 | C3's tolerance defect is real but changed nothing; the unreported defect is that C3 has no fit/score split | FLAG |
| F8 | Three of seven controls are tautologies; C4(iv) compares a matrix to itself; no positive control exists for the fidelity, geometry or duplication paths | FLAG |
| F9 | Memory: exactly one stage is measured; the external 11.186 GB cannot verify the `model` estimate | FLAG |
| F10 | Provenance: three unrelated git stamps, all dirty; sklearn unpinned; the document's "nothing committed" is stale | FLAG |
| F11 | Contamination: stated correctly; D0 is a plausible contaminant of others, not a victim | FLAG |
| F12 | The `p_max` cap does not touch L27's margin; it biases the realism check, in the flattering direction | FLAG |
| P1 | JSON↔log cross-check independently reproduced, 0 mismatches — and its coverage is a fraction of the reported numbers | **PASS** (scoped) |
| P2 | The withdrawn 5.742× over-generalisation: withdrawal is complete and correct | **PASS** |
| P3 | The identity arm and the structureless reference | **PASS** (with a wording caveat) |
| P4 | Achieved densities, token counts, seed count, corpus disjointness, C1 losslessness | **PASS** |

**Three independent over-statements (B1, F4, F5) all push in the same direction — they make the
negative look wider than the data makes it.** That is the lean I was told to look for and it is present.
Against it: every tabulated per-cell number in the document reproduces the JSON exactly, the Builder's
own cross-check is honest, and it self-reported five real defects. **The failure is in the summary
sentences, not in the measurement.**

---

## B1 — BLOCK. "`FFN_active` never goes below 0.5097 anywhere in 210 cells" is false

§11.12 point 1, the first of the four reasons given for not proceeding, reads:

> **`FFN_active` never goes below 0.5097** anywhere in 210 cells

**Measured over all 210 cells: the minimum is 0.39106** (L27, p = 0.05, block 2), and the minimum at
p = 0.10 alone is **0.4455** (L27, block 2). 0.5097 is the minimum of the **25-row condensed table** in
§11.1 — five layers × five block sizes at p = 0.10 only. §11.1 states it correctly ("the best cell
anywhere in **this table**"); §11.12 promotes it to a claim about the full sweep, which the artefact
does not support.

This matters because the sentence is doing load-bearing work: 0.5097 against 0.3333 reads as "the route
is nowhere near its floor", and 0.3911 against 0.3333 does not. See F4 — the correct comparison makes the
gap smaller still.

**Reproduction:**
```
python -c "import json;d=json.load(open('benchmarks/donor_adaptation/density/results/d0_coactivation.analyse.json'));c=[(v['coact_FFN_active_dense_gate'],L,p,b) for L,r in d['deep_layers'].items() for p,q in r['densities'].items() for b,v in q['verdict_by_block'].items()];print(len(c), min(c))"
# -> 210 (0.39105975741431825, '27', '0.05', '2')
```

---

## B2 — BLOCK. §11.7b's "per-token bytes, lossless" is not a per-token byte count, and two entries are impossible

§11.7b's table is the document's only favourable quantitative result and the only place anything beats
the dense gate. Its fourth column is computed as `(k/E) × multiplier + E/(3·d_ffn)`. I reproduced all
five rows to 5 decimal places from the JSON, so the arithmetic as written is correct. **The arithmetic
as written is the defect.**

`weight_inflation_multiplier_greedy` is `(1 + dup_extra).mean()` over all `d_ffn` neurons
(`d0_coactivation.py:986`) — the **mean** post-duplication expert size is `N·multiplier/E`. Multiplying
`k/E` by it prices a token as if it loaded `k` average-sized experts. But `duplication()` defines
losslessness as: on every token, every active neuron is in one of the loaded experts
(`d0_coactivation.py:947–970`). At `p = 0.10` every token has **896 active neurons**, so the union of the
loaded experts must contain at least 896 neurons, i.e.

> **per-token expert bytes ≥ p = 0.10, for any `E`, any `k`, under losslessness.**

Against that bound:

| L | best cell | reported "per-token bytes, lossless" | ≥ 0.10 required |
|---|---|---|---|
| 1 | E=32, k=1 | 0.1184 | above the bound, but by the same broken formula |
| 7 | E=16, k=1 | 0.2786 | same |
| 14 | E=128, k=1 | 0.3979 | same (and it is the negative row) |
| 21 | E=128, k=1 | **0.0268** | **IMPOSSIBLE** |
| 27 | E=64, k=1 | **0.0343** | **IMPOSSIBLE** |

Duplication is concentrated in exactly the experts the oracle keeps loading, so the mean is the wrong
statistic. The instrument's own docstring says so and the document changed the unit:
`weight_inflation_note` reads *"an expert-active fraction f costs f × multiplier in **resident-weight**
terms"* — the table's header says **per-token bytes**.

Corroborating evidence that the loaded set is degenerate: at L21/E=128/k=8 the naive duplication count is
83.39 while the greedy cover is 1.99 — about **two** experts are loaded on essentially every token that
misses anything. `missed_active_fraction` at the five "best" cells is 0.78 / 0.84 / 0.97 / 0.91 / 0.74:
the oracle top-k router drops three-quarters to 97% of active neurons and duplication puts them back.
**The cells that look best are the ones where routing does least work.**

**Consequences.** "Four of five layers beat the dense-gate floor on this metric" is unsupported. The
supportable statement is the opposite in spirit: a lossless top-k carve is bounded below by `p + router`
≈ 0.105 per token, i.e. **at best ≈ 3.2× better than the dense gate, never 9.7× or 12.5×**, and the
instrument does not measure where in `[0.105, 1]` any layer actually sits. §11.7b's three caveats
(in-sample cover, oracle router, resident capacity) are all real and none of them is this one.

**Reproduction:**
```
python -c "
import json;d=json.load(open('benchmarks/donor_adaptation/density/results/d0_coactivation.analyse.json'));D=d['deep_layers'];N=8960
for L in ['1','7','14','21','27']:
  g=[(int(k)/int(E)*r['weight_inflation_multiplier_greedy']+int(E)/(3*N),E,k,r['missed_active_fraction']) for E,ks in D[L]['densities']['0.10']['duplication'].items() for k,r in ks.items()]
  print('L',L,min(g))"
# L21 -> (0.02677..., '128','1', 0.9124)   L27 -> (0.03433..., '64','1', 0.7405)   both < 0.10
```

---

## B3 — BLOCK. The co-activation arm is not reproducible, at fixed seed, on the same input

§11.0 asserts the stage "is deterministic given the pinned masks and the pinned seeds". It is not.

`CLUSTER_SEED = 7` is passed to `KMeans(random_state=seed)` only. The embedding k-means clusters comes
from `torch.svd_lowrank(X, q=64, niter=4)` — `d0_layout.py:143` (`clustered_order`) and `:182`
(`balanced_labels`). That routine draws a random Gaussian sketch from **torch's global RNG**, and
`torch.manual_seed` is never called anywhere in `d0_coactivation.py`, `d0_layout.py` or `common.py`.

Measured on the donor's own mask files, `E = 64`, `p = 0.10`, `CLUSTER_SEED = 7` held fixed, three
repeat calls, same input array:

| | ARI between repeats | s @ bs 12 | s @ bs 21 | s @ bs 64 | s @ bs 140 |
|---|---|---|---|---|---|
| **L21** | 0.51 / 0.50 / 0.56 | spread **0.0043** | spread **0.0046** | 0.0011 | 0.00006 |
| **L27** | 0.43 / 0.43 / 0.42 | spread **0.0051** | spread **0.0110** | 0.0066 | 0.0022 |

**Two runs of the arm under test agree on barely half the partition and the reported statistic moves by
up to 0.011.** The document reports these values to four and five decimals and states their error bar as
the **null's** sd, 0.00079 — "roughly 140 sd above the null mean" (§11.3). The treatment's own spread at
the block sizes carrying the result is **5× to 14× the null's sd**, and it was never measured. This is
the Principal's suspicion, confirmed and worse than stated: §11.13 frames the problem as "a k-means
restart could move it", implying a deliberate `CLUSTER_SEED` sweep would be needed. **No seed change is
needed. Re-running the identical command moves it.**

**Does it overturn the direction?** No, at the large margins: L27/bs21's +0.276 and L21/bs21's +0.092
dwarf ±0.011. It does bear on the small ones — **43 of 210 margins are below 0.005 and 58 below 0.011**,
and the 15 cells §11.4 reports as failing their gap floor sit inside that band with no valid error bar
at all. It also means the pipeline's dominant noise source is invisible to every test the document
applies: `generalisation_gap_in_minus_out` uses the **same** partition for its in-sample and
out-of-sample halves, so it cannot see sketch variance by construction.

**Reproduction** (CPU-only, ≈15 s, no donor, masks already on disk):
```
python - <<'EOF'
import sys,numpy as np,torch
sys.path.insert(0,'benchmarks/donor_adaptation/density'); import d0_layout as D0
from sklearn.metrics import adjusted_rand_score
z=np.load('benchmarks/donor_adaptation/density/results/d0_masks/fit_L27.npz')
idx=z['idx'][:,:896].astype(np.int64); B=np.zeros((idx.shape[0],8960),bool)
B[np.arange(idx.shape[0])[:,None],idx]=True
_,a=D0.clustered_order(torch.from_numpy(B),64,7); _,b=D0.clustered_order(torch.from_numpy(B),64,7)
print('ARI at fixed CLUSTER_SEED=7:',adjusted_rand_score(a,b))   # -> ~0.43, not 1.0
EOF
```
Minimal mechanism check, no donor and no masks: `torch.svd_lowrank(torch.randn(300,200),q=16,niter=4)[2]`
twice on the same input returns different matrices (max |Δ| ≈ 0.37 here).

---

## F1 — FLAG. §11.12 attributes 195/210 to the wrong test

§11.12's section-6 table reads:

> co-activation beats identity **and** clears the random arm's seed spread — **YES** — in 195 of 210 cells

195/210 is the count from §11.4: cells where `margin > generalisation_gap_in_minus_out`. The row-1
criterion is `coact_beats_identity AND coact_clears_random_spread`, and that count is **209 of 210**.
The single failure is L14 / p = 0.20 / block 128, where identity, random and co-activation are all
exactly 0.0. Both counts are individually correct in the JSON; the sentence pairs one with the other.

The conflation has left the document: commit `30efdd9`'s subject is *"the co-activation structure is
real and clears its null in 195 of 210 cells"*.

Related, minor: §11.4 says the 15 failures are all "at block ≥32 where the skippable fraction has
collapsed to zero for all three arms". Block ≥32 is right; "zero for all three arms" is not, at three of
them — L7/p0.20/bs32 reads identity 0.000775, random_max 0.000794, co 0.002029, and the document's own
table prints 0.0020 and 0.0008 two paragraphs above the sentence.

**Reproduction:** count `v['coact_beats_identity'] and v['coact_clears_random_spread']` over all
`verdict_by_block` cells → 209; count `margin > gap` → 195.

---

## F2 — FLAG. §11.7a's "79 of 80" is 72 of 80

> `experts_touched` mean equals `E` exactly in 79 of 80 … The single exception is L27 at `E = 128`,
> which reads 127.99.

**Eight of the 80 records are non-saturated, not one.** `experts_touched` is routing-free, so the four
`k_router` values at a given `(L, E)` carry the same number; the two distinct non-saturated values are
L27/E=64 = 63.99944 (×4 records) and L27/E=128 = 127.99364 (×4 records). Immaterial to the conclusion —
the metric is saturated and the argument that it discriminates nothing is correct and well made — but it
is a counted claim that does not match the artefact, in a section with no log to check against.

**Reproduction:** count `abs(r['experts_touched_all']['mean'] - E) > 1e-9` over the 80 duplication
records → 8.

---

## F3 — FLAG. The block-size axis mixes two layout models; only block 64 is self-consistent

`block_sizes_for` derives its candidates from `rho_floor_blocks` (`d0_coactivation.py:272–297`):

| d_model | per-organ (48 KiB / (d·0.5)) | interleaved (48 KiB / (3·d·0.5)) |
|---|---|---|
| 1536 (this donor) | **64** | **21.33 → 21** |
| 8192 (70B class) | **12** | **4** |

So the Principal's suspicion in claim 2 is correct and it is not a coincidence: **21 is exactly this
donor's interleaved 48 KiB block**, put into the sweep by construction. But the interleaved block is
gate+up+down for the same 21 neurons — a layout in which skipping a block skips the **gate**. Amendment
1 §A1.1, which supplies the `FFN_active = (3−2s)/3` column printed beside every one of those numbers,
says the gate is never skipped because it *is* the predictor.

**The two are not compatible.** Under the dense gate only per-organ blocks can be skipped, so for this
donor the only self-consistent granularity in the sweep is **64**; 12 and 4 belong to a hypothetical
`d_model = 8192` donor (12 per-organ, 4 interleaved and therefore also inadmissible). The document's
headline — "the permutation works at block 12–21" — pairs a granularity from one cost model with a
budget column from another.

At block 64, p = 0.10, the picture is:

| L | co `s` | `FFN_active` | identity `FFN_active` | margin | own gap |
|---|---|---|---|---|---|
| 1 | 0.0286 | 0.9809 | 0.9991 | +0.02728 | 0.00582 |
| 7 | 0.0037 | 0.9975 | 0.9992 | +0.00247 | 0.00196 |
| 14 | 0.0059 | 0.9961 | 0.9993 | +0.00471 | 0.00233 |
| 21 | 0.0133 | 0.9911 | 0.9992 | +0.01201 | 0.00272 |
| 27 | 0.1593 | 0.8938 | 0.9992 | +0.15788 | 0.01037 |

**At the only granularity this donor's engine can legally skip under a dense gate, the method buys
0.1–1.1% of FFN bytes at four layers and 10.5% at L27.** That is a stronger negative than the document's,
and it is reached without the framing error. The forward-looking sentence should read "block **12** at
`d_model = 8192`", not "4–12"; 4 is the same inadmissible interleaved number, one donor over.

**Reproduction:** `rho_floor_blocks(1536)` and `rho_floor_blocks(8192)` in the instrument, read against
`ffn_active_dense_gate`'s docstring and Amendment 1 §A1.1.

---

## F4 — FLAG. 0.3333 is not the attainable floor at these densities; `(1+2p)/3` is, and it is never stated

Every `FFN_active` in Part II is compared to **0.3333**, the value of `(3−2s)/3` at `s = 1`. But `s = 1`
means every block is skipped on every token, which at density `p` is impossible: a token with `k = p·N`
active neurons must fetch at least `k/B` of the `N/B` blocks, so

> **`s ≤ 1 − p` for any permutation, any block size, any donor — hence `FFN_active ≥ (1 + 2p)/3`.**

| p | attainable floor `(1+2p)/3` | best cell observed | cells below the floor |
|---|---|---|---|
| 0.05 | **0.3667** | 0.3911 | 0 |
| 0.10 | **0.4000** | 0.4455 | 0 |
| 0.20 | **0.4667** | 0.5502 | 0 |

Zero violations across 210 cells — that is a real sanity PASS for the instrument, and it is also the
proof that the bound binds.

**Two consequences, and the first cuts against the document's own rhetoric.**

1. "**The skippable route never approaches its own floor**" (§11.1, §11.12) is wrong. At p = 0.10 the
   best cell is 0.4455 against an attainable 0.4000 — **within 11%**. At p = 0.05, 0.3911 against
   0.3667 — within 7%. The method gets close to its real ceiling. The ceiling is simply not low enough:
   0.40 against a 0.02 target. The negative survives, for a stronger reason, stated in the wrong units.
2. **The document's own recommended decisive experiment cannot change the verdict.** §11.12's closing
   recommendation is *"a donor whose `d_model` is large enough that the legal block falls to 4–12,
   where this donor's effect lives"*. The bound `(1+2p)/3` contains no `d_model` and no block size. A
   perfect permutation on a 70B-class donor at p = 0.10 still reads `FFN_active = 0.40`. **Spending a
   second donor to test that hypothesis buys a number that is already derivable.** If the Principal
   wants the skippable route closed, this is the argument that closes it, and it costs nothing.

---

## F5 — FLAG. §11.5's "the structure has vanished at block 140" is an absolute-margin artefact

§11.5 is labelled *"the pivot of the whole result"*. It reads the margin `co − random` across block size
and observes a peak at 12–21 and collapse by 140. Both arms decay like `(1−p)^B`, so their **difference**
must peak and then collapse whatever the structure does. In ratio to the structureless reference the
advantage does the opposite — it rises monotonically with block size at every layer:

`coactivation_best / q^B`, p = 0.10:

| block | L1 | L7 | L14 | L21 | L27 |
|---|---|---|---|---|---|
| 4 | 1.06× | 1.02× | 1.01× | 1.06× | 1.12× |
| 12 | 1.40× | 1.11× | 1.12× | 1.34× | 1.85× |
| 21 | 2.10× | 1.26× | 1.34× | 1.86× | **3.54×** |
| 64 | 24.3× | 3.12× | 4.99× | 11.3× | **135×** |
| 140 | 4909× | 309× | 452× | 488× | **∞** (ref = 0 to double precision) |

**At block 140 — expert granularity, the granularity the document says the structure is absent at —
co-activation is 300× to 5000× the structureless reference at every layer.** What vanishes is not the
structure; it is the *absolute quantity of skippable byte traffic available at that block size*, which is
capped by `(1−p)^B` for the null and by `1−p` overall (F4), and which no permutation can lift.

The correct statement — still a negative, and a cleaner one — is: *the structure is present at every
granularity and grows relative to chance as blocks get coarser; the byte saving it can express shrinks
faster, for arithmetic reasons independent of the donor.* §11.12's point 2, "at expert granularity the
effect is gone", is false as written; "at expert granularity there is nothing left to save" is true.

**Reproduction:** divide `coactivation_best` by `structureless_ref_qB` in `verdict_by_block` and read
down the block axis.

---

## F6 — FLAG. `coactivation_best` is a max over four arms selected on the held-out half

`d0_coactivation.py:1113`:
```
best_E = max(E_LIST, key=lambda E: dens["arms"][f"coactivation_E{E}"][str(bs)]["s"])
```
`dens["arms"]` holds the **score-half** values. So the treatment reported in every cell is the best of
four correlated arms, chosen on the data it is then reported against. The selection bonus (max minus
mean over the four `E`) has **median +0.00624 across 210 cells and reaches +0.043** — the same order as
the `generalisation_gap_in_minus_out` floor §11.4 applies (median 0.00619). And the gap is computed for
the already-selected `E` using the same partition on both halves, so it does not price this at all.

Partly offset: the null is also a max, of eight seeds. The net is second-order next to B3, but it means
the §11.4 clearance test is anti-conservative in two independent ways (test-set selection over four arms;
blindness to sketch variance) and conservative in one (max-of-8 null).

---

## F7 — FLAG. C3's tolerance defect is real, changed nothing, and hides a larger one

The Builder's self-report is accurate: `d0_coactivation.py:471` applies `abs(v_co − hg) < 0.03` to the
co-activation arm against `< 0.02` for identity and random.

**But the defect did not change this run's outcome.** The measured co-activation excess over the
hypergeometric on the i.i.d. mask is `+0.00139` (bs 4), `+0.00641` (bs 12), `+0.00788` (bs 21),
`+0.00092` (bs 64), `+3.8e-6` (bs 140). C3 fires at 0.03, and would fire at 0.02 and at 0.01. The
tolerance is ~4× the largest observed excess, so the control **cannot fail** as written — that is a real
design defect and it is correctly reported as unfixed — but no verdict in this document turns on it.

The +0.008 inference Part I drew, and the +0.01081 the donor measured at L1/p0.10/bs12, are both
confirmed against the artefacts. They are the same order and the same sign, as claimed.

**The unreported and larger defect: C3 has no fit/score split.** Its co-activation arm is fitted and
scored on the same `idx_n`. The whole methodological claim of this probe is that a *static* ordering
fitted on one half serves a disjoint half. **There is no planted-negative control for the pipeline the
donor run actually uses.** C3's +0.0079 is an in-sample number; the donor's +0.0108 is an out-of-sample
gap; they are the same class of quantity but not the same quantity, and the document treats the
agreement as corroboration of the method rather than of the arithmetic.

---

## F8 — FLAG. Three of seven controls are tautologies; the paths carrying B2 have no control at all

Asked to separate real tests from tautologies of the arithmetic:

**Real, and informative**

- **C1 (donor losslessness).** The strongest control in the probe and correctly run first.
  BPB 0.767594960 → 0.767594965 → 0.767594960 exact on restore; ΔBPB = 5.09e-9; max|Δlogit| = 1.278e-4
  against |logit|max = 31.24, i.e. 4.09e-6 relative; matches D1's published 0.767595 to 6 dp. It
  establishes that the permutation machinery is output-preserving on the real donor, which is the one
  thing everything else rides on.
- **C2 (planted positive).** Fires: recovery ratio 1.0000, ARI 1.0000, identity-on-scrambled 0.0000.
  **Note in the Builder's favour, and it is load-bearing: C2 plants its structure at group size 140 —
  exactly expert granularity — and the instrument recovers it perfectly.** So the block-140 negative on
  the donor is not an instrument that cannot see coarse structure. That licence is real and I credit it.
  Its limit: the plant is maximal-SNR (disjoint groups, whole groups fired, no noise, exact recovery),
  and there is **no graded-difficulty positive**, so nothing calibrates sensitivity anywhere near the
  donor's regime.
- **C4(i)** — the un-permute round trip, `np.array_equal(Mb[:, ppos], Ma)`. A genuine test of the
  `pos_from_order` gather direction, which is the one thing in the arm plumbing that could silently be
  backwards. Good control.
- **C5** — fast index path vs `D0.layout_stats`, agreement `0.0` at all five block sizes. Genuine
  cross-implementation check.
- **C6** — hypergeometric closed form vs Monte-Carlo. Genuine, but tests only mathematics, and its
  tolerance (<0.01) is ~4.5 standard errors at 20 000 trials, so it too cannot fail meaningfully. Two of
  its four rows sit at 1.9–2.3 se (bs 21: 0.10913 vs 0.10405; bs 64: 0.001150 vs 0.0016).

**Tautologies**

- **C7** checks `(3−2s)/3` against four hand-copied values of `(3−2s)/3`. Two of the four `abs_diff`
  entries (6.7e-4, 3.3e-4) are nothing but the brief's 3-dp rounding. Zero information, and it is counted
  in `all_fired`.
- **C4(ii)** per-token active count and **C4(iii)** firing-frequency multiset are true by construction of
  `idx_perm = ppos[idx_n]`; a permutation cannot change either.
- **C4(iv)** is worse than a tautology. `Fa = Ma[:, sub]` and `Fb = Mb[:, ppos[sub]]`; since
  `Mb[:, ppos[n]] ≡ Ma[:, n]`, **Fb is elementwise identical to Fa** and the eigenvalue comparison
  compares a matrix to itself. On disk `coactivation_spectrum_max_abs_diff` is exactly `0.0`. The
  document and the instrument both present four measured invariances; one is measured.
- **C3's identity and random rows** are guaranteed by construction on an i.i.d. mask; only its
  co-activation row tests anything (see F7).

**Absent**

- **No positive control for `routed_error`** — the whole of Part I's 120-row fidelity table. The Builder
  reports this gap itself (§11.13) and is right to.
- **No control of any kind for `moe_geometry` or `duplication`.** §11.7b — the table carrying B2 — has
  no planted positive, no planted negative, no closed-form reference and no log to cross-check against.
  It is the least-controlled measurement in the probe and it is the one carrying the only favourable
  result.

**Standing rule on this programme is that an instrument must be shown firing on a known positive before
its nulls count.** That rule is satisfied for the permutation/skippable path (C2) and **not** satisfied
for the fidelity, geometry or duplication paths.

---

## F9 — FLAG. What is actually established about this probe's footprint

Exactly one number: **`analyse` peaked at 2.749 GB**, against a 2.6 GB analytic estimate — 5.7% under.
That is a real measurement and it does validate the inventory's *method*, on the cheapest of the three
stages.

`controls` and `model` wrote `{"rss_gb": 0.0, "peak_rss_gb": 0.0}` at every checkpoint; the `_rss_gb()`
`restype` fix post-dates them. Their 0.8 GB and 9.6 GB estimates are unverified and cannot now be
verified without re-running.

`d0_peak_rss.txt` contains, BOM-prefixed, `peak_rss_gb=11.186 at 2026-08-29T01:51:35+02:00` and
`FINAL peak_rss_gb=11.186`. **No PID, no producer script, not referenced anywhere in the instrument, not
written by it.** It is unattributable in both directions: the `model` stage's own snapshot records
`llama-server` at 22.73 GB resident with 37.13 GB free at stage start, so 11.186 is consistent with a
per-process sampler on D0's python, with one on something else, and with a whole-machine delta. The
document's characterisation ("external sampler with no PID and no producer script … unattributable") is
correct, and I would go one step further: **it should not be quoted as a bound on anything.**

Nothing here threatens a result. The reason it belongs in the audit is that `model`'s 9.6 GB is the
figure that would license a future all-layer capture on a bigger donor, and it remains a projection.

---

## F10 — FLAG. What the manifest pins, and what it does not

**Pinned and verified:** donor at revision `8faed761d45a263340a0528343f099c05c9a4323`; architecture
achieved `n_layers 28 / d_model 1536 / d_ffn 8960 / hidden_act silu`; python 3.12.10, torch 2.12.0+cpu,
numpy 2.4.6, transformers 4.57.6, Windows-11-10.0.26200-SP0; `D_THREADS = 6` requested and
`torch_num_threads_achieved = 6`; command lines; all pinned seeds; slice sha256s.

**Not pinned:**

- **Git revision is worthless here, and there are three of them, not two.** `controls` →
  `3471116` ("R2a apparatus and raw results"), `model` → `1a8f512` ("R2a verdict"), `analyse` →
  `3f885c5` ("P1 pre-registration"). All three are unrelated HEADs and all three recorded
  `git_dirty_at_launch: true`, so the stamp pins nothing about the code that ran.
- **sklearn's version is unrecorded**, and `KMeans` is on the critical path of every co-activation
  number in both Parts (`clustered_order`, `balanced_labels`). Combined with **B3**, the arm under test
  has neither a pinned RNG nor a pinned library.
- **torch's global RNG state** — see B3. This is the one that actually bites.

**Correction to the document's own provenance claims.** §11.13 says the instrument is "STILL UNTRACKED"
and the closing paragraph says "Nothing in this document has been committed or pushed". Both are stale as
of the file on disk: `d0_coactivation.py` and all ten `d0_*` artefacts were committed at **`30efdd9`,
2026-08-29 16:42:18**, four minutes after the document's own mtime (16:38). `git diff HEAD` on the
instrument is empty, so the committed file is the one that ran — the provenance is now better than the
document says. The commit subject, however, carries F1's conflated claim into the permanent record.

---

## F11 — FLAG. Contamination — stated correctly, and the direction runs the other way

Recorded by the stages themselves:

| stage | UTC | resident heavyweights | free RAM |
|---|---|---|---|
| `controls` | 2026-08-28T23:35:17Z | `llama-server` 22.73 GB | 37.13 GB |
| `model` | 2026-08-28T23:36:11Z | `llama-server` 22.73 GB | 37.13 GB |
| `analyse` | 2026-08-29T09:36:47Z | second `python` pid 26112, 0.89 GB | 61.13 GB |

**No D0 number can be corrupted by any of this.** Every stage is a deterministic function of its inputs —
except through B3, which is an RNG defect and not a contention effect. The document's §11.0 correction
(it was told the machine was clear; a second python was resident) is accurate and correctly reported
rather than smoothed over.

**The exposure runs outward, not inward.** The `model` stage held ~6 GB of fp32 donor weights and ran
three full BPB passes over the 24×512 eval slice — 196.8 s, 225.0 s, 146.5 s by its own log — plus a
28-layer × 2 token-set capture, inside a window when `llama-server` was resident at 22.73 GB. **D0 is a
plausible contaminant of any bandwidth or timing number taken in that window.** It recorded its window
precisely and said so; that is the right behaviour, and other probes' timing audits should check against
it.

---

## F12 — FLAG. The `p_max` cap does not bias L27's margin; it biases the realism check, flatteringly

The question was whether L27's 4.578% cap rate contaminates the layer carrying the positive result.

**It does not touch the skippable table.** Those masks are exact top-`k` with `k ≤ 1792` (p = 0.20), well
inside the stored `P_MAX·d_ffn = 2240` ranked indices, so no cell in the 210 is censored.

**It biases the TEAL/CATS realism check, and it understates the problem.** `per_tok = (V >= thr).sum(1)`
runs over the stored top-2240 only (`d0_coactivation.py:1051–1053`), so any token whose true
above-threshold count exceeds 2240 is recorded as 2240, i.e. density 0.25. At L27 that happens on 4.578%
of tokens. **So L27's reported mean 0.11581 and p90 0.17943 are lower bounds and the true upper tail is
wider than printed.** A realisable global threshold on L27 is *less* well behaved than §11.9 shows.

§11.9's sentence — "L27 is also the one layer carrying the entire positive result, so its capture is the
one most affected by the storage cap" — worries in the right place but names the wrong mechanism: the
capture is unaffected; the realism check is. The document is right that the sensitivity of the skippable
table to the real threshold spread is unmeasured, and it remains the cheapest unmeasured thing here.

---

## P1 — PASS (scoped). The cross-check is real, independently reproduced, and covers a fraction of the numbers

I re-parsed the logs and JSONs from scratch rather than accepting the claim:

| comparison | fields | mismatches |
|---|---|---|
| `d0_analyse.log` deep rows (15 × 7: identity, random_mean/min/max, coactivation, ref, FFN_active) | 105 | **0** |
| `d0_analyse.log` depth sweep (28 × 5) | 140 | **0** |
| `d0_model.log` fidelity rows (**all 120** × act_true, relerr) — *not checked by the Builder* | 240 | **0** |
| document §11.1 table vs JSON (23 parsed rows × 7 fields) | 161 | **0** |
| document §11.6 table vs JSON (28 × 7, incl. `cluster cv`) | 196 | **0** |
| document §11.8 table vs JSON (25 × 5, incl. the `48KB-legal` boolean) | 125 | **0** |
| document §11.9 table vs JSON (5 × 5) | 25 | **0** |

`d0_coactivation.json` is byte-identical to `d0_coactivation.analyse.json`, sha256
`b024cc8df7822a2a…`, verified. **Every tabulated per-cell number in the document reproduces its artefact
exactly.** After the fabrication episode on this programme, that is worth stating plainly: I looked for
drifted numbers and there are none in the tables.

**And the coverage is the point.** `d0_analyse.log` prints only block 12 — 15 of 210 cells — plus the
28-row depth sweep. It carries **nothing** for the other 195 cells, nothing for the duplication grid,
nothing for the MoE geometry beyond `E = 64` in the depth sweep, and nothing for the threshold check.
Every error I found (B1, B2, F2) is a prose claim in the region the log cannot reach. **The JSON↔log
cross-check is a necessary control and it is not sufficient; the tables were never the risk.**

---

## P2 — PASS. The withdrawn 5.742× over-generalisation

Verified complete and correctly withdrawn. `deep_layers.1.densities.0.10.duplication.64.4` reads
`weight_inflation_multiplier_greedy = 5.742076`, `4/64 = 0.0625`, product `0.358880` — matching the
withdrawn figures to 6 significant figures for that cell, as the document says. I searched Parts I and II
for any downstream number resting on the general version and found none; the interim note is not in this
document's chain.

**Caveat: the replacement is B2.** The withdrawal was honest and the correction went in the right
direction; the corrected table is wrong for a different reason.

---

## P3 — PASS (with a wording caveat). The identity arm and the structureless reference

Claim 3 asked whether the reference is itself correct, or trivially matched.

**It is correct, and it is not the easy choice.** `analytic_iid_skippable = (1−p)^B` is the exact closed
form for an i.i.d. mask at density `p`, and the brief §4 names those exact values (0.2824 @ 12, 0.6563 @
4). The alternative — the exact-`k` hypergeometric `C(N−k,B)/C(N,B)`, arguably the *more* correct
reference given the masks have exactly `k` active per token — is also computed by the instrument, and it
gives a **larger** maximum identity deviation (0.003727 vs 0.003306). So the reference is not flattering
the identity arm; the stricter reference makes identity look marginally worse.

**Max |identity − q^B| = 0.0033064** at L27 / p = 0.05 / block 21 — the Principal's figure, reproduced
exactly. Identity tracks the reference at every one of 210 cells.

**Wording caveat.** The claim is stated in absolute units against no yardstick. Measured against the
random arm's own seed-to-seed sd at the same cell, identity sits **>3 sd from the closed form in 5 of
210 cells**, positive in 4 of the 5 (largest: L7/p0.10/bs140 at +5.1 sd; L21/p0.20/bs2 at +4.0 sd;
L1/p0.05/bs64 at +2.6 sd on a 6% relative excess). Absolutely this is negligible — ≤0.0023 on values of
0.04 to 0.66 — and it changes nothing. But "the donor's native ordering carries **no** block structure"
is a stronger statement than the measurement supports. **"Carries no *usable* block structure" is what
was measured**, and it is enough for the conclusion.

The related claim in §11.3 checks out: the random arm's residual from the closed form exceeds 3 standard
errors in only 2 of 210 cells, and the quoted cell (L1/p0.10/bs12: mean 0.28246, sd 0.00079, residual
2.65e-4) is representative rather than cherry-picked. **The null is genuinely null. It is the treatment
that has no error bar (B3).**

---

## P4 — PASS. Achieved quantities, disjointness, seeds

- Density achieved **exactly** 0.050000 / 0.100000 / 0.200000 (`k` = 448 / 896 / 1792 of 8960) on all 15
  rows; `T_fit = T_score = 16384` on all 15.
- Fit half `calib` (n_seq 32, seed 42424), score half `heldout` (n_seq 32, seed 909001), BPB eval slice
  imported from `d1_pruning` (heldout, 24×512, seed 1234). Distinct corpus sha256 and distinct ids sha256
  asserted in code before anything else runs (`d0_coactivation.py:683–688`).
- 8 random seeds against the brief §7.4 minimum of 5.
- 14 block sizes, all derived from `rho_floor_blocks`, none hardcoded to this donor — brief §8 satisfied.
- Elapsed 11:36:42 → 12:02:23 = 1541 s, consistent with the per-layer 251/264/326/295/252 s in the log.
- `E=128, k=2` reads 2/128 + 128/26880 = **2.039%** of FFN weight bytes: the document's 2.04% is right.
- All 60 matched fidelity pairs have the co-activation partition ahead of the null; I checked all 60,
  none is negative or zero.
- `missed_active_fraction` range 0.07767 (L27/E16/k8) to 0.96991 (L14/E128/k1), and the `E=64,k=4` row
  0.777 / 0.860 / 0.860 / 0.711 / 0.527 — all as printed.
- Fidelity note, not a defect but worth recording: `Xsc` is the **first 4 sequences** of the score slice
  (`sc_pref = ids_sc[:2048//512]`), so all 120 fidelity rows rest on 4 documents, not a random draw of
  2048 tokens. The skippable table uses all 32.

---

## Claim 5, answered directly: do the per-layer cells that beat the floor survive their own caveats?

**No. None of the five survives, and two are arithmetically impossible.**

| cell | status |
|---|---|
| L21 0.0268 (E=128, k=1) | **B2** — below the ≥ p = 0.10 lossless bound. Also: `engine_legal_48KB_at_min_size` = **false** (min cluster 23 neurons vs the 64-neuron floor) |
| L27 0.0343 (E=64, k=1) | **B2** — below the bound |
| L1 0.1184 (E=32, k=1) | above the bound, but produced by the same broken formula, so not established |
| L7 0.2786 (E=16, k=1) | same |
| L14 0.3979 (E=128, k=1) | the only row whose direction is safe — and it is the negative one. Also not engine-legal (min cluster 21) |

**`E = 128` is not engine-legal at any of the five layers** — minimum cluster 15 to 29 neurons against
the 64-neuron / 48 KiB floor — and two of the five "best" cells use it. §11.7b lists three caveats and
legality is not among them, although §11.8 applies exactly that test to the `E = 64` partitions and
flags `coactivation_kmeans` at L1 and L7 for it.

**Does the oracle router alone account for the margin?** No — it accounts for the *shape*.
`missed_active_fraction` at the five best cells is 0.78 / 0.84 / 0.97 / 0.91 / 0.74. The oracle top-`k`
router discards three-quarters to 97% of the active neurons and the greedy cover then reinstates them by
duplication. The optimisation `min (k/E)·multiplier` therefore selects `k = 1` at every layer — the
setting where the router does least and duplication does most. `dup_naive` = 83.4 against
`dup_extra_greedy` = 1.99 at L21/E=128/k=8 confirms the mechanism: about two experts are loaded on
essentially every token. **These are not MoE carves; they are "load the one big expert that contains
everything", priced as if it were an average expert.**

---

## Claim 4, attacked: is the FFN-sparsity lever closed?

### Are D0 and F1 commensurable? Yes — and they compose in the direction that matters

I checked the definitions rather than the prose. D0's `relerr` (`d0_layout.py:228`) is
`sqrt(Σ‖ŷ−y‖² / Σ‖y‖²)` with `y = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))` over the full neuron set.
F1's is `‖Y − Y(θ)‖_F / ‖Y‖_F` with the same `Y`. **Same donor, same revision, same fp32 path, same
`h`, both scored on held-out tokens, both relative to the full dense FFN output.** They are the same
quantity, and the Coordinator is right to be cautious but wrong to expect an incommensurability here.

They also compose in one direction only, and it is the useful one. **F1's selector is per-token and
unstructured, and its `|h|` oracle sees `W_u` as well** — strictly stronger than D0's coarse, static,
expert-granular selection. So F1's `a = 0.836` mean at ε = 0.01 is a **lower bound on what D0's carve
needs**. D0's operating points (`p` ∈ {0.05, 0.10, 0.20}, `k/E` ≤ 0.25) sit far inside the region F1
prices at tens of percent output error, and D0's own numbers agree layer by layer — L7 at 12.5% of
neurons reads `relerr` 0.774 where F1's oracle needs 88.3% of neurons for 0.01; L1, F1's one genuinely
sparse layer (oracle `a` = 0.133), is also D0's one layer where a 3% carve recovers 96%.

So the Coordinator's reading is fair. **One correction, and it makes D0 look worse, not better:** D0's
*skippable* table is not costed against any error tolerance at all. It takes a top-10% mask as given and
never measures what dropping the other 90% costs. Only Part I's fidelity table prices error, and only
for the MoE carve. So the headline "structure is real but never pays *in bytes*" is measured at an
operating point that F1 shows is not survivable — the byte accounting **understates** the cost.

### Where the closure is soft, and it is soft in exactly one place

**Every tolerance in this closure is a per-layer Frobenius bar, and no one has measured what the model
tolerates end-to-end.** F1 says so itself (§9). The FFN branch writes into a residual stream of unknown
relative norm, so 1% relative error in one block is not 1% of anything downstream. Nothing in D0, F1 or
memo §3c measures the currency the programme actually decides in.

**arXiv:2509.00454 [T] Table 1** — `S_gate` 25.93% at 99% accuracy retention on Qwen2.5-1.5B, our exact
donor geometry — is precisely the observation that would follow if the end-to-end bar were much looser
than 1% per-layer Frobenius: 25.93% sparsity is `a ≈ 0.74`, past F1's oracle bound of 0.836. I am not
resolving what `S_gate` is; the Researcher owns that. **My verdict depends on which reading is right,
and here is exactly how:**

- **If `S_gate` is comparable to F1's `a`:** F1's oracle bound and [T] contradict each other on the same
  donor, and the closure has a hole in a load-bearing wall. Nothing should close until that is resolved.
- **If `S_gate` is a looser, end-to-end-calibrated quantity:** F1 and [T] reconcile through tolerance,
  and the survivable operating point on this donor is `a ≈ 0.74` — **which D0 never swept.** D0's
  densities are 0.05 / 0.10 / 0.20.

That second branch splits again, and the two halves do not behave alike:

- **The skippable half gets stronger, not weaker.** At `p = 0.74` the packing bound (F4) gives
  `FFN_active ≥ (1 + 2·0.74)/3 = 0.827`, and `(1−p)^B` at block 21 is ~1e-12. Nothing survives at any
  block size, any donor width, any permutation. **I would let the Principal close the FFN
  block-skippability lever** — but on F4's argument, which is analytic and donor-independent, not on
  "0.5097 never approaches 0.3333", which is B1 and F4 both.
- **The MoE-carve half is not closed by this evidence.** Its budget arithmetic is B2. Its two best cells
  are B2 and the other two rest on the same formula. Its partitions at `E = 128` are engine-illegal. Its
  instrument (`duplication`, `moe_geometry`, `routed_error`) has no positive control, no negative
  control and no closed-form reference (F8). And its quality cost has **never been measured in BPB** —
  while C1 already built and ran exactly that instrument, three full BPB passes at 146–225 s each, on a
  model whose FFN was modified in place. **One carved configuration, one BPB pass, sub-hour on existing
  code, and it is not in the document's next-steps list.**

**Verdict on claim 4: the four-way closure holds for FFN block-skippability and does not hold for the
MoE carve.** The half that is closed can be closed more cheaply and more decisively than the document
proposes. The half that is open is the expensive one, and it is the one the negative would close.

### The convenience test

Three errors, all in the same direction:

1. **B1** — 0.5097 quoted instead of 0.3911 makes the negative look wider.
2. **F4** — the unattainable 0.3333 used as the yardstick instead of the attainable 0.4000 makes the
   negative look wider.
3. **F5** — the absolute margin used instead of the ratio makes the structure look absent at expert
   granularity when it is 135× the reference there.

None is fabrication, all three are in the prose rather than the tables, and each has a benign reading.
But three independent over-statements with the same sign, in a document whose recommendation is
"do not open the brief", is the pattern I was asked to look for. **The recommendation survives my audit;
three of the four reasons given for it do not survive in the form given.**

---

## What would settle what, in priority order (for the Principal, not a directive)

1. **Seed the RNG and re-run `analyse`.** `torch.manual_seed(CLUSTER_SEED)` before each
   `clustered_order` / `balanced_labels` call, then run the stage three times and report the treatment's
   own sd beside the null's. CPU-only, no capture, ~26 minutes per run, donor-free. **Until this exists,
   every co-activation number in this document is a single draw from a distribution of unmeasured
   width, and 43 of 210 margins are inside the width I measured.** (B3)
2. **Re-derive §11.7b's fourth column** as a real per-token byte count — mean over tokens of
   `(Σ sizes of loaded experts, post-duplication)/N` — or delete the column. It is a pure re-computation
   over data already on disk. (B2)
3. **One BPB pass on one carved configuration.** L27, `E = 32`, `k = 8` (`relerr` 0.2804, 25.1% of FFN
   bytes) is the obvious candidate. C1's harness already does everything needed. This is the only
   measurement that puts the MoE carve into the unit the programme decides in. (Claim 4)
4. **State `FFN_active ≥ (1+2p)/3`** and retire the second-donor recommendation, which cannot change the
   skippable verdict. (F4)
5. **Correct B1, F1, F2, F5** in the document, and note that commit `30efdd9`'s subject carries F1.
6. **Give C3 a fit/score split** and tighten its tolerance to relative. Minutes of CPU, and it converts
   a control that cannot fail into one that tests the pipeline actually used. (F7)
7. Record sklearn's version; the git stamp cannot substitute for it. (F10)

---

*Controller. No fixes implemented, nothing committed, nothing pushed. All verdicts reproduce from the
artefacts named at the head of this document.*
