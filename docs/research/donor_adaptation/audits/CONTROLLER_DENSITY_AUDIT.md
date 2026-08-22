# Controller audit — D0b (rho-floor at donor dimensions) and D1 (per-organ pruning)

**Auditor:** Controller (independent adversarial review). **Scope:** `d0b_rho_floor_donor.py` /
`results/d0b_rho_floor_donor.json` and `d1_pruning.py` + `d1b_organ_sweep_completion.py` /
`results/d1_pruning.json`, as reflected in `docs/research/DENSITY_PROBES.md` §2–3 and quoted
forward into `ADAPTER_MEMO_01_SPEED_BUDGET.md` / `DOSSIER_ADJUDICATION_JOINT_SOLVER.md`. No code
or write-up was edited. Two defects already found by the team (block-mask quantisation
contaminating `k_proj`/`v_proj`@90%; the `[SIGNIFICANT]` label bug on null-expected controls) are
not re-reported here except where they interact with a new finding.

Every finding below was checked against the actual files at git revision `e410243` on
`research/donor-adaptation` (working tree, uncommitted changes present per `git status`). Where a
number is asserted, the exact command or Python snippet used to derive it is given so it can be
re-run.

---

## FLAG — the "coarsening penalty falls with sparsity" trend hides a sign flip and a near-tie, not a smooth approach to 1×

**Claim in `DENSITY_PROBES.md` §3.3:** "the coarsening penalty Δ(block)/Δ(unstructured) FALLS
with sparsity ... 0.9× – 3.1×" at s=0.90, presented as the mild end of a monotonically shrinking
penalty.

**What I found.** Recomputing every ratio directly from `results/d1_pruning.json`
(`organ_sweep`, `mode ∈ {unstructured, block_structured}`, excluding the two quantisation-
contaminated points already struck by the team):

| s | ratio range (recomputed) | write-up range | notes |
|---|---|---|---|
| 0.25 | 17.42× – 826.24× | 17.4×–826× | matches |
| 0.50 | 7.88× – 134.58× | 7.9×–135× | matches |
| 0.75 | 2.44× – 10.30× | 2.4×–10.3× | matches |
| 0.90 | **0.89× – 3.08×** | 0.9×–3.1× | matches numerically, **but two of the seven organs are qualitatively different, not just smaller** |

At s=0.90: `q_proj` ratio = **1.00** (block Δ = 1.46233, unstructured Δ = 1.45721 — a statistical
tie, not "penalty shrinking"), and `gate_proj` ratio = **0.89** (block Δ = 3.54358 **less than**
unstructured Δ = 3.97663 — the sign has flipped: the coarser, position-constrained removal is
now *cheaper* than the free, per-weight removal at the same total sparsity). A "penalty" that goes
below 1 is not a penalty; the word choice in §3.3 papers over this.

**Why this isn't just sampling noise.** Individual paired-bootstrap SEs at these points are
0.05–0.15 (n_boot=2000, n=24 sequences); the raw deltas being compared straddle differences of
0.01–0.43 BPB, well inside what a handful of SEs would explain as a tie, but nowhere near
supporting the alternative reading ("block still costs meaningfully more, just less"). I did not
have a joint (block − unstructured) paired-bootstrap in the artefact to quote a single p-value for
the sign flip, but even a conservative combined SE (√(0.127² + 0.154²) ≈ 0.20 for gate_proj) puts
the gap (0.43) at ~2.2σ — not overwhelming, but consistent, not noise-explained away either.
**This is the one place I'd want the joint bootstrap actually computed before either the tie or
the flip is asserted as a headline number**, but the raw deltas are real (bit-reproducible, see
the PASS section below) and the direction is not an artefact of rounding.

**Root cause, verified independently (see next finding).** Both `q_proj` and `v_proj` show
**non-monotonic absolute BPB** in `structured` and `block_structured` mode: BPB gets *better*
(lower) going from 75%→90/91.7% zeroed, for the *same* organ, in a regime where `unstructured`
mode keeps rising monotonically. Concretely (delta BPB, `block_structured`):

```
q_proj:  25%→+0.513   50%→+1.715   75%→+1.900   90%(91.7%)→+1.462   <- falls
v_proj:  25%→+1.980   50%→+3.681   75%→+4.150   90%(100%,anchor≈+3.65)→+3.654 <- falls
gate_proj (vs its own unstructured): unstructured 75%→+1.470, 90%→+3.977 (rises);
    block_structured 75%→+3.589, 90%→+3.544 (flat/falls)
```
(reproduction: `python -c "import json; d=json.load(open('benchmarks/donor_adaptation/density/results/d1_pruning.json')); [print(r['organ'],r['mode'],r['level'],r['zero_frac'],r['delta']) for r in d['organ_sweep'] if r['organ'] in ('q_proj','v_proj','gate_proj')]"`)

**Is this a "degenerate-predictor ceiling"?** Partially ruled out. If BPB were saturating at a
generic floor (e.g. "the model has collapsed to guessing the unigram distribution"), I'd expect
*every* organ to plateau near the same absolute BPB at high sparsity. It doesn't: `o_proj`
block-structured keeps climbing past a plausible degenerate floor (delta +4.803 at 90% → absolute
BPB ≈ 5.57, which is **above** the uniform-over-vocab ceiling of ~4.07 bits/byte computed from
`log2(151936)/4.229 bytes-per-token` — a model can exceed that ceiling by being confidently
*wrong*, so this isn't impossible, but it is the opposite of "collapsed to a safe floor"). So the
reversal is organ-specific (`q_proj`, `v_proj`, weakly `gate_proj`), not a universal ceiling
artefact. A coherent mechanism that fits without new data: full or near-full removal of a small,
GQA-shrunk matrix (`v_proj` here has only `n_kv_heads·head_dim = 256` output rows at this donor)
produces a *clean, deterministic* zero contribution from that organ, which the residual/SSM
pathway can partially route around — whereas a *partial* removal (65–75%) leaves a noisy,
miscalibrated version of the same organ active, which is actively worse than silence. This is the
same "noise is worse than absence" asymmetry documented for the engine's own dReLU skip flag in
`ENGINE_SKIP_FLAG_FINDING.md`, applied here to weight removal instead of activation skip — but it
is an unverified hypothesis, not a measured mechanism; flagging it as such.

**Why it matters.** §3.4 uses the falling-penalty trend to argue the joint solver "has a great
deal of room to recover" at high sparsity. That argument is *stronger*, not weaker, once the
sign-flip is accounted for correctly — but the write-up currently doesn't mention the flip at all,
so a reader taking "0.9×–3.1×" at face value would conclude coarsening has become nearly free
uniformly across organs, when for 2 of 7 it has gone *past* free into "block is better than free
selection," which is the more interesting and more fragile claim and deserves its own line.

**Verdict: FLAG, not BLOCK.** The underlying deltas are real and reproducible (verified below).
The issue is entirely in how §3.3's summary phrase ("penalty falls with sparsity") describes them.
Concrete fix: report `q_proj` and `gate_proj` at s=0.90 as "tied" / "inverted" explicitly, not
folded into a range that reads as "smaller but still a penalty."

---

## FLAG — "structured" (free row/column selection) is empirically *worse* than "block_structured" (position-constrained) for attention organs, contradicting its own framing as an optimistic bound

**Claim, `d1b_organ_sweep_completion.py` docstring:** `structured` is "existing, free selection —
**an optimistic bound** on what reordering+compaction could achieve," `block_structured` is "new,
fixed contiguous grouping — what the engine's block-skip path can actually exploit **without any
reordering**." The framing asserts `Δ(structured) ≤ Δ(block_structured)` at matched sparsity,
because free selection strictly dominates a position-constrained subset of the same search space
(pick the same rows, or better).

**What the data show** (same JSON, `mode ∈ {structured, block_structured}`, q/k/v_proj — the only
organs `structured` was ever run for before the session was interrupted):

| organ | level | Δ structured | Δ block_structured | structured worse by |
|---|---|---|---|---|
| q_proj | 0.25 | 1.1638 | 0.5131 | **2.3×** |
| q_proj | 0.75 | 2.1206 | 1.9001 | 1.1× |
| k_proj | 0.25 | 0.7540 | 0.1324 | **5.7×** |
| k_proj | 0.50 | 1.5930 | 0.3549 | **4.5×** |
| v_proj | 0.25 | 2.9030 | 1.9796 | **1.5×** |
| v_proj | 0.50 | 4.1740 | 3.6812 | 1.1× |

(full table: `python -c "import json; d=json.load(open('...d1_pruning.json')); [print(r['organ'],r['mode'],r['level'],r['delta']) for r in d['organ_sweep'] if r['organ'] in ('q_proj','k_proj','v_proj') and r['mode'] in ('structured','block_structured')]"`)

In 8 of the 11 comparable (organ, level) pairs where both modes have data, the "optimistic bound"
is **worse** than the constrained alternative it's supposed to bound from below. This is not a
rounding-level effect — `k_proj` at 25% is 5.7× worse under free selection.

**A candidate mechanism, not verified here.** `block_structured`'s block size (64) happens to be
exactly half of this donor's `head_dim` (128 = 1536/12), so every contiguous block lies inside a
single attention head — block removal at any level cleanly zeroes whole half-heads and leaves
other heads fully intact. `structured` mode ranks *individual* rows by global L2 norm with no
respect for head boundaries, so at low-to-mid sparsity it tends to shave a few rows out of *many*
heads rather than fully disabling *some*. If partial, scattered damage to many heads is more
disruptive than complete damage to few (the same asymmetry as the previous finding, in a
different direction), that would explain the reversal — but I have not measured per-head row
distributions to confirm this, so it is offered as a hypothesis with a concrete follow-up
(`torch.argsort(norms)[:k] // head_dim`, histogram the head each removed row belongs to), not a
finding.

**Why it matters.** `DENSITY_PROBES.md` §3 never presents `structured`-mode results at all — only
`unstructured` vs `block_structured` are tabulated. The 29 `structured`-mode records that exist in
the artefact (q/k/v_proj, all 4 levels, from the interrupted first session) are silently absent
from the write-up's own reasoning, even though they falsify the docstring's stated ordering
between the three modes. Nothing downstream currently depends on `structured` being an upper or
lower bound, so this is not (yet) contaminating a quoted number — but the docstring's claim is
false as measured, sits unflagged in a file (`d1b_organ_sweep_completion.py`) that will be read
again, and should not be trusted as written.

**Verdict: FLAG.** Concrete fix: either drop the "optimistic bound" language or measure enough
`structured`-mode points (o_proj/gate_proj/up_proj/down_proj currently have zero) to state the
actual relationship instead of asserting it.

---

## FLAG — the down_proj "transposed storage" physical-contiguity caveat is stated once and silently doesn't cover o_proj, despite identical geometry

**Location:** `d0b_rho_floor_donor.py`, `verdict.interleave_physical_caveat` (also
`DENSITY_PROBES.md` §2.3): *"`down_proj`'s 'column' is contiguous only if `down_proj` is stored
**transposed** ... This requirement is scale-independent ... a pre-existing assumption of the
interleaved layout."*

**What I checked.** `analytic_donor_matrices()` assigns `axis="col"` to exactly two organs:
`o_proj` and `down_proj`. Both are `[out_features, in_features]` PyTorch tensors, both row-major
(each *row* contiguous in memory, standard `nn.Linear` storage), and for both, the pruned/skipped
unit under `axis="col"` is a **column** — i.e., a fixed index into `in_features`, striding across
`out_features` rows. A column of a row-major matrix is never memory-contiguous. The caveat
("needs transposed storage to be a real contiguous read") is not specific to `down_proj`'s
`d_ffn`-vs-`d_model` shape asymmetry at all — it follows purely from the `axis="col"` classifi-
cation, which `o_proj` shares exactly (`o_proj` is `d_model × d_model`, still needs `[in, out]`
storage for its "column" to be a physical run). The write-up states the caveat once, attached only
to `down_proj`, and D1's "engine-legal granularity" claim for `o_proj` block-structured pruning
(§3.3's table, `o_proj` row, +0.836 to +4.803 BPB, described as "the engine-legal `block_size=64`
... read off the D0 formula") inherits the same unstated requirement without ever naming it.

**Why it matters.** It doesn't invalidate any BPB number — those are measurements of what removing
a logical group of weights does to quality, and that's true regardless of physical memory layout.
It does mean the phrase "engine-legal" is doing work for `o_proj` that isn't backed by the same
diligence given to `down_proj` two sections earlier in the same document lineage. A reader who
takes "engine-legal" at face value for `o_proj` without re-deriving it would be missing a
precondition (transposed storage of `o_proj`) that the authors clearly know applies (they wrote
it for `down_proj`) but didn't propagate.

**Verdict: FLAG.** Cheap fix: generalize the existing caveat from "`down_proj`'s column" to "any
`axis="col"` organ's column (`o_proj`, `down_proj`)" — one sentence, no re-derivation needed since
the logic is already identical.

---

## FLAG — the "Llama-3-70B-class" donor dimensions are hardcoded, not read from a pinned artefact, in a script whose own stated law is "the artefact is the authority"

**Location:** `common.py`'s module docstring states the project law: *"the artefact is the
authority -> every dimension is read from config.json, never asserted."* `d0b_rho_floor_donor.py`
follows this rigorously for the Qwen2.5-1.5B baseline (`MODEL_ID`/`REVISION` pinned, dims read via
`arch(model)` off the loaded weights). It does **not** follow it for the donor comparison: `main()`
calls `analytic_donor_matrices(d_model=8192, d_ffn=28672, n_heads=64, n_kv_heads=8)` with these
four numbers typed as literal function arguments, labelled in comments and prints as
"Llama-3-70B-class." No `config.json` for any Llama-3-70B variant exists anywhere in this repo or
in the local HuggingFace cache (checked: `find . -iname "*llama-3-70b*"` and
`ls ~/.cache/huggingface/hub | grep -i llama` — only `huggyllama/llama-7b` is present, an unrelated
7B model). There is no pinned revision, no sha256, no artefact reference for these four numbers —
they are asserted from the author's memory of the architecture.

**On the numbers themselves.** They are, as far as I can independently confirm, the correct
published values for Meta's Llama-3-70B (`hidden_size=8192`, `intermediate_size=28672`,
`num_attention_heads=64`, `num_key_value_heads=8`) — I am not flagging a wrong number. I am
flagging that this script has no way to *prove* that from what's in this repository, in violation
of its own stated methodology, and the project's own standing rule
(`feedback_literature_fabrication.md`: a figure doesn't enter a decision until read in its own
table) exists precisely because "the number happens to be right" has been the failure mode before
in this project (`DOSSIER_ADJUDICATION_JOINT_SOLVER.md` §4b documents exactly this kind of
contamination in the dossier under adjudication — 9.4B misquoted as 100B, a citation `[10]`
fabricated).

**Verdict: FLAG**, not BLOCK, because the numbers check out against external knowledge and nothing
downstream is silently wrong — but the script should either download/pin an actual Llama-3-70B
`config.json` (no weights needed, `config.json` is tiny and license-free to fetch) or explicitly
label the 8192/28672/64/8 block as "ASSERTED, not artefact-pinned" the same way the codebase
labels `[A]`/`[X]` provenance elsewhere in this research stream.

---

## FLAG — provenance gap: 29 of 73 organ_sweep records (all of session 1) carry no git revision or command line

**Location:** `results/d1_pruning.json`. Top-level keys are
`['arch','slice','baseline_bpb','threads','levels','controls','organ_sweep','depth_sweep',
'session2']` — there is **no top-level `git_revision`**. Only `log["session2"]` (written by
`d1b_organ_sweep_completion.py`) carries `git_revision`/`git_branch`/`command_line`. The original
`d1_pruning.py` run (29 of the 73 `organ_sweep` records — all of `q_proj`/`k_proj`/`v_proj`'s
`unstructured` and `structured` rows, plus the empty `depth_sweep`) has no git-revision field
anywhere in the artefact. The seed and thread count *are* recoverable (via `slice.seed=1234` and
top-level `threads=10`), so this is a partial gap, not a total one — but "which commit produced
this number" is not answerable from the JSON alone for the older third of the sweep.

**Also noted, not separately flagged:** `depth_sweep` is present as a key but is an **empty
list** — the per-layer depth sensitivity experiment defined in `d1_pruning.py` (`DEPTHS = [0, 4,
9, 13, 18, 22, 27]`) never ran to completion in either session (session 1 was interrupted before
reaching it; `d1b` only resumes `organ_sweep`). `DENSITY_PROBES.md` §3 does not claim any
depth-sweep result, so this is not a false claim — just an orphaned, silently-abandoned part of
the artefact worth knowing about before anyone assumes depth data exists.

**Verdict: FLAG**, minor. Fix: `d1_pruning.py` should log `git_revision`/`git_branch` at the top
level the same way `d0b_rho_floor_donor.py` does; retroactively, the missing 29 records can likely
be dated from file mtimes / the surrounding commit history if it ever matters.

---

## PASS (independently reproduced by re-running, not just re-reading) — baseline BPB and one control are bit-exact across a fresh process

**What I did.** I did not just re-read the artefact; I re-ran the actual code in a new Python
process, on this machine, right now, reusing the pinned cached slice
(`results/slice_heldout_24x512_s1234.pt`) and the unmodified `common.py`/`d1_pruning.py`:

```
m, tok = C.load_model()
ids, byts, meta = C.get_slice(tok, "heldout", D1.N_SEQ, D1.SEQ_LEN, D1.SEED)
base, base_per = C.bpb(m, ids, byts, return_per_seq=True)
# then D1.prune_structured(m, all_layers, "o_proj", 0.05, snap); re-evaluate
```

**Result:**
```
REPRO baseline BPB      = 0.7675949677540373   (128s)
ARTEFACT baseline BPB   = 0.7675949677540373   -> BIT-EXACT MATCH: True
REPRO CTRL struct@5pct_o_proj delta = 4.0244205337274925   zero_frac=0.050132   (99s)
ARTEFACT (session2) same control     = 4.0244205337274925   zero_frac=0.050132
```

Both numbers match to all 16 printed significant figures, in a separate process, on CPU, with
`D_THREADS=10` (same as the original run). This is real evidence the harness is deterministic and
that the extraordinary size of `CTRL struct@5pct_o_proj` (see next item) is not a one-off fluke,
race condition, or thread-count-dependent nondeterminism — it reproduces exactly.

**Verdict: PASS.** Determinism claim in `d1_pruning.py`'s docstring ("the harness's own
determinism ... makes the deltas exact") holds up under an actual independent re-run, not just
inspection.

---

## FLAG — the "minimal significant corruption" control (`struct@5pct_o_proj`) is not minimal: it is nearly as catastrophic as fully zeroing v_proj, and this is never remarked on

**What the control is for, per its own comment:** *"minimal significant corruption: structured 5%
on ONE organ, one level below the grid. If the instrument cannot see the removal of the 5%
least-important o_proj columns ... nothing it reports about 25% means anything."* The implicit
expectation is a *small*, barely-detectable perturbation — "one level below the grid," i.e.
smaller than the smallest sweep point (25%).

**What it measures:** delta = **+4.024 BPB** (bit-reproduced above), zero_frac = 5.01%. For
comparison: `CTRL struct@100pct_v_proj` (fully zeroing an entire organ across all 28 layers) gives
delta = **+3.654 BPB** — *smaller* than the "minimal" 5% o_proj control. Removing 5% of `o_proj`'s
columns is measurably **more destructive than removing 100% of `v_proj`**. Nothing in `d1_pruning.py`
or the write-up remarks on this; the control is logged as simply "fired" (delta > 0, trivially
true) with no check on *magnitude* plausibility.

**Is this a bug or a real property of the donor?** I cannot rule out a real, if surprising,
mechanism (removing scattered `o_proj` input columns — which mix contributions from all 12 heads,
since `o_proj` is not head-block-diagonal — could break inter-head recombination in a way that
full `v_proj` zeroing, which cleanly routes around dead attention, does not). But this is
precisely the kind of surprising result that the mandate asks to be attacked, not accepted at
face value:

- I confirmed `zero_frac` is correctly ≈5.0%, not silently something else (ruling out the
  known mask-quantisation bug class).
- `o_proj`'s `structured`-mode sweep (25/50/75/90%) was **never run** (see the "optimistic bound"
  finding above — `structured` is empty for every organ except q/k/v) — so there is no dose-
  response curve to sanity-check +4.02 at 5% against +X at 25%. The nearest comparable number is
  `o_proj`/`block_structured`/25% = +0.836 — 4.8× *smaller* damage from *5×* more sparsity, in a
  different (block-constrained) mode. That is a real, internally-inconsistent-looking gap that a
  `structured` 25%/50% run for `o_proj` would resolve in under 10 minutes of CPU time and should
  be run before this control is relied on further.

**Verdict: FLAG**, not BLOCK — the number reproduces bit-exactly and isn't a code bug I could find
(organ name, axis, and zero_frac are all correct), but it is not "minimal" as claimed, its
magnitude is currently unexplained, and the missing adjacent sweep points mean nobody has actually
checked whether it fits a curve or is itself an outlier.

---

## What I tried to break and could not, and why it held

1. **Is the i.i.d. `block_skippable_fraction` claim (D0b §2.2, the "unstructured sparsity is not
   obviously worthless at donor scale" reopening) just `q^B` arithmetic dressed up as a
   measurement?** Yes, mechanically — I confirmed `0.9^12 = 0.28237...` and `0.9^4 = 0.65610...`
   match the simulated `0.2824`/`0.6563` to 3–4 significant figures, i.e. the Monte Carlo estimate
   (T=256 tokens × N=28672 neurons, ~2400–8500 blocks/token) is just converging to the closed
   form. **But this is not a violation** — both the probe script's own comments
   (`"isolates the effect of block size shrinking, holding the activation pattern fixed"`) and
   `DOSSIER_ADJUDICATION_JOINT_SOLVER.md` (`"Flagged honestly by the probe as a synthetic i.i.d.
   mask, not a real donor co-activation pattern"`) state exactly this, prominently, before the
   conclusion is drawn. I looked for the conclusion being smuggled in as if it were empirical
   donor-scale evidence; it isn't — the claim being defended ("this loosens, not tightens, the
   constraint vs. the D=1536 assumption") is a correct, disclosed piece of arithmetic, not a
   measurement claiming more than it is. **PASS.**

2. **Does D0b's ANALYTIC/MEASURED separation actually leak?** I read every number in
   `d0b_rho_floor_donor.json` against its generating function. Section (a) numbers
   (`analytic_row_D`, `analytic_donor_matrices`) touch no data, no model, no RNG — pure arithmetic
   on Python floats. Section (b) numbers (`measured_controls_at_donor_scale`) use `np.random`
   with a fixed seed against `d0_layout.run_lengths`/`layout_stats`, imported unmodified. I did
   not find a number presented as "measured" that was actually just re-typed analytic arithmetic,
   or vice versa. **PASS.**

3. **Would the planted controls catch a wrong-axis or wrong-organ substitution?** I could not find
   an executed instance of this in the artefact — but by construction, I could not make any of the
   four D1 controls (`prune@0pct`, `struct@5pct_o_proj`, `struct@100pct_v_proj`, `restore_exact`)
   *fail to fire* under a plausible corruption: swap `"o_proj"` for `"q_proj"` in the 5% control's
   organ argument, and `delta > 0` would almost certainly still hold (every single-organ removal
   at ≥5% structured produces positive delta somewhere in this dataset) — the control would report
   `[FIRED]` while silently testing the wrong organ. **This is not a defect I found in the executed
   run** (I verified the actual argument passed is the literal string `"o_proj"`, correctly) — it
   is a structural weakness in what "fired" can prove: the controls test "does *some* damage
   register," not "did the *requested* organ/axis/fraction register." A stronger control would
   assert an organ-specific expected *magnitude range*, or cross-check that two different organs'
   5%-controls produce measurably different deltas (which they would — o_proj's is >4, a q_proj
   5% would almost certainly be far smaller per the sweep) as a specificity check. Noted as a
   design gap, not something currently wrong.

4. **Reproducibility, actually executed, not just asserted.** See the PASS section above — I
   re-ran baseline + one control from scratch in a new process and got bit-exact agreement. I did
   not re-run the full 73-record sweep (CPU cost: ~100s/eval × ~150 evals ≈ 4+ hours), so I cannot
   claim every one of the 73 numbers reproduces — only that the harness's core determinism claim
   does, on the one baseline and one control I actually ran twice.

5. **`k_proj`/`v_proj` quantisation-to-100% at the "90%" label — already found by the team.** I
   independently re-derived it (`n_blocks_per_layer` for k/v_proj at D=1536: `1536/8/64=3` rows/
   head × ... = 4 blocks, `round(0.9·4)=4=100%`) and confirm `zero_frac` in the JSON matches
   `1.0` exactly for both, and that `v_proj/block_structured/90%`'s BPB
   (`4.421361169900009`) is bit-identical to `CTRL struct@100pct_v_proj`'s BPB. Correctly struck.
   **PASS on the fix**, not a new finding.

---

## Which headline claims survive, and at what confidence

| claim | verdict | confidence |
|---|---|---|
| D0b: transplanted `B_block≥22` is wrong at donor scale by 5.33×; correct floor at `d_model=8192` is 4 neurons | **survives** | high — independently re-derived, `32768/D` checked at both ends, control `reproduces_d0_layout_json_at_D1536` re-verified by hand |
| D0b: 3-organ interleave survives donor shapes because `d_ffn` cancels out | **survives** | high, for the FFN interleave as stated; the parallel `o_proj` caveat (this report's 3rd finding) is a related but separate gap in D1, not a flaw in D0b's own claim |
| D0b: i.i.d. block-skippable numbers reopen "unstructured sparsity is not obviously worthless at donor scale" | **survives, correctly labeled as algebra** | high on the arithmetic; this is a reframing of a known identity, not new empirical information — already disclosed as such |
| D1: block-structured pruning is catastrophic (σ_seed=0.005) at every organ and every level tested | **survives** | high — every one of the 71 clean (non-contaminated) points is tens to hundreds of σ from zero; verified independently on 2 points via re-run |
| D1: coarsening penalty falls with sparsity, 17–826× down to 0.9–3.1× | **survives numerically, needs revision in wording** | medium — the range is arithmetically correct (independently recomputed) but "falls" undersells that 2 of 7 organs cross into a tie/inversion at s=0.90, driven by a non-monotonic collapse this report surfaces and the write-up doesn't discuss |
| D1: dossier Table 1's rising-with-sparsity trend is wrong in both magnitude and direction | **survives** | high — unaffected by the findings above, which only complicate the *upper* end of D1's own curve, not its comparison to Table 1 |
| D1's numbers upper-bound the donor-scale cost (finer engine-legal block at donor width should be milder) | **untested, correctly labeled as untested** | the authors flag this themselves as an open question; no donor-scale BPB measurement exists to confirm or deny it, and this audit did not attempt one |
| "structured" mode is an optimistic bound on "block_structured" | **does not survive as stated** | low — falsified on 8 of 11 measurable (organ, level) pairs for q/k/v_proj; needs either retraction of the ordering claim or more `structured`-mode data |
| Llama-3-70B-class donor dims (8192/28672/64/8) | **numerically correct, provenance unverified in-repo** | the numbers check out against external knowledge but are not artefact-pinned as the script's own methodology requires |

No headline claim required outright withdrawal. Two require re-wording (`structured` = optimistic
bound; "penalty falls with sparsity" at s=0.90). Two are FLAGged for provenance/rigor rather than
correctness (Llama-3-70B dims; missing git revision on 29 records). The `struct@5pct_o_proj`
control's unexplained magnitude is the item I'd most want resolved before anyone builds on it —
it is currently an unreconciled fact sitting inside a "control," not a sweep point, so it gets
less scrutiny than it would if it were labelled as data.

---
---

# 2026-08-22 — Controller audit — D4 (Hessian-weighted layer-wise reconstruction), pre-belief

**Auditor:** Controller (independent adversarial review). **Scope:** `d4_reconstruction.py`
(read at the version on disk at audit time; `git_revision` in the live run's own log =
`e410243`, working tree has further uncommitted edits per `git status`), its pre-registration
block (`log["prereg"]`, written in `main()` before any number is computed), the rank diagnostics
in `results/d4_reconstruction.json`, and the preserved failure record in
`results/d4_reconstruction.ABORTED_session1.json`. **No code was edited, no heavy run was
started.** A D4 probe was live-running for the entire audit (60/80 GB resident); every finding
below either reproduces on tiny synthetic matrices (numpy, a few MB, `<1s` each) or reads the
live run's own append-only log (`d4_run.log`) and JSON checkpoint without modifying either. The
live run had reached `ABLATION real_H down_proj@50%` (control 3, real-H arm only) by the time
this audit closed; `identity_H`/`shuffled_H` arms of control 3, control 4, and the entire main
sweep were **not yet computed** — findings that need those numbers are marked as such, with an
exact reproduction/verification command for after the run finishes.

All synthetic-matrix reproductions live in
`C:\Users\giosa\AppData\Local\Temp\claude\...\scratchpad\probe{1..5}_*.py` (session-local scratch,
not part of the repo) — commands to regenerate each number are given inline below; they are cheap
enough (`<1s`, `<50MB`) to paste into any Python REPL.

---

## BLOCK — the mandatory rank(H)/N pre-solve safety gate cannot fire in this run's own regime; it is not a weakened version of the diagnostic that caught the first failure, it is a diagnostic that has been made structurally unable to detect a second occurrence of the SAME failure mode

**Claim.** `check_hessian_rank_by_layer` (`d4_reconstruction.py:236-273`, calling
`hessian_rank_ratio` at `:202-216`) is supposed to be Failure-Mode-1 protection — the exact
mechanism the Principal ordered added after the aborted session 1 (`ABORTED_session1.json`:
`rank(H)/N = 4096/8960 = 0.457 < 0.5` was root cause #2 of the +0.349 BPB / 0.6545-weight-deviation
abort). But `rank(H) ≤ min(T, D)`, and this run always has `T = 16384 > D` for every organ tested
(`down_proj` D=8960, `o_proj` D=1536) — so the diagnostic's own upper bound is `min(T,D)/D = 1.0`
**before a single eigenvalue is computed.** The live run's own log confirms this is not a
theoretical worry but exactly what happened: `rank(H)/N [down_proj_real_H, method=exact]: min=1.000
mean=1.000 max=1.000` and identically for `o_proj_real_H` (`d4_run.log`, `results/d4_reconstruction.json
["rank_diagnostics"]`). The threshold check (`summary["ratio_min"] < threshold` at `:267`) can
therefore never trigger for either organ in this configuration, regardless of how poorly
conditioned the *interior* eigenvalue spectrum actually is — full rank and "well-conditioned
enough for a Tikhonov solve with λ ~ 1e-4·trace/N to be numerically meaningful" are different
properties, and the code only ever asks the first question.

**This is not a hypothetical gap — the project's own prior adjudication already said so.**
`DOSSIER_ADJUDICATION_JOINT_SOLVER.md:193` (a document produced *in this repo, before D4 was
written*): *"§6.8 reassures us that `B_tokens = 4096 ≥ N = 4096` makes rank deficiency rare — **at
exact equality, with a generically atrocious condition number.**"* — i.e. the exact failure mode
of "technically full rank, still numerically dangerous" was named and warned about by name before
this diagnostic was implemented, and the implementation only carries forward the `rank(H)/N < 0.5`
half of that document's own §7.1 (`DOSSIER_ADJUDICATION_JOINT_SOLVER.md:250`), not a conditioning
check.

**Concrete demonstration that "full rank" and "safe to solve" are different claims, at exactly
D4's own T/D margin.** Reproduced numerically (`probe5_lambda_tight_ratio.py`, D=4000/T=7314,
T/D=1.8285 — matched to D4's real `down_proj` ratio `16384/8960=1.8286` to 4 significant figures,
scaled down only for CPU cost):

```
D=4000 T=7314 T/D=1.8285  (D4 real: T=16384 D=8960 T/D=1.8286)
eig(H): min=4.637e+01 max=2.431e+06 cond=5.243e+04
rank(H)/D (numpy tol) = 4000/4000        <- reads 1.000, exactly what the live run reports
```

A condition number of `5.2×10^4` with genuinely full numerical rank — the diagnostic reports
"1.000, PASS" on data whose smallest eigendirection is nearly five orders of magnitude weaker than
its largest, and would report the identical "1.000, PASS" on data ten or a hundred times worse
conditioned still, because the check never looks past the rank/full-rank boundary. I could not
determine from the artefact whether the *actual* `down_proj`/`o_proj` real-H spectra at 128K-donor
scale are this badly conditioned or better — **that is precisely the point: the instrument that
was supposed to tell you cannot tell you, in either direction.**

**What was asked for and not built.** The brief (and, by inheritance, the Controller's own
standing request after the joint-solver dossier) asked for "condition number, λ_max, λ_min, count
of eigendirections below the Tikhonov λ." None of these are computed or logged anywhere in
`d4_reconstruction.py`. `hessian_rank_ratio` (`:202-216`) computes `eigvalsh` (so the eigenvalues
*exist in memory, transiently*, at `:213`) but immediately discards everything except a boolean
count against a threshold `tol = eigvals.max() * D * eps` (`:214`) — a threshold roughly
`eigvals.max() * 2×10⁻¹²` for `D=8960`, which is **8-11 orders of magnitude smaller** than the
actual Tikhonov `λ = 1e-4·trace(H)/N` used downstream in every solve (`reconstruct_organ_level`,
`:289`; concretely order `10⁻²`-`10^0` at realistic activation scales per the λ-sensitivity probe
below). The rank check and the regularisation strength are answering two unrelated questions at
two unrelated scales, and the code never connects them.

**Failure scenario this misses.** If `down_proj`'s real activation covariance at 128K-donor scale
turns out to have (say) a condition number of `10^8`-`10^10` — plausible for a wide, sparsely-
activated SwiGLU intermediate calibrated on only 32 sequences — the Cholesky solve would still
"succeed" numerically (λ=trace/N-scaled regularisation masks the ill-conditioning rather than
exposing it), rank(H)/N would still read exactly 1.000, and **nothing in the current instrument
would ever flag it.** The resulting `recovery` numbers would be real floating-point outputs, not
crashes or NaNs — silently regularisation-dominated rather than data-dominated, and indistinguishable
from a well-conditioned success in the artefact's own diagnostics.

**Reproduction.**
```python
# probe5_lambda_tight_ratio.py logic, condensed:
import numpy as np
D, T = 4000, int(round(4000*16384/8960))
X = ...  # any generative process with a shared low-rank factor (see probe file)
H = X.T @ X
print(np.linalg.matrix_rank(H), D)              # -> 4000 4000  (reads "full rank")
ev = np.linalg.eigvalsh(H)
print(ev.max()/ev.min())                         # -> ~5.2e4, NOT reported by d4_reconstruction.py
```
To check the REAL donor spectrum once the machine is free (do not run now):
```
python -c "
import sys; sys.path.insert(0,'benchmarks/donor_adaptation/density')
import common as C, torch
m, tok = C.load_model()
# reuse d4's own capture_organ_inputs_multi on the pinned calib slice, form H for down_proj L0,
# report eigvalsh min/max/cond directly instead of just the >0.5 rank fraction
"
```

**Verdict: BLOCK.** This is a message to the Principal, not just a line item: the specific
protection ordered after session 1's abort has been reinstalled in a form that is mathematically
guaranteed to read "safe" for every point in this sweep, independent of whether the underlying
data is actually safe. Whatever the real conditioning turns out to be, the artefact currently
provides no evidence either way, and reads as if it does.

---

## BLOCK — `shuffle_columns`'s "off-diagonal → ~0" claim is false whenever the underlying activations have nonzero column means (i.e. essentially certain for a SwiGLU/gated MLP intermediate), so control 3's `shuffled_H` arm is not a clean "H carries no usable information" null — it can show real, non-trivial `recovery` on its own, and nothing downstream corrects for this

**Claim in the code's own docstring** (`d4_reconstruction.py:155-160`): shuffling each column's
token order independently *"[destroys] cross-feature correlation (H_shuf's off-diagonal -> ~0).
This is the 'fake but same-scale' Hessian: if it recovers as much as the real one, the recovery is
not coming from correlation structure."* The reasoning only holds if `X`'s columns are mean-zero.
`shuffle_columns` permutes the *token* axis per feature independently, so it exactly preserves each
column's own values (hence its mean, variance, and full marginal) while permuting which *token*
each value is attached to. For two columns `i,j` with nonzero means `μ_i, μ_j`, the shuffled
cross-term `H_shuf[i,j] = Σ_t X_shuf[t,i]·X_shuf[t,j]` retains an expected `T·μ_i·μ_j` component
(a random-pairing sum of two independently-permuted sequences does not average to zero when both
have nonzero mean) — only the *centered* covariance is destroyed, not the raw second moment. Down
`_proj`'s input is `silu(gate)·up`, a SwiGLU intermediate — this project's own prior probes
(`project_probe2_sparsity`, `project_probe4_moe` in memory) already establish this class of
activation is heavily skewed/sparse (dReLU ≈90-92% near-zero), i.e. exactly the "large mean
relative to spread, mostly non-negative" shape where this failure mode bites hardest.

**Numerically demonstrated** (`probe3_shuffle_control.py`, T=16384 to match D4 exactly, D=500,
synthetic SwiGLU-shaped X = `silu(context@loadings + noise) * |up-noise|`, genuinely correlated via
a rank-6 shared context factor — i.e. real, exploitable cross-feature structure is present, same
as the real down_proj input is presumed to have):

```
X stats: mean=0.2493  frac near-zero(<1e-3)=0.24%  min=-1.180 max=19.425
diagonal exactly preserved by shuffle: True (max diag diff = 2.5e-11)
off-diagonal Frobenius norm: real=1.151e+06  shuffled=5.984e+05
  ratio shuffled/real = 0.520          <- NOT "~0"; the shuffle leaves 52% of the off-diagonal mass intact
real_H      : recovery=+0.680
shuffled_H  : recovery=+0.148          <- NOT "~0"; a "carries-no-cross-feature-correlation" H recovers 15pp
identity_H  : recovery=+0.000          <- this one IS exactly 0, but not empirically -- see next finding
GAP (real - shuffled) recovery = +0.531
```

**Consequence for the actual metric being reported.** `recovery = 1 - delta_recon/delta_naive` is
computed against `real_H` alone, for *every* point in the main sweep (`d4_reconstruction.py:709`)
— there is no per-sweep-point subtraction of a shuffled-H baseline anywhere in the code, and
`summarize.py`/`tables.md` currently have **no D4 section at all** (`grep -n "d4\|D4"` on both
returns nothing), so there is no later stage where this correction could be applied either. The
*only* place `shuffled_H` is even computed is the single `down_proj@50%` ablation point
(`d4_reconstruction.py:596-607`) — it is never run for `o_proj` or for any other sparsity level,
and even there it is reported as a bare `recovery_point` alongside `real_H`'s, with no subtraction
performed (`:618-620`). If the real run's `down_proj@50%` shuffled arm shows anything like the
15-22% (of the real-H recovery) leakage this synthetic shows, then the headline claim "Hessian-
weighted reconstruction recovers X% of the block-pruning loss" is, for an unknown fraction of X,
actually "reconstruction that merely re-scales by each surviving feature's own average magnitude
recovers a chunk of X, and true cross-feature correlation recovers the rest" — a materially weaker
and different claim, and the artefact as currently structured cannot distinguish the two even
where it has the data to (the one ablation point) — it just prints three numbers side by side.

**Reproduction / verification once the live run's control 3 finishes** (do not run now; read the
artefact after):
```python
import json
d = json.load(open("benchmarks/donor_adaptation/density/results/d4_reconstruction.json"))
ab = d["controls"]["hessian_ablation"]
print("real_H recovery:", ab["real_H"]["recovery_point"])
print("shuffled_H recovery:", ab["shuffled_H"]["recovery_point"])
print("identity_H recovery:", ab["identity_H"]["recovery_point"])
print("off-diagonal leakage estimate (genuine-correlation-only recovery):",
      ab["real_H"]["recovery_point"] - ab["shuffled_H"]["recovery_point"])
```
If `shuffled_H`'s `recovery_point` prints anywhere near `real_H`'s (not ~0), that is this finding,
confirmed on the real donor Hessian rather than a synthetic proxy.

**Verdict: BLOCK.** This is the message-to-the-Principal item the brief specifically asked to hunt
for ("construct the minimal case where a useless H produces a real-vs-shuffled gap anyway") — I
found it, on a generative model chosen to resemble the actual organ under test, not an adversarial
edge case. `real_H − shuffled_H`, not raw `real_H` recovery, is the quantity that isolates
genuine-correlation value; it is currently computed nowhere in the pipeline for anything but one
(organ, level) point, and even there it is not surfaced as the headline number.

---

## FLAG — `identity_H` is not an empirical "no-information" baseline; it is algebraically forced to reproduce naive masking exactly, for any mask, independent of data — reported and read as if it were a measurement

**Claim.** For `H = I_D` (`d4_reconstruction.py:582`, control 3's second ablation arm),
`H_reg = (1+λ)I`. For *any* surviving set `S` (not just `S`=full): `A = H_reg[S,S] = (1+λ)I_{|S|}`,
`rhs = W·H_reg[:,S] = (1+λ)·W[:,S]`, so `sol = A⁻¹·rhsᵀ = W[:,S]ᵀ` exactly — the reconstruction
returns the surviving columns of `W` completely unchanged, i.e. **bit-for-bit the same operation as
naive masking**, for every sparsity level, regardless of what `W` or the calibration data are.
This is a closed-form algebraic identity, not something that could come out any other way — the
same category of "forced, not measured" result the prereg already (correctly) flags for the
ROW-structured organs (`:373-381`), but this one is not flagged anywhere in the code or prereg.

**Verified numerically** (`probe3_shuffle_control.py` above): `identity_H: recovery=+0.000` to 3
decimals, reproduced on a completely different generative model in `probe4_lambda_sensitivity.py`
too (not shown, same result). This is not a coincidence of the synthetic data; it will read
(within float64 rounding, order `1e-10`–`1e-14`) as **exactly** 0 on the real donor `down_proj`
run too — a testable, sharp prediction: `d["controls"]["hessian_ablation"]["identity_H"]
["recovery_point"]` should print a number indistinguishable from 0.000 once the live run reaches
that arm; if it does not, that is itself a bug (not evidence about the mechanism).

**Confirmed live, mid-audit, on the real donor weights — the prediction landed before this report
did.** The live run advanced past `identity_H` while this section was being written (read-only,
`d4_run.log`, not disturbed): `ABLATION identity_H down_proj@50%  BPB 2.32471  d=+1.55712
+-0.05797`. D1's own recorded `down_proj/block_structured/50%` naive delta is
`1.5571150213992855`. **`1.55712` vs `1.5571150214` — matches to 5-6 significant figures, i.e.
`identity_H`'s reconstruction reproduced D1's naive-masking number almost exactly, on the real
128K-donor model, not just the synthetic proxy.** This is about as clean a real-data confirmation
of a derived-not-measured algebraic fact as this kind of audit gets.

**Why this matters.** The prereg frames control 3 as three informative arms (`real_H`,
`identity_H`, `shuffled_H`) whose spread is supposed to establish whether Hessian information
matters. Only two of the three arms are actually capable of returning anything other than a
foregone conclusion: `identity_H` is guaranteed ≈0 by algebra (this finding), and `shuffled_H` is
*not* guaranteed ≈0 despite intent (previous finding). The entire empirical burden of control 3
rests on the `real_H` vs. `shuffled_H` gap alone, not the three-way comparison the prereg's framing
suggests — worth stating explicitly since "shuffled ≈ identity ≈ 0, real high" is the intuitive
success pattern the write-up will reach for, and one of those two "≈0" readings is not actually
informative even when it is observed.

**Verdict: FLAG**, not BLOCK — the number itself is correct and harmless (0 is genuinely the right
answer for "no information"), but its status (tautology vs. measurement) is undocumented, and a
reader who doesn't independently re-derive the algebra (as I just did) would credit it as
empirical confirmation that the harness is working, when it is actually confirmation of nothing
data-dependent at all.

---

## FLAG — the paired-bootstrap SE on `recovery` holds `delta_naive` fixed at its D1 point estimate; the ignored uncertainty is largest exactly at the `recovery ≈ 0` boundary the INCONCLUSIVE/NEGLIGIBLE classification most depends on

**Claim.** `bootstrap_recovery` (`d4_reconstruction.py:316-332`) resamples sequences for
`delta_reconstructed` only; `delta_naive` is read as a constant from D1's own point estimate
(`dn` in the main-sweep loop, `:679-683`). This is disclosed in the prereg (`:360-364`, "D1's own
paired_se on delta_naive is reported alongside per point, not re-bootstrapped jointly —
assumption stated here, not hidden") — so this is not an undisclosed problem, but its *magnitude*
was never quantified, and the brief specifically asks for that quantification (citing this
project's own history of an under-covering `±2·SE` when `t=4.30` was required at `n=3`).

**Propagation, done here.** Treating `delta_naive` and `delta_recon` as independent (a
conservative-in-one-direction approximation — see caveat below) and linearising
`recovery = 1 - delta_recon/delta_naive`:

```
Var(recovery) ≈ Var(delta_recon)/delta_naive²  +  (1-recovery)² · Var(delta_naive)/delta_naive²
                \_________________________/        \_______________________________________/
                 = D4's reported rec_se²             ignored entirely by bootstrap_recovery
```

So `SE_true/SE_reported = sqrt(1 + (1-recovery)² · (dn_se/recon_se)²)`. This ratio is **1 at
`recovery=1` (full recovery — the ignored term vanishes) and grows as `recovery→0`** — i.e. the
correction is smallest exactly where the point estimate is least interesting and largest exactly
at the boundary the pre-registered classification (`:335-343`, INCONCLUSIVE iff the `±2σ` band
spans 0) most depends on getting right.

**Using the one real number available from the live run** (`down_proj@50%` ablation, real_H arm:
`delta_recon=0.80546 ± 0.06898`; D1's `down_proj/block_structured/50%`: `delta_naive=1.5571150214
± 0.0579684340`): `recovery ≈ 0.4827`, reported `rec_se ≈ recon_se/|delta_naive| = 0.0443`, full
(independence-assumption) `SE ≈ sqrt(0.06898² + (0.5173·0.05797)²)/1.5571 ≈ 0.0483` — a **~9%**
understatement at this particular (moderate-recovery) point. This is mild here specifically
*because* `recovery` is comfortably away from 0 and `dn_se` is small relative to `recon_se` at this
point; per the formula above, a sweep point that lands near `recovery≈0` (plausible for `gate_proj`,
which the prereg itself predicts should be ≈0 by construction — see below) would see the ratio
approach `sqrt(1+(dn_se/recon_se)²)`, materially larger.

**Direction-of-bias caveat, stated honestly.** Treating the two deltas as independent may itself
be conservative in the *other* direction: both are paired-bootstrap SEs over resampling the *same*
24-sequence eval slice (D1's slice, reused bit-identically by D4, `:412`), and harder/easier
sequences plausibly move both a naively-masked and a reconstructed model's loss in the same
direction — positive correlation between the two deltas would *shrink* `Var(delta_recon -
delta_naive·recovery-term)` relative to the independence assumption used above, partially
offsetting the ignored-`dn_se` term. I did not have D1's and D4's raw per-sequence bootstrap draws
in the same process to compute the true joint covariance (D1's `paired_se` and D4's
`bootstrap_recovery` use the same RNG algorithm and could in principle be run with a shared seed
against the same `per`/`base_per` sequence order to get this exactly) — flagging this as the
correct next step rather than asserting a number I have not derived.

**Reproduction.**
```python
import json, math
d1 = json.load(open("benchmarks/donor_adaptation/density/results/d1_pruning.json"))
d4 = json.load(open("benchmarks/donor_adaptation/density/results/d4_reconstruction.json"))
for rec in d4.get("organ_sweep", []):
    dn, dn_se = rec["delta_naive_d1"], rec["delta_naive_d1_paired_se"]
    recon_se, recovery = rec["recovery_se"] * abs(dn), rec["recovery"]  # invert to get raw delta_recon SE
    full_se = math.sqrt(recon_se**2 + ((1-recovery)*dn_se)**2) / abs(dn)
    print(rec["organ"], rec["level"], f"reported_se={rec['recovery_se']:.4f} full_se={full_se:.4f} "
          f"ratio={full_se/rec['recovery_se']:.3f}")
```

**Verdict: FLAG.** Disclosed assumption, quantified impact is modest-to-moderate (≤~10% at the one
point currently measurable, larger near recovery≈0 by the formula's own shape), and there is a
concrete, cheap fix (shared-seed joint bootstrap) that was not applied. Not a BLOCK because the
assumption is stated, not hidden, and the current sweep's classification bands (`NOISE_BAND_SIGMA=
2.0` against `n=24` sequences, not `n=3` — the historical failure this project already learned
from) are not as fragile as that prior incident.

---

## FLAG — the mask D4 uses is *recomputed*, not reused from D1's artefact; the code is byte-for-byte the same selection algorithm, but there is no runtime assertion that the two actually agree, and the quantisation-fragile levels are silently outside the tested grid

**Claim.** `select_zero_blocks` (`d4_reconstruction.py:79-101`) is a literal copy of
`d1b_organ_sweep_completion.py`'s `prune_block_structured` selection math (`:63-95`) — same
`W.view(...).norm(dim=(1,2))`, same `k = round(frac*nblocks)`, same `torch.argsort(...)[:k]` — but
it is a *second, independent computation*, not a load of D1's stored indices (D1 never persisted
the actual selected block indices to JSON, only the resulting `zero_frac`). As long as (a) the
donor weights are bit-identical between the D1 and D4 processes and (b) `torch.argsort`'s
tie-breaking is deterministic given identical float inputs (true for this PyTorch build absent
literal float ties, which are measure-zero for real-valued norms), the two masks will match — I
verified the *algorithm* is identical by direct source comparison, but this was never verified at
*runtime* by the harness itself: no assertion anywhere compares D4's computed `zero_frac_achieved`
against D1's own recorded `zero_frac` for the matching `(organ, level)` point before `delta_naive`
is used as a denominator.

**Checked and clean for the levels actually swept.** `LEVELS = [0.25, 0.50, 0.75]`
(`d4_reconstruction.py:53`) against `n_blocks_per_layer` from D1's own artefact
(`gate_proj`/`down_proj`: 140, `o_proj`: 24 — `results/d1_pruning.json`) all divide exactly
(`0.25·140=35`, `0.5·140=70`, `0.75·140=105`; `0.25·24=6`, `0.5·24=12`, `0.75·24=18`) — no
`round()`-induced quantisation divergence is possible at these three levels for these three organs
(unlike the already-struck `k_proj`/`v_proj`@90% case, `D0`/`D1` history, not re-litigated here).
So this specific run is very unlikely to be silently comparing mismatched masks — **but the
invariant that makes that true is un-asserted, un-tested, and would silently break** the moment
someone adds a level or organ where the division is not exact.

**Reproduction (the missing assertion, written but not run against the live process's
memory — safe to add for a *future* run, not this one):**
```python
# after computing mask in the main sweep loop, before using dn:
d1_rec = next(r for r in d1["organ_sweep"]
              if r["organ"]==organ and r["mode"]=="block_structured" and abs(r["level"]-level)<1e-9)
assert abs(zero_frac_achieved - d1_rec["zero_frac"]) < 1e-6, \
    f"D4 mask diverges from D1's recorded mask at {organ}@{level}: {zero_frac_achieved} vs {d1_rec['zero_frac']}"
```
Post-hoc verification once the artefact is complete (read-only, safe to run any time):
```python
import json
d1 = {(r["organ"], r["level"]): r["zero_frac"] for r in json.load(open("results/d1_pruning.json"))["organ_sweep"] if r["mode"]=="block_structured"}
d4 = json.load(open("results/d4_reconstruction.json"))
for r in d4.get("organ_sweep", []):
    key = (r["organ"], r["level"])
    print(key, "D4:", r["zero_frac_achieved"], "D1:", d1.get(key), "match:", abs(r["zero_frac_achieved"]-d1.get(key,-1))<1e-6)
```

**Verdict: FLAG.** No evidence of an actual mismatch (the divisibility check above rules out the
one mechanism — quantisation rounding — that has bitten this exact codebase before), but the
absence of a runtime cross-check means this correctness property is currently an assumption held
by the auditor, not a guarantee held by the harness.

---

## FLAG — `LAMBDA_SCALE = 1e-4` is asserted, not justified; sensitivity is small at 2-3 orders of magnitude below the chosen value but grows to a ~11 percentage-point swing in `recovery` as λ_scale rises through the range a less careful choice could plausibly have landed on

**Claim.** `LAMBDA_SCALE = 1e-4` (`d4_reconstruction.py:54`) is a bare constant with no
sensitivity analysis anywhere in the artefact. Swept synthetically at D4's *actual* T/D margin
(`probe5_lambda_tight_ratio.py`, D=4000/T=7314 matching `16384/8960` to 4 s.f., 50%-column mask,
same SwiGLU-shaped generative model as the shuffle probe):

```
lam_scale=1e-6..1e-4:  recovery flat at 0.605           (D4's own choice sits deep in this plateau)
lam_scale=1e-3:        recovery 0.607   (+0.2pp)
lam_scale=1e-2:        recovery 0.616   (+1.1pp)
lam_scale=1e-1:        recovery 0.664   (+5.9pp)
lam_scale=1e0:         recovery 0.716   (+11.1pp)
lam_scale=3e0:         recovery 0.722   (peak, +11.7pp)
```

At a *looser* T/D margin (D=500, T=16384, `probe4_lambda_sensitivity.py`, T/D≈33×) the same sweep
is completely flat across all 8 orders of magnitude tested — so the sensitivity is specifically a
symptom of running near the T/D≈1.83× margin `down_proj`/`o_proj` are actually run at (consistent
with the conditioning concern in the first BLOCK finding above: a tighter margin means more
regularisation-sensitive directions in the spectrum). **Direction of the risk:** more `λ` only ever
*increased* recovery in both sweeps tested here (never decreased it) — meaning `1e-4` sits at the
conservative, *lower*-recovery end of the plausible-choice range for this synthetic, not a value
that was tuned upward to inflate the headline number. I cannot rule out the real donor spectrum
behaving differently (this is exactly what the missing conditioning diagnostic, BLOCK finding #1,
would tell you), but on the evidence available, this specific hyperparameter is not obviously
doing the paper's work for it.

**Reproduction:** `probe4_lambda_sensitivity.py` / `probe5_lambda_tight_ratio.py` in the scratch
directory — both run in `<2s`, no torch, no model load.

**Verdict: FLAG.** Real, measurable, correctly-disclosed-as-arbitrary sensitivity exists, but its
magnitude at the values actually used is modest (sub-percentage-point moving `λ_scale` down from
`1e-4`, single-digit percentage points moving it up an order of magnitude) — not the dominant
source of uncertainty in this instrument (that is the two BLOCKs above). No sweep in the actual
artefact tests this; a one-line λ-ablation at the same `down_proj@50%` point control 3 already
uses would settle it directly on real data at near-zero marginal cost.

---

## PASS — the fixed `reconstruct_row` genuinely IS exact at `S`=full regardless of `H`'s rank or conditioning, as long as `trace(H) > 0`; verified both analytically and against the live run's real weights

**What I checked.** The Builder's claim ("the corrected solve returns identity exactly, regardless
of `H`'s rank or conditioning") is algebraically true: `W_new = W·H_reg·H_reg⁻¹ = W` whenever
`H_reg = H + λI` is invertible, which `λ>0` guarantees regardless of `H`'s own rank — this does not
depend on any property of `H` except its trace being strictly positive (so that `λ = 1e-4·trace(H)/N
> 0`). Verified on seven adversarial synthetic `H`s (`probe1_identity_control.py`): rank-1, identity,
huge/tiny uniform scale, a near-singular (`cond≈1e14`) random-eigenbasis matrix, and a non-PSD
symmetric-garbage matrix — every one gives `max|dev| < 3e-10` at `S`=full except the one genuinely
degenerate edge case, `H` = all-zeros exactly (`dev=5.5e-2`, because `trace(H)=0 ⇒ λ=0 ⇒ H_reg=0` is
singular too, and `lstsq`'s minimum-norm fallback returns 0, not `W`) — a case that cannot occur
with real captured activations (it would require every single calibration token to produce exactly
zero activation on every feature of a layer, i.e. the layer is entirely dead across the whole
32×512-token calibration set) and is noted here only for completeness, not as a live risk.

**Confirmed against the actual donor model, not just synthetic data** — read from the *live run's*
own log without disturbing it: `CTRL identity_mask0pct_down_proj  BPB 0.76759  d=+0.00000
+-0.00000` (`d4_run.log`), and the JSON checkpoint gives the exact figures:
`delta=5.94e-09`, `max_abs_weight_deviation=2.98e-08` — essentially float32 machine epsilon. This
is real evidence, not just the synthetic reproduction, that the fix does what it claims on the
actual 128K-donor `down_proj` weights and the actual T=16384 calibration Hessian.

**Verdict: PASS.** The exactness-at-identity property is real, robust, and now independently
double-confirmed (synthetic + live donor data). This is precisely what makes it non-diagnostic of
`H`'s informativeness (see the two BLOCK findings above) — a control that is unconditionally true
cannot also be evidence of anything conditional, and that is the correct reading of "the fix hides
the problem it was fixing" from the brief: the fix is not wrong, it is simply no longer able to do
the discriminating work control 1 used to do, and that work has not been fully relocated anywhere
else in the instrument.

---

## PASS — the ROW-structured short-circuit (`gate_proj`) is implemented exactly as derived: no solve, no `H`, no numerical drift possible

**What I checked.** `reconstruct_row_axis_level` (`d4_reconstruction.py:296-312`) — the entire
function body for a ROW-structured organ is `Wnew = W.detach().clone(); Wnew[idx,:] = 0.0` (no
`H`, no Cholesky, no lstsq, no floating-point operation beyond a copy and a zero-fill). This is
bit-for-bit the same operation `d1b_organ_sweep_completion.py`'s `prune_block_structured` performs
for ROW-structured organs (`W[b*block_size:(b+1)*block_size,:] = 0.0`, `:88-89`) — so `gate_proj`'s
reconstructed `delta` should be **bit-identical** to D1's naive `delta_naive_d1` at every level,
not merely close, giving `recovery` exactly 0 (not "close to 0 within noise") once both runs are
compared. This is a sharp, checkable prediction that was not yet verifiable at audit time (the live
run had not reached the main sweep), given here as the reproduction path:

```python
import json
d4 = json.load(open("results/d4_reconstruction.json"))
for r in d4.get("organ_sweep", []):
    if r["organ"] == "gate_proj":
        print(r["level"], "delta=", r["delta"], "delta_naive_d1=", r["delta_naive_d1"],
              "recovery=", r["recovery"], "  (expect delta≈delta_naive_d1, recovery≈0)")
```

**Verdict: PASS**, on code inspection; the empirical confirmation is a one-line check to run once
the live process reaches `gate_proj` (do not run the whole model now — just read the finished
JSON later).

---

## What I tried to break and could not, and why it held

1. **Did the identity-control fix merely move the S=full exactness bug somewhere else (e.g. does
   it silently corrupt at partial `S`, not just `S`=full)?** No — `probe3_shuffle_control.py`'s
   `identity_H` arm exercises the solve at `S`≠full (the 50%-column mask) and still returns exactly
   naive masking (see the FLAG on `identity_H`'s tautological status above) — the closed-form math
   is correct for arbitrary `S`, not just the `S`=full edge case the Builder's own validation
   apparently targeted. I looked specifically for an off-by-one or index-alignment bug in
   `reconstruct_row`'s `index_select`/`index_copy_` calls (`:188-198`) that might only manifest at
   partial `S` — none found; `A = H_reg.index_select(0,S).index_select(1,S)` and
   `rhs = W_full @ H_reg.index_select(1,S)` correctly restrict to exactly the surviving-column
   submatrix on both sides.

2. **Does `control 2` (saturation, mask=100%) actually test anything, or is "naive-100% ==
   reconstructed-100% by construction" a way of dodging a real check?** The code's own comment says
   this explicitly (`:548-553`) — I tried to find a way this construction argument could be false
   (e.g. if `S`=empty were handled differently by `reconstruct_organ_level` vs. the naive zeroing
   path) and could not: `reconstruct_row`'s own `if S.numel()==0: return torch.zeros_like(W_full),
   False` (`:184-185`) is the literal same operation as directly zeroing `W`. Confirmed by the live
   run: `saturation` control fired (`delta=+3.745, fired=True`), consistent with a real, large,
   correctly-computed number, not a null result masquerading as a pass.

3. **Is `hessian_rank_ratio`'s tolerance (`eigvals.max()*D*eps`) at least internally consistent with
   standard numerical-rank conventions (i.e. is this a reasonable rank definition, even though it
   answers the wrong question for this project)?** Yes — this is exactly `numpy.linalg.matrix_rank`'s
   own default tolerance formula, and my synthetic reproductions (`probe1`, `probe5`) used
   `numpy.linalg.matrix_rank` directly for cross-checks and got matching rank counts to the
   `torch.linalg.eigvalsh`-based version. The *definition* of numerical rank is not the bug; using
   it as the sole conditioning gate for a regularised solve is the bug (BLOCK finding #1).

4. **Does `theoretical_rank_ratio`'s `min(T,D)/D` bound ever get used somewhere it could actually
   mislead (e.g. an organ where `T<D`)?** Checked every call site (`:600, :636`, both
   `down_proj`'s `shuffled_H` and `leakage_H` secondary arms) — both use `T_cal=16384 > D=8960`
   (`shuffled_H`) or `T_eval_tok = 24×512=12288 < D=8960` for the **leakage** arm's own capture
   (`:632-636`, using the eval slice's 24×512 tokens, not the calibration slice's 32×512) — this one
   is `T<D`! `theoretical_rank_ratio(12288, 8960) = min(12288,8960)/8960 = 1.0` still (since
   `T>D` here too, 12288>8960) — so it still reads 1.0, not actually a counter-example, but it is
   the one call site where the ratio was not guaranteed positive by construction at a glance and
   needed checking. No live bug found, but noted because it was close enough to be worth writing
   down: if the eval slice were ever shrunk below 8960 tokens, `leakage_H`'s reported rank ratio
   would (correctly, for once) drop below 1.0 and could fire the abort — the one code path where
   the diagnostic actually has teeth, by accident of which organ/slice combination it's applied to.

5. **Could `select_zero_blocks`'s `torch.argsort` tie-break non-determinism (documented as a class
   of bug elsewhere in this project, e.g. the D0/D1 PS `$R`/`$r` case-sensitivity note) silently
   produce a different mask between D1 and D4 runs?** I could not test this against real weights
   without loading the donor model (out of bounds this session), but the *algorithm* comparison
   (source-level, both quoted above) is identical, `torch.argsort`'s default is a stable-enough sort
   for non-tied float32 norms (ties on real-valued Frobenius norms of independent weight blocks are
   measure-zero), and the divisibility check (FLAG above) rules out the one concrete quantisation
   mechanism that has actually bitten this codebase (`k_proj`/`v_proj`@90%) at the three levels D4
   actually sweeps. Left as a FLAG (missing runtime assertion) rather than a found bug, since I
   could not exercise it against the real model this session.

---

## When D4's numbers land: what can be believed, what cannot, and what would have to be re-run

| what | can be believed as reported? | why / what's needed to fix |
|---|---|---|
| Control 1 (identity, `down_proj`@0%) fires with `delta≈0` | **Yes, fully.** | Provably exact (algebra) and empirically confirmed on real donor weights (`delta=5.9e-9`). Not diagnostic of `H`'s informativeness (that burden has moved to control 3), but as a "the plumbing works" check it is solid. |
| Control 2 (saturation, `down_proj`@100%) | **Yes**, as a self-consistency check only — it was never claimed to be more than that. | — |
| `gate_proj` sweep `recovery ≈ 0` at every level | **Will be exactly true by construction** once computed — not a finding about the mechanism, a mathematical guarantee of the code path taken. | Verify with the one-line check in the ROW-axis PASS section above; if it's NOT ≈0, that's a bug, not a result. |
| `down_proj`/`o_proj` sweep `recovery` **point estimates**, taken at face value as "% of quality Hessian-weighted reconstruction recovers" | **No — believe only after subtracting the `shuffled_H` baseline at the one point where it's measured, and treat the un-ablated sweep points (every level except `down_proj`@50%) as upper bounds, not point estimates, on genuine-correlation recovery.** | Need: (a) `shuffled_H` recovery at more than one (organ, level) point — currently only `down_proj`@50% is ablated; (b) ideally a shuffle construction that also destroys the mean (e.g. per-column mean-subtract before shuffling, or shuffle *row blocks* jointly across all columns instead of per-column independently) so `shuffled_H`'s off-diagonal really does go to ~0 rather than retaining `T·μᵢμⱼ`. |
| Reported `recovery_se` / INCONCLUSIVE-vs-not classification | **Believe the point estimate's rough scale; do not trust the classification boundary to the precision the `2σ` band implies**, especially for any point landing near `recovery≈0`. | Need a joint (shared-seed) bootstrap of `delta_naive` and `delta_recon` together, or at minimum report the `full_se` correction from the FLAG above alongside the current one. |
| `rank(H)/N = 1.000` diagnostic, as evidence the solve is numerically safe | **No — it provides zero information in this run's T>D regime, by mathematical construction, regardless of what it prints.** | Need `condition_number`, `λ_min`, `λ_max`, and a count of eigendirections with `λᵢ < ` the actual Tikhonov `λ` used downstream, computed and logged per layer (the eigenvalues already exist in memory inside `hessian_rank_ratio` — this is a few extra lines, not a new heavy computation, and does not require rerunning the calibration capture). |
| `LAMBDA_SCALE=1e-4`'s effect on the headline numbers | **Believe the numbers are not wildly λ-sensitive in the immediate neighbourhood of the chosen value** (synthetic sensitivity was sub-percentage-point moving down, single-digit pp moving up an order of magnitude, at D4's own T/D margin) — this is a secondary source of uncertainty, not the dominant one. | A one-point λ-ablation at `down_proj`@50% (already-ablated point, near-zero marginal cost) would settle this on real data instead of a synthetic proxy. |
| Mask agreement between D1 (`delta_naive`) and D4 (`delta_reconstructed`) at the same nominal `(organ, level)` | **Believe it for the three levels actually tested** (divisibility check above rules out quantisation divergence) **but do not generalise that confidence to any future level/organ** added to this sweep without the same check. | Add the runtime assertion given in the FLAG above; costs nothing at solve time, only a JSON lookup. |
