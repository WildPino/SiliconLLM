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
