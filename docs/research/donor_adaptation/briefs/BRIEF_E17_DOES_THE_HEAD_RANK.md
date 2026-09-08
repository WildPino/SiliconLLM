# BRIEF E17 — does the ternary HEAD rank, and what does the ranking band actually cover?

**Status: PRE-REGISTERED.** Pushed before any arm ran. Directions are called in §6.

---

## 1. The question

E16 closed with `SCORE-CROSSES-RANK-DOES-NOT`: the 7 B arm `B2` reads **4.017233 BPB**, `0.052874`
*below* the 4.070106 chance line — 59× the 0.000896 band, real information — and agrees with its
own fp32 donor on **0 of 160** greedy tokens, diverging at token 0 on 5/5 prompts.

Every E16 arm was exported `--head-ternary` (`e16_r3_at_7b.py:167`). On `Qwen2.5-Coder-7B`
`tie_word_embeddings=False`, so that flag ternarizes a standalone **152064 × 3584 = 545 M-parameter**
output projection — the one tensor whose entire job is ranking. A ternary head is therefore a
candidate mechanism for "better than uniform on average, unable to pick a top-1 token."

E17 asks: **is the head what fails to rank?** And, because the answer needs a scale to be read
against: **what agreement rate does a model that is not broken actually produce?**

## 2. Why this is not already answered

**In BPB it is answered, and the answer is "the head is almost free."** E1 §4.3:

| donor | ternary FFN+attn (`TQ`) | + ternary head (`TQH`) | head's marginal cost |
|---|---|---|---|
| 0.5 B | `4.509163909048454` | `4.531233733626962` | **`+0.022069825`** (4.4 σ_seed) |
| 1.5 B | `3.484253077409102` | `3.475706691632780` | **`−0.008546386`** — free, and negative |

**In ranking it is not answered at any scale.** No probe has ever run a `TQ` arm through
`--generate`. E6 ran `A1` (0.5 B fp32), `A2` (0.5 B `TQH`), `A3` (1.5 B `TQH`); E7 ran the 7 B fp32
and packed; E16 ran three 7 B head-ternary arms. The head-fp32 cell is empty everywhere.

This is the exact shape E14's law was written against — *pair every scoring metric with a ranking
one* — applied to a pair where the scoring metric says the change is **free**.

**And the reference point E16 reasoned from is already known to be broken.** E16 chose `R3`
because it "works at 1.5 B": `qwen25-15b_tqh.bin` at `3.475707`, `0.594` below chance. That file is
E6's arm `A3`, which E6 labelled **"planted control at 1.5 B"** — a known-**negative** — on
2026-09-05, and which agrees with PyTorch on **10/160** tokens, diverging at token 0, emitting
`" the\n\n the\n the the\n the\n the\n the\n the"`. E16's brief read that artifact's BPB and not
its transcript. **Nothing in this programme has ever demonstrated a ternary arm that ranks, at any
scale.** E17 is where that is stated as a measured population rather than noticed twice.

## 3. Design

Five arms. Two are replications that must reproduce published numbers before anything new is read;
three are the empty cells. **Nothing is exported and nothing is re-ternarized** — every artifact is
already on disk from E1, and every reference is already on disk from E6.

| arm | weights | role | what it moves vs its pair |
|---|---|---|---|
| `H0a` | `e1\qwen25-05b_f32.bin` | **known-positive**, must FIRE | — (replicates E6 `A1`) |
| `H0b` | `e1\qwen25-15b_tqh.bin` | **known-negative**, must replicate | — (replicates E6 `A3`) |
| `H0c` | `e1\qwen25-15b_f32.bin` | **known-positive at the second scale**, new | — |
| `H1` | `e1\qwen25-15b_tq.bin` | the head axis at 1.5 B | `head_ternary` only, vs `H0b` |
| `H2` | `e1\qwen25-05b_tq.bin` | the head axis at 0.5 B | `head_ternary` only, vs E6 `A2` |

**The single-axis claim is checked in the sidecars before any agreement is read**, the way E16 §3
enforced it. `qwen25-15b_tq` vs `qwen25-15b_tqh` may differ on `head_ternary` (False→True),
`rule_applied_to` (7→8 entries, the added entry being exactly `lm_head`), `tied` (1→0), and on
`bytes`/`sha256`/`mean_ternary_zero_fraction`, and **on nothing else**. Same for the 0.5 B pair. If
any other field moved, more than one thing moved and the contrast is not the registered contrast.

**Reference**: `results/e6/ref.json`, PyTorch fp32 greedy, keyed by HF model id, already holding
both `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B`. `PROMPTS` and `N_NEW` are **imported from
`e6_generate`**, never re-typed — one definition, as E16's ranking run did.

**Binary**: `benchmarks/donor_adaptation/engine/donor_engine.exe` — the build E6 used, so `H0a` and
`H0b` are replications and not re-measurements. E16 used a *different* binary
(`D:\_ktmp\e13\donor_engine.exe`, 317952 B / 18:01 vs 315904 B / 15:25), and E17 compares numbers
across the two, so `G-H0d` tests that rather than assuming it.

## 4. Instruments

- `donor_engine.exe --weights W --threads 6 --generate <ids> 32 <prefix>`, greedy argmax.
- Agreement = positions where our id equals PyTorch's, denominator fixed at 5 × 32 = **160**,
  every position counted including those after a divergence has already split the contexts (E6's
  convention, kept so the numbers are comparable).
- `first_div` = (prompt, step) of the earliest disagreement.
- BPB is **not re-measured**. E1's engine column is quoted for all four ternary arms.

## 5. Gates and bands — registered before the run

**The known-positive band.** E11's law: draw it from *every* reading of the known-positive. Two
exist — E6 `A1` at 0.5 B and E7's 7 B fp32 — and both read **exactly 160/160 = 100%**. `H0c` adds
a third at 1.5 B. The band is therefore the point **100%**, and `G-H0a`/`G-H0c` must land on it.

**The known-negative population.** Five readings: `A2` 1.875%, `A3` 6.25%, E16 `B1`/`B2`/`B3` all
0%. **Ceiling 6.25%** (10/160).

**Thresholds, derived rather than chosen** (E8 §3: a measured quantity over a structural factor).
The gap between the negative ceiling and the positive band is `100 − 6.25 = 93.75` points.

- **`G-H0a`** `H0a` = 160/160 exactly, or **E17 is VOID**. The instrument must fire on the
  known-positive before any null it produces counts.
- **`G-H0b`** `H0b` = 10/160 with `first_div = (0,0)`, exactly, or **E17 is VOID** — the artifact and
  binary are not the ones E6 measured.
- **`G-H0c`** `H0c` = 160/160. *Not* a void condition: it is a new cell, and E1 §4.4 records that the
  engine once could not load a >2 GB file — a failure here is a finding about the fp32 path at
  1.5 B, reported, not a reason to discard the ternary arms.
- **`G-H0d`** `H0b` re-run under `D:\_ktmp\e13\donor_engine.exe` produces **token-identical** ids.
  If it does not, E16's and E6's agreement numbers are not comparable and every cross-probe
  statement in §5 of this brief is withdrawn.
- **`G-H1`** agreement(`H1`, 1.5 B, head fp32):
  - ≥ **53.125%** (85/160 — the negative ceiling plus half the gap) → **`HEAD-IS-THE-MECHANISM`**
  - ≤ **12.5%** (20/160 — twice the negative ceiling) → **`HEAD-IS-NOT-THE-MECHANISM`**
  - between → **`INDETERMINATE`**
- **`G-H2`** the same three labels at 0.5 B, `H2` against E6's `A2`.
- **`G-H3`** the pairing E14 requires, reported for both scales:
  `ΔBPB(head)` from E1 against `Δagreement(head)` from here. Descriptive; no label.
- **Stage 2, conditional and registered now so it is not chosen after the fact**: if `G-H1` reads
  `HEAD-IS-THE-MECHANISM`, export `Qwen2.5-Coder-7B` with `--rule R3 --calib-seqs 32 --fold layers`
  and **without** `--head-ternary`, then read its BPB and its greedy agreement against E7's stored
  `f32_p*.ids.bin`. If `G-H1` reads anything else, stage 2 **does not run** and this brief says so
  before the data exists.

## 6. Predictions — directions called

1. **`G-H0a` fires at 160/160** and `G-H0b` replicates 10/160. High confidence; these are the same
   file, binary and prompts.
2. **`G-H0d` passes** — E13's build was gated bit-identical on the packed path.
3. **`G-H1` → `HEAD-IS-NOT-THE-MECHANISM`, agreement in 2–15%.** Reasoning, stated so it can be
   scored as reasoning and not only as a number: at 1.5 B the fp32 donor sits `0.767595` BPB and the
   `TQ` arm `3.484253`. Converting to nats/token at `4.229452` bytes/token, the fp32 donor is
   **9.68 nats/token** better than uniform (`ln 151936 = 11.931`) and `TQ` is **1.72**. The
   body-only ternary arm therefore retains **≈18%** of the information the donor has over guessing,
   *before* the head is touched at all. A model holding 18% of its donor should not reproduce the
   donor's argmax.
   **This reasoning is one quantity extrapolated to a different metric, which is precisely the move
   E16's law names** (*a trend read from two points is a direction, not a law*) — weaker even than
   two points. It is registered as a direction, and if `G-H1` comes back at 60% the reasoning was
   wrong regardless of what the label says.
4. **`G-H2` → `HEAD-IS-NOT-THE-MECHANISM`**, and `H2 > A2` at 0.5 B (the head costs `+0.022` there,
   so removing it should help slightly), with both under 12.5%.
5. **`G-H3`: |Δagreement| < 10 points at both scales** — i.e. the head is nearly free in *ranking*
   too, matching its BPB. The interesting outcome would be a sign disagreement at 1.5 B, where the
   head *helps* BPB by `0.008546`.
6. **Stage 2 will not run**, because prediction 3 says `G-H1` will not fire.

## 7. What E17 cannot claim, whatever it returns

- **It does not supply the ranking band that E14 and E16 owe.** The known-positive at 100% is a
  *numerical-equivalence* reading — the same weights, engine against PyTorch — not "a different but
  equally good model." It bounds what identity produces, not what a harmless difference produces.
  **E14's 45.6% still has no band after E17**, and that item stays owed and open.
- **A null on the head does not identify the body.** Ruling one tensor out is not evidence for
  another. If `H1` also fails, E17 says the head is not *sufficient* to explain the failure, and
  nothing more.
- **No speed number moves.** Nothing here is timed and nothing may be quoted from it; 6.79 tok/s
  exact stands. (A head-fp32 7 B arm *would* move one — 545 M fp32 parameters is 2.18 GB per token
  against the whole packed model's ~1.77 GB — but stage 2 is conditional and would need its own
  idle-machine timing under the standing law.)
- **The 7 B readings enter only as members of the known-negative population**, never as a scale
  term: `Coder-7B` is a different model family from `Qwen2.5-0.5B/1.5B`, exactly as E16 §0 refused
  to read `+0.692619` as a pure scale term.
- **E1 §2.2's 1.5 B fp32 BPB is a 4-sequence subset and is not quoted.** `H0c` is a *generation*
  arm; it borrows no BPB.
- Agreement is a coarse instrument: 160 positions, 5 prompts, one decoding rule. It cannot
  distinguish "wrong model" from "right model, unlucky prompts", which is why the population and
  not any single reading carries the argument.

## 8. Cost and protocol

Five generate runs of 5 prompts × 32 tokens, on artifacts that already exist. Minutes, not hours.
Single job, no concurrent trainer or exporter. Decode rates are recorded by the engine and are
**explicitly not quotable** — the machine's idleness is not being controlled for and a contended
timing is not a timing.

## 9. Checked, not assumed

- Both `TQ` artifacts exist with sidecars reading `head_ternary: false`, `rule: R3`,
  `calib_seqs: 32`, `calib_matches_t2_operating_point: true` — read, not presumed.
- `results/e6/ref.json` holds **both** `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B` — verified by
  reading its top-level keys.
- The two engine binaries differ in size and timestamp → `G-H0d`, rather than an assumption.
- E6 tokenized the prompts with the **0.5 B** tokenizer and used those ids for the 1.5 B arm as
  well. Both are the same Qwen2.5 tokenizer at `V = 151936`, but the runner re-derives the ids under
  each arm's own tokenizer and asserts equality with `engine.json`'s stored `prompt_ids`, so the
  crossing is checked and not inherited. (`common.get_slice`'s tokenizer-blind cache key is not
  involved: no corpus slice is used here.)
- `qwen25-15b_tqh.bin` is byte-for-byte the file E16 loaded as `C0` and E6 loaded as `A3` — same
  path, same sha256 in the sidecar.
