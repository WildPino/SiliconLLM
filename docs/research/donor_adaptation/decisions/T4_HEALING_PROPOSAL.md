# T4 healing — what I would spend the weeks on, costed, and what must happen first

**Written 2026-09-11, after E22.** This is the communication the goal directive asks for:
*"Se ti serve possiamo fare sessioni brevi (settimane) su T4 per rifiniture, in quel caso però
me lo devi comunicare."*

**Yes, I need them. Here is the target, the cost, the gates, and the one CPU probe that has to
land first so the weeks are not spent healing a fiction.**

---

## 0. Read this part even if you read nothing else

**Healing a 1.5 B does not reach the goal, and I am not going to pretend it does.** The goal is a
10 B at 50 tok/s. What the T4 weeks buy is a **structure validation**: proof that the structure
E22 identified can be *trained* into working, cheaply, on a donor small enough to iterate on. If
it validates, the structure is what a big model gets trained into — which is Phase 64's business,
not the donor programme's. If it fails, the donor route is finished and I will say so.

**The honest prior is against it.** Every post-hoc route this programme has tried has closed:
precision (E18), sparsity (E19), rule (E20), rank (E21), and composition-into-the-format (E22).
Healing is the only branch never tested, which is the reason to test it and not a reason to
expect it to work.

## 1. The target, and why it is this one

E22 measured the first configuration ever derived inside the 50 tok/s budget:

| `QO512+V52` on Qwen2.5-1.5B | |
|---|---|
| active weights/token, charged ternary | **`0.9441 G`** (budget `0.982–1.060 G`) |
| BPB, fp32 | `1.005039` (`+0.237444` over the intact donor) |
| teacher-forced top-1 | **`126/160`** — above every ternary head E20 measured |
| free-running greedy | `15/160` — three above the floor. **It does not generate.** |

**And the same configuration in the format it must ship in reads `4/160`** (`STACK`: ternary
low-rank factors, ternary FFN, ternary head). E22 §5: the two ternary factors' errors multiply,
so post-hoc conversion of the factored form costs `+0.908657` BPB against ternarizing the dense
matrix.

**That gap is exactly what training is for.** The job is not "make a damaged model better" in the
abstract. It is one specific thing: **learn the format instead of applying it**, starting from a
structure that already gets 79% of tokens right per step at fp32.

## 2. What must happen first — and it is CPU, free, and mine

**Every carve number in this programme — D0, D0c, E19, E22 — uses an ORACLE router.** It reads
the true squared activation mass and then keeps the top `k` experts. No real router has ever been
built here. D0c §132–135 says so in its own text: *"the oracle router flatters the carve"*.

So `V52`'s `+0.141846` and `QO512+V52`'s `126/160` are **ceilings**. Spending T4 weeks healing a
configuration whose FFN routing is a ceiling would be healing something that cannot be built.

**`E23` measures a real router on CPU, and it is pre-registered at
`briefs/BRIEF_E23_A_REAL_ROUTER.md`.** It is cheap, it is mine to run, and it decides whether the
healing target is `QO512+V52` as written or a shallower carve. **I will not ask you to launch
anything until E23 reports.**

## 3. The staged plan, costed

T4 on Kaggle, `fp16` (Turing — no bf16), 16 GB, 30 GPU-h/week/account × 3 accounts.
Teacher logits precomputed on **this** CPU box and shipped as data, so the 16 GB holds only the
student and its optimizer.

| stage | question | budget | gate to continue |
|---|---|---|---|
| **H0 — MVE** | does straight-through training move the ternary factored form *at all*? | **≤ 3 GPU-h** | `STACK`'s teacher-forced rises from `4/160` by **≥ 20 tokens**. Below that, stop. |
| **H1 — heal the format** | can QAT recover the fp32 configuration's per-step fidelity in ternary? | **≤ 25 GPU-h** | teacher-forced **≥ 107** (into E20's ternary band) at the `STACK` configuration |
| **H2 — heal the drift** | can it generate? | **≤ 60 GPU-h** | free-running **≥ 80/160** (`RANKS`), the band E17 fixed and nothing has ever reached |

**Total ≤ 88 GPU-h ≈ one week of one account's quota, spread over three.** H0 and H1 are the
cheap, decisive part; H2 is the one that would actually matter and the one I would not start
without H1 passing.

**Why the gates are where they are.** `4 → 24` at H0 is a signal-detection bar, not a success
bar — it only asks whether gradients move this object. `107` at H1 is E20's measured ternary
band, so passing it means *the trained ternary model is no worse per step than ternarizing a
single organ post-hoc*, which is the whole claim. `80` at H2 is `RANKS`, the band that says a
model reproduces its own donor's text — never reached by anything, by any route, in this
programme.

## 4. What I need from you, concretely

1. **Nothing yet.** E23 runs on CPU first and I will report it.
2. **Then: launch H0.** I will hand you a smoke-tested command and a `STOP` — the standing rule is
   that you launch the long training runs, I do smoke + stop + ready command.
3. **A decision after H0**, which is a 3 GPU-h question, not a week.

## 5. What would make me withdraw this request

- **E23 says the real router costs most of the carve.** Then `QO512+V52` is not the target and I
  would come back with a different one before asking for GPU.
- **H0 fails its gate.** Then post-hoc *and* trained conversion have both failed on this donor,
  the donor route is closed end to end, and the honest recommendation is to put everything into
  training a model into the format from scratch — Phase 64's ladder — and stop adapting donors.

**Either way the weeks are not wasted, because both outcomes close a route that is currently
open and expensive to leave open.**

## 6. Standing caveats that do not go away

- **No speed claim anywhere in here.** `6.79 tok/s` is exact and untouched. `engine.c` has no
  factored matvec and `QWENDON1` has no kind for one; until it does, no configuration in §1
  converts to a tok/s number, healed or not.
- **The head is untouched by all of this.** `233 M` at 1.5 B, `545 M` at 7 B = 51–56% of a 7 B's
  entire budget. Neither cut in §1 touches it and no lever measured here shrinks it (E21 §7).
- **1.5 B is not 10 B.** §0.
