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

## 2. The precondition is DISCHARGED, and it moved the target

**E23 has run.** `probes/E23_A_REAL_ROUTER.md` (`e23_router.json`, 7 arms, `VOID: none`). It
replaced the oracle with a per-layer ridge regression from the block input to `sqrt(group mass)`
— closed form, no gradients, no tuning, `11.0 M` charged to the budget — and the answer is split:

| configuration | oracle tf | **real-router tf** | retention | band |
|---|---|---|---|---|
| `V52` (carve only) | 117 | **110** | 0.9067 | **`ROUTER-HOLDS`** |
| `QO512+V52` (§1's target) | 126 | **102** | 0.7143 | **`ROUTER-COSTS`** |

**The carve is not an oracle artefact** — that half of E23's registered alternative fired. **But
§1's composed target is, partly**: the composition *gains* 9 teacher-forced tokens under the
oracle and *loses* 8 under a real router, because the oracle scores from true post-gate
activations while a real router reads the block input that the rank cut has already perturbed.

**So §1's `QO512+V52` at `k = 133` is NOT the H1/H2 target any more.** §5's withdrawal trigger
was *"the real router costs most of the carve"*; retention `0.71` is not "most", so the request
stands — but the brief's own band for `ROUTER-COSTS` says the target must be **re-derived at a
different depth**, and E23 §7 did that arithmetic:

> With the router charged, fixed cost is `354.5 M` (`q/o` at r=512 `88.1` + `k/v` `22.0` + head
> `233.4` + router `11.0`), so the FFN allowance is `627.5–705.5 M` = **54.28%–61.03% activation
> = `k = 139 … 156` of 256**. **`k = 133` gives `0.9551 G`, 2.8% UNDER the budget floor.**

`k = 133` was inherited from E19, fixed before any router cost anything. **The shallower carve is
not a retreat — it is the depth the budget always permitted, and it is free quality.** Whether it
recovers the 5 teacher-forced tokens needed to clear `107` is **not predicted**; that is **E24**,
CPU, mine, and it gates H1 and H2 — **not H0.**

**H0 IS UNAFFECTED AND CAN START NOW.** H0 trains the ternary low-rank attention factors with
**no FFN carve and no router at all** — a choice made when H0 was designed, specifically so that
E23 could not invalidate it. Its start state (`QO512-TB`, tf `28/160`), its fp32 ceiling
(`QO512`, tf `144/160`) and its gate (`≥ 48`) contain nothing E23 measured.

## 3. The staged plan, costed

T4 on Kaggle, `fp16` (Turing — no bf16), 16 GB, 30 GPU-h/week/account × 3 accounts.
Teacher logits precomputed on **this** CPU box and shipped as data, so the 16 GB holds only the
student and its optimizer.

| stage | question | budget | gate to continue |
|---|---|---|---|
| **H0 — MVE** | does straight-through training move the ternary factored form *at all*? | **≤ 3 GPU-h** | **`QO512-TB`'s teacher-forced rises from `28/160` to `≥ 48`** — the same `+20` delta, read on the ROUTER-FREE object. Below that, stop. |
| **H1 — heal the format** | can QAT recover the fp32 configuration's per-step fidelity in ternary? | **≤ 25 GPU-h** | teacher-forced **≥ 107** (into E20's ternary band) at the `STACK` configuration |
| **H2 — heal the drift** | can it generate? | **≤ 60 GPU-h** | free-running **≥ 80/160** (`RANKS`), the band E17 fixed and nothing has ever reached |

**Total ≤ 88 GPU-h ≈ one week of one account's quota, spread over three.** H0 and H1 are the
cheap, decisive part; H2 is the one that would actually matter and the one I would not start
without H1 passing.

**Why the gates are where they are.** `28 → 48` at H0 is a signal-detection bar, not a success
bar — it only asks whether gradients move this object. **The gate was restated from the original
`STACK` `4 → 24` when H0 was built**, for a reason that E23 then vindicated: `STACK` bundles the
ternary factors with a ternary FFN, a ternary head AND an oracle-routed carve, so a movement in
it could not be attributed, and the carve half was exactly what E23 went on to unsettle.
`QO512-TB` isolates the one thing H0 trains. The delta is unchanged at `+20`. `107` at H1 is E20's measured ternary
band, so passing it means *the trained ternary model is no worse per step than ternarizing a
single organ post-hoc*, which is the whole claim. `80` at H2 is `RANKS`, the band that says a
model reproduces its own donor's text — never reached by anything, by any route, in this
programme.

## 4. What I need from you, concretely

1. ~~**Nothing yet.** E23 runs on CPU first and I will report it.~~ **DONE — E23 reported, §2.**
2. **Launch H0.** I will hand you a smoke-tested command and a `STOP` — the standing rule is
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

## 5b. Two hardware facts measured on a real T4, which change the trainer

**These were measured on a Tesla T4 (`torch 2.10.0+cu128`) in a previous session and were sitting
in an untracked file. They are load-bearing and would each have cost a week.**
Artefact: `benchmarks/donor_adaptation/s1/results/nanhunt_gpu/nanhunt/fp16_gpu_nan_hunt_qwen2.5-1.5b.json`,
runner `s1/fp16_gpu_nan_hunt.py`, same frozen 512-token slice (`a1a48dc9…`).

**(1) This donor in fp16 on a T4 is non-finite under `eager` attention, and finite under `sdpa`.**

| config | finite? | first non-finite module |
|---|---|---|
| `eager`, fp16 | **NO** | **`model.layers.0.self_attn.o_proj`** — index 6 of 284 |
| `sdpa` default, fp16 | yes | — |
| `sdpa` MATH, fp16 | yes | — |
| `sdpa` EFFICIENT / FLASH, fp16 | *unavailable* | `RuntimeError: No available kernel` |
| `sdpa` default, fp32 (control) | yes | — |

The CPU probes that preceded it (`fp16_range_diag.py`, `fp16_first_nan.py`) had already refuted
"the numbers overflow fp16": everything is finite on CPU in fp16 at 128 and 512 tokens. **The
variable is the GPU execution path, not the dtype.** So **every T4 script here must pass
`attn_implementation="sdpa"`**, and flash/efficient attention are not options on this card.

**And the organ that blows up first is `o_proj` — exactly the organ H0 trains.** That is not a
coincidence to shrug at; it is the reason for (2).

**(2) ~~The factored form must carry an explicit fp32 scale, not fold the range into a factor.~~**

> **WITHDRAWN 2026-09-11, same day, by the file written to implement it.** I argued that E22's
> fold — A's column norms `c` pushed into B's rows — puts a dynamic range of `5.2e2` into `B·x`,
> the one intermediate fp16 must hold, and that an explicit `diag(s)` would pull it out. I had
> not measured it. `s1/h0_factorize.py` measures it, and **it is false in the opposite
> direction**: the fold is the *better*-conditioned half, on both axes I invoked, in 4/4 organs.
>
> | layer.organ | row scale of `B·x`, **folded** | unfolded | row-norm spread, **folded** | unfolded | `c` spread |
> |---|---|---|---|---|---|
> | `L00.q_proj` | **22.4** | 36.7 | **15.6** | 51.7 | 762 |
> | `L00.o_proj` | **3.4** | 24.0 | **2.9** | 28.0 | 78 |
> | `L27.q_proj` | **8.2** | 12.4 | **4.6** | 22.4 | 97 |
> | `L27.o_proj` | **7.2** | 41.0 | **4.7** | 62.8 | 286 |
>
> **Why I had it backwards.** `5.2e2` is the spread of `c` — a *weight-space* quantity — and I
> read it as an activation range. It is not. `c = ‖A column‖` and `‖B row‖` are near-perfectly
> reciprocal, `corr(log c, log‖B row‖) = −0.997 … −0.984`, because `A = W H^½ Bᵣ` carries the
> singular values and `B = Bᵣᵀ H^-½` carries their inverse. **Folding `c` into `B` balances the
> pair; it does not unbalance it.** That is the same reason E22's balanced arm worked at relative
> error `0.9874` and its registered arm failed at `1.5570`, which I should have read off E22's own
> diagnostic table before writing this section.
>
> **What is kept instead.** E22's fold, verbatim. `s` survives as something I can defend: a
> **per-rank learned scale initialised to ones** — the standard learned-step-size device in QAT, a
> continuous degree of freedom the ternary codes cannot express. At `s = 1` the initialisation is
> `QO512-TB` *identically*, so the trainer's planted control (`G-H0a`) is an identity rather than
> an argument, and it measures **exactly `0.0`** on codes, scales and products against
> `e22_compose.ternary_factors(balanced=True)`. `s` is 28,672 floats = **0.03% of the trainable
> mass**, so it cannot be doing the work on its own, and H0 can measure whether it does anything.
>
> **Item (1) is untouched.** The fp16/eager/`sdpa` result is measured on the card. Only my
> inference from it was wrong, and the inference is what is withdrawn.

**`bf16_supported_achieved` reported `True` on that T4**, contradicting the standing note that
Turing is fp16-only. It is emulated and slow, so H0 will not rely on it for throughput — but it
is available as a numerical fallback if fp16 loss-scaling misbehaves, and that is worth knowing
before the week starts rather than during it.

## 6. Standing caveats that do not go away

- **No speed claim anywhere in here.** `6.79 tok/s` is exact and untouched. `engine.c` has no
  factored matvec and `QWENDON1` has no kind for one; until it does, no configuration in §1
  converts to a tok/s number, healed or not.
- **The head is untouched by all of this.** `233 M` at 1.5 B, `545 M` at 7 B = 51–56% of a 7 B's
  entire budget. Neither cut in §1 touches it and no lever measured here shrinks it (E21 §7).
- **1.5 B is not 10 B.** §0.
