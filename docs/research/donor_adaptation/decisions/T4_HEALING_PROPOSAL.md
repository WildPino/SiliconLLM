# T4 healing — what I would spend the weeks on, costed, and what must happen first

> **SUPERSEDED NUMBER, 2026-09-12, from E28 / E30 / E31.** Every budget figure in this
> document is derived from `THROUGHPUT_G = 49.9 G active weights/s`, which gave **`<= 0.998 G`
> active weights per token for 50 tok/s at T10**. That constant has moved three times since, and
> **the budget now depends on HOW the arm reads its weights**:
>
> | | budget for 50 tok/s at T10 | applies to |
> |---|---|---|
> | as written here (E10/E25) | `0.998 G` charged | — superseded |
> | E28 `CONTAINER-COSTS` | `1.2328 G` charged | E13's layout at the goal's shape, `61.64 G-w/s` |
> | **E30 `AT-THE-WALL`** | **`1.452 G` moved** | the PHYSICAL ceiling: this box's entire `36.30 GB/s` with a perfect kernel. **Nothing can exceed it.** |
> | **E31 `GATHER-COSTS`** | **`1.371 G` in `>= 32 KB` blocks, `0.840 G` at row granularity** | **only arms that ACTIVATE A SUBSET** — carve, MoE, structured sparsity |
>
> **A dense or rank arm streams its weights contiguously and pays no gather penalty**: it is
> priced against E30's line, not E31's. **A carved or routed arm is priced against E31's**, and
> which E31 row applies depends on the exporter's group size.
>
> **For this document specifically**: the proposed object is a **rank** object at `0.9441 G`
> charged, and a rank arm STREAMS — E31's gather penalty does not apply to it. Against E30's
> physical ceiling of `1.452 G` moved it sits comfortably inside, **more comfortably than the
> `0.982–1.060 G` window quoted in §1**. Nothing in §§1–9 needs re-deciding on budget grounds.
> What E30 and E31 change is the *road after* H0: see `probes/E30_IS_THE_ENGINE_AT_THE_WALL.md`
> §6 and `probes/E31_WHAT_A_GATHERED_BYTE_COSTS.md` §6, which together say the remaining gap is
> architectural and name the activation granularity it has to respect.


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

**(3) A third trap, found by building the trainer, that would have FAKED H0's null.**
`density/common.py` line 28 calls `torch.set_grad_enabled(False)` **at module import**. That is
correct for every probe in this programme — D0 through E23 are all inference — and it saves
memory. But **any trainer that imports it, directly or transitively, silently trains nothing**:
every gradient is `None`, the optimizer steps on nothing, the loss curve is flat, and the run
reports *"gradients do not move this object"* — **which is word for word H0's null hypothesis.**
A T4 week would have come back with a confident, wrong FAIL, and §5's second withdrawal
condition would have closed the donor route on an artefact.

`common.py` is **not** changed — every published number depends on its semantics and nothing
else in the repo trains. The defence is in H0 instead, in three layers:

- `h0_qat.py` imports **nothing** from this repo (the Kaggle bundle is self-contained anyway),
  and re-enables grad explicitly with an assert.
- **`G-H0e`**, the planted control for the trainer itself: three masters are snapshotted and
  must have **moved** after the first optimizer step, or the run aborts. Measured in the CPU
  smoke at `1.59e-04` on all three. **A null from H0 only means something if the instrument can
  be shown to move first** — the planted-control law, applied to a trainer rather than a probe.
- `h0_selftest.py` re-enables grad before it checks the straight-through derivative, which is
  how the trap surfaced at all.

A second, independent instance of the same class: with the embedding frozen and gradient
checkpointing on, the checkpointed blocks receive inputs that do not require grad, so autograd
builds no graph through them and **every master grad is `None` again** — same silent flat loss.
Fixed with `model.enable_input_require_grads()`. `G-H0e` catches both, which is the point of
having it rather than trusting the fix.

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

---

## 7. H0 IS BUILT AND VERIFIED — the handover, 2026-09-11

**Bundle:** `benchmarks/donor_adaptation/s1/_h0_bundle/` — 417.0 MB, `RUN.md` and `MANIFEST.json`
inside it. Commits `e3dcdd1`, `511cf42`, `cd92b36`.

### 7.1 Every gate, measured

| gate | what it asserts | result |
|---|---|---|
| **`G-H0a`** | the masters ARE E22's `QO512-TB`: ternary codes identical as integers, scales identical, products identical, vs `e22_compose.ternary_factors(balanced=True)` | **FIRES at exactly `0.0`, 56/56 organs** |
| **`G-H0b`** | the self-contained trainer's quantizer == `t2_rules.r3_actsearch`, on the real masters | **112/112 factors, codes and scales** |
| **`G-H0c`** | the assembled module's forward == E22's ternary product, probed with the identity | **`1.665e-07` relative** |
| **`G-H0d`** | the straight-through derivative is the identity it claims | **exactly `1`** |
| **`G-H0e`** | the masters MOVE after one optimizer step, or the run aborts | **`1.59e-04` on all three watched, CPU smoke** |
| **end-to-end** | the assembled model reproduces E22's published start state | **tf `28` ✓, free `1` ✓, mean rank `1476` ✓, BPB `4.34e-06` relative** |

`88,109,056` trainable fp32 masters — matching to the parameter the `88.1 M` the budget
arithmetic charges to rank-512 `q/o`. Training stream `16,000,000` tokens from `calib.txt` only,
seed `90011`, disjoint from the frozen eval half by corpus construction.

### 7.2 One new fact, found by building it

**The factored form has never before been EXECUTED as a factored matvec here.** E21 and E22 both
computed `A·B` and installed the result as a single dense matrix. `h0_eval.py` installs a module
that actually runs two GEMVs with an intermediate of size `r = 512` — **the form `engine.c` would
use**. The two agree exactly on both token metrics and to `4.34e-06` relative on BPB, the
residue being fp32 summation order over 512 terms.

That is a small, real, favourable result: **the factored execution path is not a source of
error beyond round-off.** It does **not** discharge the owed `engine.c` item — there is still no
factored matvec in the engine and no kind for one in `QWENDON1`, so **no tok/s number moves and
`6.79 tok/s` stays exact.**

### 7.3 What is still true and unpleasant

- **§0 stands.** A healed 1.5 B is a structure validation, not the goal.
- **The honest prior is still against it.** Five post-hoc routes have closed.
- **`>= 48` is a signal-detection bar, not success.** Clearing it means gradients move this
  object; it does not mean the object works. `RANKS >= 80` free-running is H2's bar and nothing
  in this programme has ever reached it.
- **A FAIL closes the donor route end to end**, and §5's second withdrawal condition then
  applies in full.

---

## 8. §6's blocker is gone — and the distance is now measured, not derived

**2026-09-11, after E25** (`probes/E25_WHAT_THE_RANK_COSTS.md`).

§6 listed, as the item that keeps every rank result off the speed ledger, that `engine.c` has no
factored matvec and `QWENDON1` no kind for one. **Both exist**: a factored `mat_t` kind running
`y = A·(s ⊙ (B·x)) + b` as two calls to the same kernels, and `quant == 3`, a tagged container
in which every matrix carries its own kind — which is what finally makes *this proposal's own
object* expressible in a file, since it leaves `k/v` fp32 while cutting `q/o`. The exported
`QO512-TB` artifact passes end-to-end parity against `h0_qat.TernaryLowRank` at
`6.445e-04` / top-1 `1.0000`.

**This changes nothing about H0's gate**, which is `tf ≥ 48` measured on CPU in fp32, and nothing
about the T4 request, which is unchanged and still waiting. What it changes is the honesty of the
surrounding arithmetic:

- The rank charge `2·D·r` that §3's budget table and E23 §7's `k = 139…156` both rest on is
  **measured correct**: at the goal's shape the factored form returns `+14.12%` against a
  `+12.86%` byte prediction, and the byte-neutral planted control shows the extra call costs
  less than this box can resolve.
- **The distance to the goal is now a measurement.** A 10.60 G-active shape with the goal's
  dimensions runs at **4.70 tok/s** on this box — 10.6× short of 50, 9.3× with rank-512 `q/o`.
  E18 §31's budget band maps to 47.0–50.7 tok/s at that measured throughput, so the budget line
  holds at the shape it was written for.
- **Nine tenths of the remaining gap is the FFN** (8.45 G of `T10`). That is E19's carve and
  E23/E24's router, and **E24 is still unrun** — which is where the next CPU work belongs, not
  on the GPU.

---

## 9. E24 restores the H1/H2 target — at a different depth, and with a caveat that is new

**2026-09-11, after E24** (`probes/E24_THE_DEPTH_THE_BUDGET_PERMITS.md`, brief `b78ce4d`).

§8 ended with "E24 is still unrun — which is where the next CPU work belongs". It has now run,
nine arms, `VOID: none`, no timing taken.

**The verdict is `DEPTH-RECOVERS`, and E24 §5's registered alternative is the one that fired.**
E23's `ROUTER-COSTS` — the result that put this proposal's H1/H2 request in doubt — was measured
at `k = 133`, a depth inherited from E19 and fixed before a router cost anything, and **2.8%
below the budget floor**. At the depths the budget actually permits, with the same closed-form
ridge router charged at `11.0 M` and executed:

| depth | active/token | teacher-forced | band |
|---|---|---|---|
| `k = 133` (E23's, out of budget) | 0.9551 G | 102/160 | WORSE |
| `k = 139` | 0.9822 G | 105/160 | WORSE |
| `k = 148` | 1.0228 G | 110/160 | **COMPARABLE** |
| **`k = 156`** | **1.0590 G** | **118/160** | **COMPARABLE** |

**So the healing target is restored, and it is `QO512 + K156` (`E = 256`, `k = 156`), not
`QO512 + V52` (`k = 133`).** That object is in budget, uses no oracle anywhere, and ranks
COMPARABLE with the router included in its own cost. Nothing new was invented to get there — it
is the depth E23 §7's arithmetic had already permitted.

**What is still 12 tokens away, and is what healing would be for.** `K156-ORACLE` reads `130`
against `K156-LINEAR`'s `118`. The oracle is a ceiling no router can reach, but the gap is the
measured size of what a better router — or training the model to live with this one — has to
work with. It was 24 tokens at `k = 133`; going 9 points shallower halved it.

**The caveat, which is new and belongs here rather than in a footnote.** E24 §6 transposes the
winning recipe to `T10`, the goal's shape, on E25's measured charged throughput: `QO512 + K156`
is `6.14 G` active there = **≈ 8.1 tok/s, not 50**, and at `T10` the 50 tok/s budget leaves the
FFN only `k ≈ 2 … 8` of 256 — and **nothing at all** if `q/o` stays dense or is cut only to
`r = D/3`. The 1.5 B lands in the budget band at 58–61% activation *because it is a 1.5 B*.
**Healing `QO512 + K156` would therefore validate the mechanism at the donor's scale; it would
not, on its own, produce the goal's artifact.** The lever that opens the goal's shape is the one
that lowers the non-FFN floor — rank on `q/o` at `D = 4096`, then the head — which is E21 §8's
and E17's open ground, and which is unmeasured on quality at `r/D = 1/8` and `1/16`.

**Nothing here changes H0.** H0 is router-free and carve-free by construction, its gate is still
`tf ≥ 48` measured on CPU in fp32, and it is still the first thing the T4 is being asked for.
H1/H2, if they are requested, should now be requested at `k = 156` and their success criterion
read against `118/160`, not `102/160`.

---

> **E29 (2026-09-12) checked this recommendation against a fair control and it stands, at half the
> advertised size.** §10 rests on `L21-MINRES`'s `113/160`, quoted in E27 as `+56` over `L21-LAST`.
> `LAST` was a bad control. Against seven layers drawn at random from the same interior band the
> margin is **`+24`** (113 vs 89.0 over three seeds, spread 3), and the anti-rule `MAXRES-IN` reads
> 78 — so the residual does rank layers and `L21-MINRES` is still the right candidate, but the
> case for it is 24 tokens wide, not 56. `probes/E29_DOES_THE_RESIDUAL_RANK.md`.

## 10. E27 closes the hope §9 ended on — and names a cheaper H-arm than either

§9 closed by naming the lever that would open the goal's shape: *"rank on `q/o` at `D = 4096`,
then the head … unmeasured on quality at `r/D = 1/8` and `1/16`."* **E27 measured exactly those
two fractions and they do not survive:**

| `r/D` | teacher-forced | BPB |
|---|---|---|
| 1/3 (E21's validated point) | **144/160** | 0.820284 |
| 1/8 | 56/160 | 1.856378 |
| 1/16 | 42/160 | 2.097275 |

It is a cliff, not a slope. **There is no rank fraction that lowers the `T10` floor into the
budget and leaves a model behind**, and `FLOOR-MIN` — `r_qo = 192`, `r_kv = 96`, 36 of 48 layers,
the cheapest floor E27 could assemble and the only arm with real FFN room (`k = 17.3` of 256) —
reads `38/160`. §9's closing sentence should now be read as answered in the negative.

**What E27 found instead is worth more to this proposal than what it closed.** At 21 of 28 layers,
choosing *which* layers to drop by the pre-registered `MINRES` rule instead of dropping the last
`n` is worth **56 teacher-forced tokens at identical cost**:

| arm | layers dropped | charged | teacher-forced | BPB |
|---|---|---|---|---|
| `L21-LAST` | 21–27 | 1.2160 G | 57/160 | 2.500083 |
| **`L21-MINRES`** | **12–18** | **1.2160 G** | **113/160** | **0.993446** |

**`L21-MINRES` is the best quality-per-weight point the donor branch has produced without an
oracle, and it is the cheapest thing to train in this whole proposal.** It has **no router to fit**,
**nothing to calibrate at inference**, and **no factorisation to keep numerically stable** — it is
the donor with seven contiguous middle blocks deleted and `layer_idx` reindexed. Compared with
`QO512 + K156`, which needs a ridge router fitted on a calibration slice and a low-rank form that
has to survive ternarisation (E22's `+0.908657` BPB when both factors are ternarised), it is a much
smaller ask of a T4.

**Proposed, but NOT requested yet.** If H0 returns and its gate fires, the natural next arm is
`H3 = L21-MINRES`, healed, with its criterion read against **`113/160`** — the same way §9 reset
H1/H2's criterion to `118/160`. I am not asking for it now: **H0 is still the only outstanding
T4 request, and nothing here changes it.** H0 remains router-free and carve-free by construction,
its gate is still `tf ≥ 48` measured on CPU in fp32, and E23, E24 and E27 all leave it untouched.

**And the honest caveat, in the same place as the last one.** `L21-MINRES` at `T10` is `7.75 tok/s`
with a dense FFN, not 50. The floor alone at 36 layers is `1.6819 G` = **1.68× the entire per-token
budget**, so even deleting the FFN completely reaches `29.7 tok/s`. Healing `L21-MINRES` would
establish that depth surgery plus healing is a real mechanism at the donor's scale. **It would not
produce the goal's artifact, and this proposal should stop implying that any single one of these
arms can.**

---

## 11. H0 RAN, AND ITS GATE PASSED — 2026-09-12

**Probe**: `probes/H0_TRAINING_INTO_THE_FORMAT.md`. **Gate as registered in §3**: *"`QO512-TB`'s
teacher-forced rises from `28/160` to `≥ 48`. Below that, stop."*

| | BPB | vs intact | free | **teacher-forced** |
|---|---|---|---|---|
| `base`, intact — **the planted control, run today** | **0.7675949641** (E22: 0.767595, **diff −3.6e-08**) | — | **160** | **160** |
| `QO512-TB` **init** — the low anchor, run today | **2.8122382** (E22: 2.812226) | +2.044643 | 1 | **28** |
| **`QO512-TB` trained** | **0.825358** | **+0.057763** | **8** | **111** |

**PASS, at 111 against a bar of 48**, and **97.2% of the BPB damage removed** — on **500 of 4000
steps**, because the job stopped itself at its 2.8 h wall. Both anchors reproduce E22, so the
instrument is the one §3 specified.

**§0's own framing holds exactly as written.** This is a **structure validation**, not the goal:
1.5 B, `q/o` only, no carve, no router, no ternary FFN, no ternary head. §0 said *"if it
validates, the structure is what a big model gets trained into — which is Phase 64's business."*
It validated.

**And §0's honest prior was wrong in the productive direction.** Every post-hoc route had closed
(E18, E19, E20, E21, E22 — and E37 closed the carve on the same day). The one branch never tested
is the one that moved.

**The axis that did NOT move, and it is the one the goal's sentence is about**: free-running
generation reads **8/160**, in the `AT-FLOOR` band, *below* E22's fp32 `QO512+V52` at `15/160`.
**Teacher-forced and BPB recovered; the generator did not.** No arm proposed in this document
addresses that directly, and the next request should.

**What §4 asked for is discharged.** There is no outstanding T4 request. The three cheapest
follow-ups, in the order I would ask for them:

1. **Finish H0's own schedule** — the remaining 3500 steps, ~20 GPU-h. The cheapest question in
   this document now that the gate is passed, and the only one whose answer is already half paid.
2. **`STACK` under training** — the object that actually ships (ternary factors + `V52` + ternary
   FFN + ternary head), which reads `4/160` post-hoc and which H0 deliberately excluded.
3. **The carve, trained rather than applied.** E37 measured that applying it costs `+0.5537` BPB
   at the speed's activation rate, **with the router provably not the constraint**. H0 measured
   that the format can be trained into. **Nobody has put those two facts in one run**, and that
   is now the single most informative GPU-hour available to this programme.

---

## 12. H0 RAN AGAIN — the healing had NOT stopped, and H0 closes anyway — 2026-09-12

**Run 3**: `results/h0_kaggle_run3/h0_trained3.{npz,json}`, eval `results/h0/h0_eval_trained3.json`.
Resumed from a file **bit-identical** to run 2's output (`sha256` both begin `6d3fd3e3d0a15f5c`),
so this is a genuine cumulative **1000 of 4000 steps** — with `adam_state_restarted: true`, which
§12.4 charges. It hit the same **2.8 h wall** at step 500, not the ~1100 that was estimated.

### 12.1 The four rows, one instrument, all re-run today

| | BPB | vs intact | free | band | **tf** | mean rank | rank≤5 |
|---|---|---|---|---|---|---|---|
| `base`, intact — **planted control** | **0.767595** | — | **160** | `RANKS` | **160** | **1.00** | 160 |
| `QO512-TB` **init** — low anchor | 2.812238 | +2.044643 | 1 | `AT-FLOOR` | **28** | 1476.41 | 68 |
| **run 2** — 500 steps | 0.825358 | +0.057763 | 8 | `AT-FLOOR` | 111 | **7.10** | 144 |
| **run 3** — 1000 steps | **0.810022** | **+0.042428** | **15** | **`PARTIAL`** | **115** | **20.51** | 148 |

**The planted control was re-run today against this exact reading** and reproduces the morning's
intact row **bit-exactly** — `bpb` diff `0.0` to the last bit, 160/160 on both axes, and the
same generated text token for token (`results/h0/h0_eval_intact_recheck.json`). `h0_eval.py` is
unchanged since `511cf42`. **The instrument fires on the known-positive, so its readings count.**

**Gate `tf ≥ 48`: PASS at 115, delta +87.** Damage removed **97.175% → 97.925%**. The second 500
steps removed **26.6% of the residual that survived the first 500**.

### 12.2 The finding, and it is about the METRIC, not about the GPU

The operator reported the in-training fp16 proxy as **111 → 112 → 110, flat**, and explicitly
declined to interpret it. On the registered CPU fp32 instrument the same window reads:

- **BPB 0.825358 → 0.810022** — moved;
- **free-running 8 → 15/160**, band `AT-FLOOR` → `PARTIAL` — nearly doubled;
- **tf 111 → 115** — **also nearly flat.**

**So the proxy was not wrong about `tf`. `tf` really is flat. The error is that the gate is
written on `tf`, and `tf` is the one axis with almost no headroom left** — at 115/160 with
rank≤5 at 148/160, a further 500 steps of real healing can only show up in it as single counts.
The axes that moved are the **continuous** one (BPB) and the one with **145 counts of headroom**
(free-running). That is mine to own: §3 chose `tf` as the gate, and §11 read progress off it.

This is **E14 §3 running in reverse.** E14 §3 says every SCORE metric needs a RANK partner, because
a score can move while the ordering does not. Here the gate is the count-like metric and it
**saturated**, and only its SCORE partner could see that the run was still working. **A gate is a
floor, not a progress meter** — and nothing in this programme had said so out loud before.

**Registered consequence:** no future H-arm reports progress on `tf` alone. `tf` stays the gate
(it is a floor and it is doing its job); **BPB and free-running are what a continuation is judged
on.** The in-training fp16 proxy stays useful for exactly one thing — confirming the run is alive
and finite — and is not evidence about whether to buy more steps.

### 12.3 The counter-signal, which must not be buried

**Mean rank got WORSE: 7.10 → 20.51**, while **rank≤5 got better, 144 → 148**. Both are true and
they are not in conflict: the body of the distribution **tightened** (four more tokens entered the
top five) and the **tail lengthened** — mean rank is an average over 160 and a handful of very bad
positions dominate it. BPB, free-running and rank≤5 all improved; mean rank is the single number
that went the other way, and it is reported here because it went the other way.

It does **not** flip the reading — BPB is the calibrated continuous measure and it fell — but it
is the seed of a real question: **whether extended healing trades tail behaviour for body
accuracy.** Nobody has measured that, and one more H0 session would answer it for free.

### 12.4 What run 3 does NOT establish

1. **Two points cannot fit a curve.** 500 and 1000 steps are the only two trained readings that
   exist. Any claim about where H0 asymptotes is a **desk model**, and is labelled as one below.
2. **`adam_state_restarted: true`.** The second 500 steps ran on a **fresh Adam state**, so
   "1000 steps" is not the same object as 1000 contiguous steps would be. The measured
   improvement is therefore, if anything, a **lower bound** on what a contiguous schedule does —
   but that direction is an argument, not a measurement.
3. **Still H0's scope**: 1.5 B, `q/o` only, **no carve, no router, no ternary FFN, no ternary
   head**. Nothing here touches the object that ships, and nothing here is a speed number.

**DESK MODEL, explicitly not a measurement** — holding the *fractional* residual removal (26.6%
per 500 steps) constant, which is exactly the assumption two points cannot test:

| cumulative steps | residual | BPB | damage removed |
|---|---|---|---|
| 1,000 *(measured)* | **0.042428** | **0.810022** | **97.925%** |
| 2,000 | 0.022890 | 0.790485 | 98.880% |
| 3,000 | 0.012349 | 0.779944 | 99.396% |
| 4,000 *(full schedule)* | 0.006662 | 0.774257 | 99.674% |

### 12.5 The decision this was bought to make — the registered branch FIRES, and H0 closes

The branch was **pre-registered in `COMMUNICATION.md` APERTO 1, before the session was launched**,
and it was written on `tf`:

> *"Se il numero torna vicino a 111, i 3500 step che restano non valgono sei sessioni e H0 e
> finito a 111. Se sale, allora te le chiedo, con il motivo in mano."*

**`tf` came back at 115.** On a 160-count integer metric that is "vicino a 111". **The registered
branch fires: H0's remaining 3,000 steps — ~17 GPU-h, six more 2.8 h sessions at the measured
rate — are NOT requested, and H0 closes here.**

**And I am not allowed to reverse that because BPB is more interesting.** This is the E40
addendum A precedent applied to a branch instead of a gate: *a registered rule that fires is
doing its job and is not re-argued once the data are in.* The rule was mine, it was written
down before the run, and I had the whole session to write it on BPB instead.

**Stated plainly, because it is the uncomfortable half: had the branch been registered on BPB, it
would have gone the other way.** BPB fell 0.0153 and the second 500 steps removed a quarter of
the surviving damage — on that metric the schedule visibly has more in it. The decision therefore
rests on a metric choice §12.2 has just shown to be a poor progress meter, and that is a defect
in the pre-registration, recorded here rather than repaired retroactively.

**What makes me comfortable is that the cost argument agrees independently.** H0 is the structure
validation — 1.5 B, `q/o` only, no carve, no router, no ternary FFN, no ternary head. It has
already returned its finding (the format **can** be trained into, gate passed at +87) and 97.9%
of the damage is gone. Seventeen GPU-h to move 97.9% → ~99.7% (desk model, §12.4) on **an object
that does not ship** is the most expensive remaining question in this document, not the cheapest.
§11 ranked #1 "cheapest" when the healing curve was unknown; **run 3 bought that unknown for
2.8 h, which is exactly what one session was for.**

### 12.6 Where the GPU hours go instead — APERTO 0 unchanged, and STRONGER

**APERTO 0 stands exactly as written: H1, the carve trained rather than applied, two 2.8 h
sessions (~5.6 GPU-h).** Run 3 does not weaken it and does not merely leave it alone:

- §11's #3 is still the single most informative GPU-hour available, for §11's own reason: E37
  measured that *applying* the carve costs **+0.5537 BPB** at the speed's activation rate **with
  the router provably not the constraint**; H0 measured that the format can be *trained* into;
  **nobody has put those two facts in one run.**
- **Run 3 says one session will under-read H1.** H0's first 500 steps left 2.8% of the damage and
  the *next* 500 removed **a quarter of what remained**. An H1 session that reads disappointing
  at 500 steps **cannot be called a failure** — H0 would have been called flat at that point, by
  this very document, on this very metric. **APERTO 0 already asked for two sessions; run 3 is
  why the second one is not optional, and why H1 must be judged on BPB and free-running.**

**What run 3 actually delivered is an instrument finding, not an H0 finding** — and instrument
findings transfer, which is why 2.8 h on a non-shipping object was still worth spending:
*healing continues measurably past the point where the cheap in-training signal says it has
stopped, and the gate metric is not the place to look for it.*

---

## 13. E57 CHANGES WHICH ARM IS WORTH THE HOURS — and it prices the 10 B clause of the goal for the first time

**2026-09-13, after E57.** Nothing here is requested yet: **APERTO 0 (H1, in flight) stays the
only open ask**, and this section exists so that the *next* one is chosen on measured grounds.

### 13.1 What E57 measured that this document did not have

Every arm in §§1–12 was chosen from **BPB and teacher-forced counts**. Nobody had a **rate** for
a trained model — E17 §7 explicitly refused to quote its decode rates, and every tok/s in the
programme was synthetic. E57 supplies both halves in one table
(`briefs/BRIEF_E57_…md` addendum A, ledger §56):

| arm | rate (p25) | greedy vs HF | BPB (E1, R3, `fold=none`) | chance = 4.070 |
|---|---|---|---|---|
| `05b_f32` | **19.09** | **160/160** | 0.872 | −3.198 |
| `05b_tq` | **43.69** | 3/160 | **4.509** | **+0.439 ABOVE** |
| `05b_tqh` | **83.41** | 3/160 | 4.531 | **+0.461 ABOVE** |
| `15b_f32` | 6.25 | **160/160** | 0.702 | −3.368 |
| `15b_tq` | 19.15 | 12/160 | 3.484 | −0.586 |
| `15b_tqh` | 29.18 | 10/160 | 3.476 | −0.594 |

**Three things follow that bear directly on where hours go.**

1. **The organ is the BODY, not the head** (addendum C.3): `tq → tqh` costs **+0.022 BPB** at
   0.5 B and **−0.0085** at 1.5 B, where ternarising the head *improves* BPB. H0 healed the
   *factored `q/o`* — a body organ — and that choice is now measured to be the right family.
2. **At 0.5 B the post-hoc ternary model is above chance.** Healing at 0.5 B would have to
   rebuild a non-model, not repair a damaged one. **H0's 1.5 B remains the right size**, and
   E57 is the reason, measured: at 1.5 B the converted model still predicts (0.586 below
   chance) and there is something for gradients to hold on to.
3. **A healed arm now has a RATE waiting for it.** `15b_tq` runs at **19.15 tok/s** and
   `15b_tqh` at **29.18**. If healing recovers usable quality on the ternary body at 1.5 B,
   the programme gets its first point that is **trained, faithful-enough, and timed** — and it
   lands at 1.5–1.9× the fp32 arm's 19.09 with a real quality number attached.

### 13.2 The arm E57 would put next, costed — **`H2T`, heal the ternary BODY at 1.5 B**

| | |
|---|---|
| **object** | `Qwen2.5-1.5B`, body ternary by `R3`, head fp32 — i.e. the **`15b_tq` artifact that already exists and already has a rate** |
| **trained** | fp32 masters of the ternarised body organs, STE through the shipped quantizer, exactly H0's mechanism (`s1/h0_qat.py`), head/embedding/norms frozen |
| **why it is not H0 again** | H0 trained `q/o` **low-rank factors** (88.1 M, 5.7% of the donor) as a *structure* validation on an object that does not ship. `H2T` trains **the organs the shipped artifact actually converts**, and the artifact is one this engine already runs at a measured 19.15 tok/s |
| **cost** | **2 sessions × 2.8 h = 5.6 GPU-h** for the first read, on H0's measured step rate; §12.6's rule applies — *one session will under-read it* |
| **criterion** | **BPB and free-running**, never `tf` (§12.2, and `feedback_gate_is_not_a_progress_meter`): the bar is BPB crossing **below 1.5** (the converted arm sits at 3.484, the fp32 at 0.702) and free-running ≥ E17's `RANKS` band |
| **what a null means** | if gradients cannot move the *full body* the way they moved `q/o`, then post-hoc conversion is not merely lossy but **unrepairable at this budget**, and the donor route to a fast trained model is closed — which is a finding worth 5.6 GPU-h on its own |

**Not requested here.** H1 is in flight and its two sessions are the standing ask.

### 13.3 The 10 B clause, priced — **desk model, explicitly marked**

The goal is *"un modello grande (es 10B) a 50 tok/s"*. E55 showed the **engine** does that shape
at 49.57–118.47 tok/s. E57 showed the **weights** are the problem. So: what would a 10 B trained
into this format cost, and can the offered T4 weeks buy it?

**They cannot, and the blocker is memory, not hours.** A T4 is **16 GB**. A 10 B model in fp16
masters plus AdamW states is ~**120 GB** before activations; even fp16 weights alone are 20 GB.
**No 10 B trains on a T4 at any number of hours**, with or without the 30 GPU-h/week ceiling.
The hour arithmetic is therefore not the binding constraint and is not offered as one.

Two consequences, and they should be said plainly rather than deferred:

* **The literal 10 B clause is out of reach of the hardware available to this project**, by a
  factor that no scheduling fixes. What the T4 weeks *can* buy is the **1.5 B demonstration**
  (13.2) and the scaling statement that goes with it.
* **The honest deliverable is therefore: a trained LLM, faithful, fast, at 1.5 B, with the 10 B
  requirement stated as a measured budget rather than a demonstration.** E18 already gives that
  budget — ≤0.98–1.06 G active ternary weights/token — and E39 gives the shape that fits inside
  it. What is missing is not an argument; it is 120 GB of accelerator.

This is written here, in the document that asks for the GPU hours, because **the person being
asked should know what the hours cannot buy before deciding to spend them.**

### 13.4 APPENDED 2026-09-13 after E58 — the 1.5 B demonstration has a ceiling, and it is below the bar

§13.2 offers `H2T` on the grounds that a healed 1.5 B "lands at 1.5–1.9× the fp32 arm's 19.09
with a real quality number attached." **That is still true and the arm is unchanged.** What E58
adds is the number the *other* end of it stops at, and it is not the one E57 published.

E57 addendum D said a healed 1.5 B at `15b_tqh`'s bytes would read **47.71 tok/s — 95% of the
good bar**. **Withdrawn** (`probes/E58_WHERE_DID_THE_1_75x_GO.md` §5, ledger §57.1): that divided
a packed-kernel rate by the *fp32* streamed floor, a bound §23.3 had withdrawn for the packed
path on 2026-09-07. E58 opens the 1.5 B artifact's organs and measures the real ceiling:

| | measured today | ceiling, every organ at E10's best kernel cell |
|---|---|---|
| `15b_tqh` | **29.51 tok/s** | **33.54 tok/s** — **67% of the 50 tok/s bar** |

**So the honest offer to the person spending the hours is:**

* `H2T` buys a **trained, healed, faithful-enough 1.5 B at ~29.5–33.5 tok/s**. That is the first
  point in this programme that is trained *and* timed *and* quality-scored, and it is worth
  having. **It is not 50 tok/s and no amount of engine work makes it 50 tok/s** — the kernel has
  ×1.14 left, measured, and E10 already said that on 2026-09-07.
* Reaching 50 tok/s **even at 1.5 B** needs **33.7% fewer streamed weights** — a **45% FFN cut** —
  on top of healing. E19 measured a 52% FFN carve and E22 assembled a budget-feasible object at
  0.9441 G that **stops working when ternarised**. So the 50 tok/s bar is a *joint* quality +
  sparsity problem at every scale, not only at 10 B.
* §13.3's conclusion is **unchanged and now better supported**: the deliverable the T4 weeks can
  buy is the 1.5 B demonstration plus a measured budget, and E58 reproduces that budget
  (1.024 G active ternary weights/token) from the engine side, inside E18 §31's 0.982–1.060 G
  derived from the artifact side.

**Still not requested. H1 remains the only open ask.**

### 13.5 CORRECTION to 13.4, same evening — the ~33.5 tok/s ceiling is the DEFAULT kernel's

§13.4 wrote *"a healed 1.5 B tops out at ~33.5 tok/s — 67% of the bar"* and *"the kernel has ×1.14
left, measured."* **Both are true of the packed default and false of the binary.**
`donor_engine_e53.exe` also carries **`--lutblk`**, measured **end-to-end** by E13 §5 at **×1.217**
(0.5 B, `--bench 300`, 7 interleaved reps) and **×1.358** (Coder-7B, `--bench 100`). Transferring
those ratios to E58's measured rates, `15b_tqh` reads **35.9–40.1 tok/s = 72–80% of the good bar**.

**What that does and does not change for the hours being asked for:**

* **`H2T` is unchanged and still the arm.** Healing is a quality operation; which kernel serves the
  result afterwards is a separate decision made after the fact.
* **The demonstration's honest range widens to ~29.5–40 tok/s**, depending on a kernel choice that
  is **not free**: the LUT path quantises activations to int8 (E11: 1.40e-01 rel-L2 whole-vector,
  3.10e-02 at `--lut-group 32`), and E14 measured **greedy agreement 45.6% / 64.4%** against fp32
  activations while BPB *improved* by 0.017. **A healed model served on that kernel may lose the
  fidelity the healing bought.** E13 §8 item 2 has owed this measurement since it was written;
  **E59** is registered to take it, on the trained artifacts, before any of it is quoted.
* **The 50 tok/s conclusion is unchanged**: even at ×1.358 a healed 1.5 B is short, and closing the
  gap still needs **33.7% fewer streamed weights** (a 45% FFN cut). That is quality work, not
  engine work, and it is the same answer §13.4 gave.

**Still not requested. H1 remains the only open ask.**

### 13.6 APPENDED 2026-09-13 after E59 — `H2T` should heal against the ACTIVATION quantiser too, and it costs no extra hours

**This is the only change E59 makes to the ask, and it changes what the same sessions optimise,
not how many there are.**

§13.5 said the demonstration's honest range widens to ~29.5–40 tok/s "depending on a kernel choice
that is not free", and owed the measurement. `probes/E59_THE_FAST_KERNEL_DOES_NOT_SURVIVE.md` took
it, on the trained artifacts:

| `qwen25-15b_tqh` | speed | greedy agreement with the packed path, same bytes |
|---|---|---|
| `PACKED` | 28.87 tok/s | — |
| `--lutblk` | **38.48** (×1.333, intervals separated) | **55/160 = 34.4%** |
| `--lutblk --lut-group 32` | 38.38 (×1.329) | **92/160 = 57.5%** |

**So the ×1.33 is real and it is currently unusable on anything that works.** A model healed to be
right would be served by a kernel that changes 42–66% of its tokens.

**But every one of those numbers is int8 activations applied POST HOC to a model that never saw
them — and post-hoc is the exact thing this programme keeps measuring as fatal and training as
survivable.** Ternary weights post-hoc: 3/160 (E1/E17/E57). The same organs *trained*: H0 removed
**97.9%** of the damage in half a schedule. E18 §8: *"the model must be trained into the format
rather than converted into it."* **The activation quantiser is a format, and nothing has ever been
trained into it.**

**The change to `H2T`:**

> Heal the ternary body at 1.5 B **with a fake-quant of the layer inputs in the forward pass**,
> matching what `--lut` actually does: per-group amax to int8 at `AQ = 63`, **group 32**
> (`donor_engine.c:1531` requires an even group; `--lut-group 32` is the arm E59 measured as the
> Pareto point). STE through it, exactly as `s1/h0_qat.py` already does through the weight
> quantizer.

| | |
|---|---|
| **cost** | **zero extra GPU hours** — same 2 × 2.8 h, same data, same optimiser; one extra fake-quant in the forward |
| **engineering** | small: the STE machinery exists; the activation quantiser is a per-group amax, and E59 fixes its parameters (`AQ=63`, `G=32`) rather than leaving them to be chosen |
| **what it buys if it works** | the healed 1.5 B is served at the **measured 38.48 tok/s = 77% of the good bar**, not 28.87 = 58% |
| **what a null means** | the activation quantiser is *not* learnable at this budget, the LUT lever is closed for faithful models for good, and E58's packed ceiling is final. **That is worth knowing and costs nothing to find out** |
| **risk to `H2T` itself** | it makes the healing problem harder (two quantisers instead of one). **Mitigation: the run reports BPB and free-running for the weight-only objective as well**, so a `H2T` that succeeds on weights and fails on activations is still a success on its own terms |

**Still not requested. H1 remains the only open ask**, and this is written down now so the next
one is specified before it is made rather than after.
