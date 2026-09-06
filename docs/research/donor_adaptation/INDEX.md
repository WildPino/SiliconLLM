# Donor Adaptation — what has been tried, what it cost, where to look

**The goal:** run somebody else's pretrained LLM on our architecture (`engine.c`), target **~10B at
50 tok/s** (good) / **100 tok/s** (excellent).
**Last updated: 2026-09-06 (E3 closed, `RESERVATION-BREAKS`: the engine is measured at the target shape for the first time — 3.090 tok/s on a 10.6 B, and the non-weight cost is 9–22× the reservation the budget was built on. §7's twelve-donor screen is superseded).**

This is the map. Every row names the artefact that holds the detail; nothing here is a claim that
is not written up somewhere with its controls and its pre-registration.

---

## 0. Where the goal actually stands

| | status |
|---|---|
| **A pretrained donor executes on our runtime** | ✅ **YES** — Qwen2.5-0.5B, parity vs PyTorch `rel l2 2.8e-06`, top-1 `1.0000`; and since E1 the engine **scores the same BPB as PyTorch to `1.5e-05`** on both donors, so the quality numbers below are statements about the deliverable, not about a simulation |
| **At the target speed** | ❌ **measured at the target shape now, not extrapolated: 3.090 tok/s** (E3 `T10`, 10.6 B active, `--bench 300`, 3 reps, idle) — **15.7× short of 50**. The delivered rate is *better* than the ledger (32.8 vs 27.7 G-w/s, +18%); what breaks is the **non-weight** term: `f` = rope+attention+norm is **10.576 ms/token at 300 context and 26.088 at 800** against a **1.17 ms** reservation. **At 800 context a 10 B shape cannot pass 38.3 tok/s even with a free weight path** |
| **At usable quality** | ❌ **NO**, but the number keeps moving: FFN conversion **+3.309 → +1.260 BPB** (T2), still 252 σ_seed. The **whole runnable model** cost **+2.708111** (T2b) and is now **+2.465779** — E2 confirmed the RMSNorm fold *through the engine* at T3's exact `−0.220001` and it is **adopted as the exporter default** |
| **The binding constraint** | **still quality — but it is the RULE, not the format** (T2, `RULE-HELPS`) |

> ⚠ **Until E1 (`33f0add`) the runtime could not load a model over 2 GB at all** — 32-bit
> `ftell`, silently reported as `bad magic` against an intact file. The largest artifact it had
> ever been given was 1.84 GB. **A 10B ternary packed model is ~5 GB**, so the target was not
> slow, it was unloadable, and no speed probe could have found it.

**The one-line state:** the road exists end to end — safetensors → export → ternary runtime →
generated tokens, with a parity gate at the seam. T2 has now shown the damage at the far end was
**62% a bad map into the format**, not the format itself, and removed that much of it on CPU with
no gradients. What remains is a real quality gap and a 20× speed gap.

**The number that prices everything else, re-derived by E3 at the shape it is about:** a 10B donor
needs **≤318 M active weights/token — 3.0% of a 10 B — for 50 tok/s at 300 tokens of context**, and
**at 800 tokens of context there is no budget at all**, because `f` alone exceeds the whole 20 ms.
100 tok/s is **out of reach at any weight cost** at a 10 B shape: `f` = 10.576 ms > 10.

The old figure, **522 M**, is superseded twice over. It was also **arithmetically wrong in its own
terms**: `27.7 G-w/s` is `493,961,216 / 17.833 ms` = weights / **wall**, so `f` is already inside the
denominator, and `× (20 − 1.17)` charges it a second time; self-consistently it is 554 M.

**The head is no longer the floor.** It is 20.5% of the token at 0.5 B and **1.1% at a 10 B shape**;
the attention projections are 17.9%, sixteen times more. The tokenizer claim itself survives and is
now end-to-end rather than a weight count — `M7` and `Q8`, same width, **4.637× the vocabulary and
4.629× the head time, 0.2%** (§7).

## 1. The runtime (this is the deliverable)

| what | where |
|---|---|
| Export a donor to a flat binary (fp32 / ternary / packed; optional ternary head) | `benchmarks/donor_adaptation/engine/qwen_export.py` |
| The runtime: RMSNorm, GQA + RoPE + KV cache, SwiGLU, ternary matvec, packed `pshufb` path | `benchmarks/donor_adaptation/engine/donor_engine.c` |
| Parity gate vs PyTorch on identical weights | `benchmarks/donor_adaptation/engine/parity_gate.py` |
| Report: profile, the two optimisations, honest position vs the goal | `probes/R1_DONOR_RUNTIME.md` |

**Measured, Qwen2.5-0.5B, 3600X, `--bench 300`, idle, median of 3:** 12.45 tok/s (t1) →
**56.1 tok/s (t6, packed)**. Trajectory: 23.5 (fp32 head) → 38.0 (ternary head) → 48.3 (packed)
→ 50.9 (same binary, re-measured 2026-09-05) → **56.1 (rope hoisted out of the head/layer loops,
bit-identical, `SPEED_LEDGER.md` §12)**. `--fuse --lut --lut-group 32` adds ~3.9% on top, but the
LUT half of that is not numerically free.

> **Two rates have been withdrawn here for two different reasons, and both were load-bearing.**
> R1's "40–46" and §10's 36.1 were taken on a **contended machine**. §11.4's per-organ table was
> taken with **the wrong work inside the timer** — both `rope()` calls sat in the `qkv` bucket, so
> it reported qkv at 4.1 GB/s (real: 14.2) and produced a "32 µs per call" anomaly that does not
> exist. **A tok/s figure is also only comparable at the same `--bench` length**: attention is
> `O(position)`, worth 1.7 ms/token between 300 and 800.

## 2. The speed side — what is priced and what is measured

| probe | question | answer | where |
|---|---|---|---|
| **Ledger** | what must a donor cost per token to hit 50/100 tok/s? | **≤522 M active weights/token** for 50 tok/s, ≤245 M for 100 — from the *measured* **27.7 G-weights/s**, not a byte-rate ÷ bits-per-weight | `SPEED_LEDGER.md` §12.2 |
| **P2** | is the expert path bandwidth- or compute-bound? | **MIXED — ~60% arithmetic.** A denser pack buys ≤1.32× on the FFN, not 2.5× | `probes/P2_EXPERT_PATH_DECOMPOSITION.md` |
| **P3** | what do donor SHAPES cost on the engine's kernels? | 0.5B 81 / 1.5B 30.7 / 3B 15.1 tok/s (matvec only). The ledger was 18% conservative because it priced dense FFNs at a *gather* rate | `probes/P3_DONOR_SHAPE_ON_ENGINE.md` |
| **R1** | what does a real runtime cost? | **56.1 tok/s** at 0.5B (§12). The packing bought nothing, exactly as P2 predicted; the LUT buys 2.2% and is not free; **the one big win was a profiling artefact — `rope()` was 9.6% of every token** | `probes/R1_DONOR_RUNTIME.md`, `SPEED_LEDGER.md` §12 |
| **E3** | what does the engine actually do at the target shape, end to end? | **`RESERVATION-BREAKS`.** 6 shapes 0.5–10.6 B x 2 context lengths, synthetic weights gated against real artifacts at 0.000% (1.5 B) and 0.357% (0.5 B). `T10` = **3.090 tok/s**; rate **rises** to 32.8 G-w/s; **`f` is 9.0× the reservation at 300 context and 22.3× at 800** | `probes/E3_ENGINE_AT_TARGET_SCALE.md` |
| **P64 matrix** | is the runtime's rate set by OpenMP region count or thread wake-ups? | **neither.** A region costs **2.5–3.2 µs**, measured two ways; `OMP_WAIT_POLICY=active` does nothing. After the rope hoist all four organs sit in a **1.3× band** | `SPEED_LEDGER.md` §12.4, `engine/bench_matrix.py` |

**Terms nobody had attacked before the ledger, now priced:** the **output head** (was 40.4% of every
token in fp32 — fixed; and E3 shows it is **not** the floor at 10 B, §7), the **KV cache** (still
fp32, untouched — and E3 says this is now the binding term), the **attention projections** (17.9% of
a 10 B token, `qkv` + `o_proj`).

> **E3 changed which term this section is about.** The weight path is *faster* than the ledger says
> at every shape above 0.5 B. What does not scale is `f`. E3 §4.6 measures the attention loop at
> **one FMA per ~4 cycles per thread across twelve points, ±7%**, invariant to KV size (16× range),
> GQA factor and vocabulary — the signature of a serial FP reduction, not a bandwidth wall:
> `d += qh[i]*kt[i]` compiled without `-ffast-math` cannot be reassociated or vectorised, while
> every matvec is hand-written AVX2 with an 8-wide accumulator. **The single highest-value engine
> experiment now open is giving that loop independent accumulators.**

### 2.1 The LUT kernel — `donor_engine.c --lut`, the lever P2 and R1 named

| what | status |
|---|---|
| tile-major layout, `pshufb` tables, grouped scales | **built.** A `--lut` model and a `--quant packed` model hold **bit-identical weights** — the copy is a pure transpose |
| kernel correctness | **bit-exact** vs a scalar-integer reference; two planted controls fire (`--selftest-lut` cases B and C) |
| numeric cost of the int8 activations it requires | **measured.** rel l2 `1.40e-01` per-vector → **`3.10e-02` at G=32 channels per scale** |
| why it costs that | activation crest factor `amax/rms` is 8.3 avg / 69.6 max, so a 63-step grid leaves ~4–8 usable levels. `--lut-diag` measures it on `x` alone, so the kernel is not implicated |
| **rate** | ⚠ **measured: 1.022×** after the rope hoist (`SPEED_LEDGER.md` §12.4). It helps the two big-matrix organs (ffn −0.40 ms, head −0.33) and **hurts the two small ones** (qkv +0.31, o_proj +0.06); `--fuse` first removes that penalty and the pair is worth 1.039×. **Not the 1.83× the ledger assumed** |
| BPB through the runtime | ❌ not run |

Two predictions were written before their sweeps and **both were wrong in magnitude**: clipping the
grid to `k·rms` (predicted an interior optimum; it is 3× worse at every `k`) and per-group scales
(predicted <0.02 at G=128; it is 0.052). Recorded in commits `95b7fd3` and `e02285c`.

## 3. The quality side — every structural result, and the conversion that undercuts them all

| probe | what was tried | outcome | where |
|---|---|---|---|
| **D0 / D0c** | carve the FFN into co-activation experts | **+1.09 BPB at E=32; +0.706 at the finest legal E.** Granularity helps, but the random null helps *more* → the mechanism does not strengthen (PARTIAL) | `probes/D0_COACTIVATION.md`, `probes/D0C_GRANULARITY.md` |
| **D1** | magnitude / structured pruning | see report | `results/d1_pruning.json` |
| **D2 / D3** | basis rotation, low-rank | see reports | `results/d2_basis.json` |
| **D4** | solve for thin replacement weights (Hessian) | recovery 0.483 honest vs 0.859 leaked → in-sample optimism; budget never swept | `probes/D4_RECONSTRUCTION.md` |
| **S1** | which bar predicts BPB under sparsity | \|h\| is the bar; A≡D at every digit | `benchmarks/donor_adaptation/s1/` |
| **T1** | **ternarize the donor — the engine's own rule** | **+4.738 BPB = 948 σ_seed. CONVERSION-FAILS** (measurement stands; verdict superseded in scope by T2) | `probes/T1_DONOR_TERNARIZATION.md` |
| **T2** | was that the FORMAT or one naive RULE? | **`RULE-HELPS`. It was the RULE.** FFN +3.309 → **+1.260**, 62% removed with no training. **BitLinear158 is statistically indistinguishable from RANDOM SIGNS** (−0.064 ± 0.126) | `probes/T2_TERNARIZATION_RULE.md` |
| **T2b** | does the winning rule survive outside the FFN? | **`UNIFORM`, by 1.0% of its bar.** The runnable model (197 tensors, R3) costs **+2.708**, `1.584×` the FFN alone vs a 1.60 bar — ci95 `[1.514, 1.649]`, the bar is INSIDE it. Head ternarization **not** withdrawn: the head costs `+0.339` alone and **`−0.009 ± 0.020` on top of a ternary FFN+attention**. Per weight **attention is 4.98× the FFN** | `probes/T2B_ORGAN_COVERAGE.md` |
| **T3** | rotate the residual basis (QuaRot/SpinQuant) before ternarizing | **`VOID` as written, `NULL` once the brief's own gate constant is corrected.** Rotation does not fail to help, it **hurts**: `+0.634 ± 0.039` (dense orth) and `+1.138 ± 0.072` (Hadamard) vs a fold-matched control. Kurtosis is not a predictor and cannot be made one. **Keeper: the RMSNorm fold alone is `−0.220 ± 0.053` free** | `probes/T3_ROTATION.md` |
| **E1** | does the model the ENGINE executes score the BPB PyTorch says it does? | **`LOOP-CLOSED`** on the 0.5B; every term of the same rule met on the 1.5B (labels there are `INCOMPLETE` because the brief split the arms across two runs). Largest disagreement over 5 arms and 2 donors **`+1.53e-05` BPB** = `0.003 sigma_seed`, and the delta **does not grow with the arm**. Gate A bit-identical on up to **1,543,569,408 codes**. T2b's `+2.708111` replicated **bit-identically** by a second runner through a file round-trip. **Found on the way: the engine could not load a model over 2 GB** | `probes/E1_BPB_THROUGH_ENGINE.md` |
| **E2** | does T3's RMSNorm fold survive into the artifact the engine runs? | **`FOLD-CONFIRMED`, and adopted.** `NL - TQ = -0.220001`, paired SE `0.052861`, ci95 `[-0.324988, -0.119593]`, 44 sigma_seed on the 1.5B -- **T3's point AND its dispersion**, now through the exporter, in the file format, on weights Gate A proves identical. The **runnable** model goes `+2.708111` -> **`+2.465779`**. Gate F passes on the deciding donor with 4.2e+04x of margin. **`--fold all` is NOT adopted**: folding the final gain into a *ternary* head COSTS `+0.176983` (1.5B) / `+0.629949` (0.5B). **Every sign replicates across donors; no magnitude does, and one comparison inverts** | `probes/E2_RMSNORM_FOLD.md` |

**T2's decomposition, paired between arms** (`probes/T2_TERNARIZATION_RULE.md` §4):

| what | Δ BPB | significant |
|---|---|---|
| searching the per-row scale at all | −0.686 | yes |
| **weighting that search by activation RMS** | **−0.914** | yes |
| GPTQ error compensation **on a well-placed grid** | −0.449 | yes |
| GPTQ error compensation **on the naive grid** | +0.223 | **no** |
| TWN instead of BitLinear158 | −0.225 | **no** |

> ⚠ **Every structural result above was measured on fp32 weights the engine cannot consume**, and
> none was ever composed with the conversion. "Does carving a *ternarized* donor cost the same as
> carving an fp32 one?" has never been asked — and it is now a cheaper question than it was, since
> the conversion it would have to compose with costs +1.260 rather than +3.309.

## 4. Open, in priority order

0. **The attention accumulator** — new, and it is now first. E3 §4.6 measured the attention loop at
   **one FMA per ~4 cycles per thread across twelve points (±7%)**, invariant to KV size, GQA and
   vocabulary: the signature of a serial FP reduction, and `d += qh[i]*kt[i]` built without
   `-ffast-math` cannot be vectorised or reassociated. Give it 4–8 independent accumulators. **If
   `f` drops by roughly the accumulator count it was latency; if it does not move, the lever is an
   int8 KV cache instead** (the cache is fp32 and has never been touched). This is the only open
   item that can move `f`, and until `f` moves there is **no active-weight budget at all at 800
   tokens of context** (§0, §7). It is engine-only: no format change, no quality question, no
   retraining. **Parity gate mandatory** — accumulation order is exactly what E1 traced its
   `1.5e-05` engine/PyTorch delta to.

1. **D4b** — the calibration budget. Promoted from bookkeeping: T2's two best arms are both
   calibration-driven, so every one of their numbers is a **floor** — and after E2 that now
   includes the fold's own `−0.220001`, which was measured at 32 calibration sequences like
   everything else.
2. **Healing** (QAT / layer-wise distillation) — still on the critical path per T2 §7. **Its
   starting point moved**: the whole runnable model is now **`+2.465779`** (E2 arm `NLH`), not
   `+2.708111`. T2b §6 says where to aim it: **attention, 4.98× the FFN's damage per weight at 10%
   of the weights**. Note that T2's GPTQ result — at ternary width the per-layer error is the wrong
   objective on a badly-placed grid — constrains what "layer-wise" is allowed to mean here.
3. **Why the same fold buys on the layers and costs on the head** (E2 §5.4). `q,k,v,gate,up` gain
   `−0.220`; `lm_head` loses `+0.177`. E2 killed the obvious explanation — zero fraction does not
   predict it and does not even hold its sign — and put nothing in its place. The per-row structure
   of R3's threshold search against the per-column structure of a gain fold has not been measured.
   Cheap, exporter-only, and it is the one place the fold left value on the table.
4. **S1's scale arm** — blocked on the fp16 NaN (`eager` attention overflows QK^T; diagnosed, §5).
   Every sparsity result this programme owns is measured at one size.
5. An already-MoE donor, and **a donor with a small vocabulary** (§7). E3 sharpens both: a small
   vocabulary buys much less than §7 used to imply (Mistral-7B is still at **141% of budget at 800
   context** with the smallest vocabulary on the disk), and **nine of eighteen donors — every MoE
   and every hybrid — have no measured `f` at all**, so their screen rows are withdrawn, not
   restated.
6. **`f` beyond 800 tokens of context.** It grows with position and nothing measured bounds it. The
   two lengths E3 ran were chosen before the result; the shape of the growth past 800 is unknown.

**Closed since the last revision.** The `--fuse` × `OMP_WAIT_POLICY` matrix ran (`SPEED_LEDGER.md`
§12.4 — both hypotheses die; `--fuse` not adopted). T3 ran and closes the residual-stream rotation
line together with brief §5's two follow-ons (online Hadamard, activation-side rotation).
**E1 ran** (`probes/E1_BPB_THROUGH_ENGINE.md`): the engine scores what PyTorch scores to `1.5e-05`,
T2b's `+2.708111` is replicated bit-identically through a file round-trip, and the 2 GB load ceiling
that would have blocked the target model outright was found and fixed.
**E2 ran and closes the standing item 1** (`probes/E2_RMSNORM_FOLD.md`): the fold survives to the
artifact at T3's exact value **and its exact dispersion**, Gate F passes on the deciding donor with
4.2e+04× of margin, and it is **adopted** — `qwen_export.py --fold` defaults to `layers` as of
`49b6654`, with `--fold none` pinned in E1's runner in the same commit so its published numbers keep
reproducing (both directions checked by sha256 against artifacts already on disk). `--fold all` is
measured and **rejected**: into a ternary head the final gain costs BPB on both donors.

## 5. Bugs found in our own instruments (all fixed, all with controls added)

| bug | how it presented | fix |
|---|---|---|
| **The A1.2 gate reported PASS over an all-NaN run** | `max(0.0, nan) == 0.0` in Python swallowed all 51 comparisons | hard-fail on any non-finite, minimum comparison count, and **7 planted self-test cases run before any model loads** — the old code fails 3 of them |
| **`l1_keep_count` turned a NaN into a plausible measurement** | `clamp_(1, F)` mapped a NaN row to "keep 1 neuron" → achieved sparsity `8959/8960` | raises `FloatingPointError` on non-finite input |
| **fp16 NaN blamed on the GPU** | my own diagnostic did not pass `attn_implementation` and tested SDPA, not the `eager` path the probe uses | reproduced on CPU with `--attn eager`; cause is HF eager computing QK^T in fp16 (**274,672 vs the 65,504 limit**) before dividing by √head_dim |
| **T1's planted control was mis-specified** | required random signs ≫ ternarization; they are only 1.21× apart, so it returned VOID on sound numbers | identity-substitution control added (bit-exact). **T2 §4 then showed the premise itself was false**: random signs are not ≫ the treatment, they are indistinguishable from it |
| **`--calib-seqs` defaulted to 8 while T2 measures at 32** | exporting `--rule R3` would have built a model on a quarter of the calibration budget that produced the number, and the BPB gap would have read as the runtime disagreeing with PyTorch | default → 32, stderr warning otherwise, and the sidecar records the budget and the organ lists |
| **the LUT diagnostic reported whole-vector crest while groups were active** | it kept calling the derived figure "effective levels" when the grid was per-group, i.e. a plausible number describing the wrong thing | crest computed in-group, both labels corrected, header states which scale is in force |
| **both `rope()` calls sat inside the `qkv` timer** | the per-organ table read qkv at **4.1 GB/s** against the head's 17.6 — a 4× anomaly that does not exist — and two experiments were built to chase it. It also hid that rope was **9.6% of every token** | `T_ROPE` is its own bucket; rope hoisted out of the head and layer loops (**+10.3%, bit-identical**); §11.4 marked superseded and its two derived claims withdrawn in §12.1 |
| **the bench harness called an idle machine CONTENDED, twice** | once because it compared an 800-token run to a 300-token reference (attention is `O(position)`), once because min-max over 6 rounds is set by a single bad round | reference must match `--bench` length; the gate is now the IQR; the harness prints which of its two blocks is readable |
| **a pre-registration contradicted itself and the gate fired on the contradiction** | T3 returned `VOID`: brief §3 fixed the organ set at FFN+attention (196 tensors), brief §4 pinned the replication constant to T2's **FFN-only** Δ (`+1.709372`, 84 tensors). The runner obeyed both halves and arm Q missed by 1.007 | the constant was checkable and was checked: arm Q reproduces T2b's arm FA **bit-identically** from a different runner. **A replication constant must name the arm, the organ set and the file it came from**, so a mismatch with the arms section is visible on the page |
| **`--quant ternary` had been dead since `--rule` landed** | `w_tern` squeezed a scale that `quantize()` already returns as a `[out]` vector → `IndexError`. Nothing caught it because every artifact this programme built used `--quant packed`, which does not go through that line | the second squeeze removed; verified empirically before the fix, not assumed |
| **two exports of the same command produced different files** | R3's calibration forward changes its reduction order with torch's thread count: 6 vs 1 threads moved **102,123** `act_rms` elements (worst `1.9e-06`), so the sidecar recorded a sha256 it could not reproduce | `--threads`, recorded in the sidecar. **The thread count is part of the artifact's identity, not a speed knob** |
| **the exporter was not loading the model the probes measured** | it omitted `attn_implementation`, so HF gave it **sdpa** while `common.load_model` — and therefore T1, T2, T2b, T3 — uses **eager**. E1's Gate A fired at `2.980e-08`, exactly one ulp. Measured: `act_rms` differs on **142,977** elements (worst rel `8.2e-06`), moving **132,844** stored scales by up to **6 ulp**. All **357,826,560 codes were identical throughout** — the artifact's *identity* was wrong, not its content | `eager` pinned in the exporter and recorded in the sidecar. Gate A then passed **as pre-registered**: 0 scales differing, 0.00 ulp. Same law as the fp16 row, in a second place: **reproduce the configuration, not just the model** |
| **the best quality claim had no artifact** | `qwen05b_packed.bin`'s sidecar has no `rule` field — it predates `--rule`, so every speed number was taken on **R0/BitLinear158** (zero fraction `0.327`), the worst of T2's five rules. **No R3 model had ever been exported.** The format is identical so the speed numbers stand | E1 exports R3 (zero fraction `0.4922`, consistent with T2b's `0.4714` at full budget). Found by reading the sidecar of the file on disk rather than the command that was supposed to have written it |
| **the engine could not load a model over 2 GB, and blamed the file** | E1's 1.5B fp32 arm died with `FATAL: bad magic -- not a QWENDON1 file` on a 6,174,857,268-byte export whose magic was intact. `long` is 32 bits on Windows even on x64, so `fseek(SEEK_END)` FAILS above 2 GB and `ftell` reports 0 → a zero-byte blob was allocated, zero bytes were read (**which equals the zero requested, so the `short read` guard passed**), and `memcmp` compared the magic against an empty buffer. The largest file the engine had ever been given was 1.84 GB, just under the ceiling. **A 10B ternary packed model is ~5 GB: the target was unloadable, and no speed probe could have found it** | 64-bit offsets chosen by platform (`_fseeki64`/`ftello`), a real error when the seek fails, 1 GB chunked reads. **Planted control before the rebuilt binary produced any number**: it re-scored the measured 0.5B TQ artifact at `NATS_TOTAL 162120.4241599279`, bit-identical. **Known-positive**: the 5.75 GB file then loaded and scored |
| **a sweep's `untied` field recorded the wrong thing, and arm state leaked** | `t2b_organs.json` says `"untied": false` on arm FAH, an arm that ternarizes a *tied* head — which would mean the embedding was ternarized too. It was not: E1's arm TQH reproduces FAH **bit-identically**, and two different models cannot. Arm `I` reports `untied: true` and every later arm `false`, because T2b's `restore()` restores weights but never re-ties | no number changed and none is withdrawn — arm `I` returns the base BPB exactly. Recorded because it was caught by a **replication**, not by the sweep: **the field means "did this arm untie", not "is the head untied here"**, and state crossed arm boundaries |
| **a result file was named after the model alone** | E1's pre-registered 1.5B fp32 **subset** run was about to overwrite the 55-minute TQ/TQH JSON written under the same name | a subset run now carries its arms and sequence count in the filename; only the canonical run keeps the bare name |
| **a gate that was never evaluated reported that it had failed** | E2's run 2 carries only the five ternary arms, so Gate F -- which is built from `XF`/`XA` against `F32` -- would have found no arms to compare, left `gate_f` empty, and printed `VOID (Gate F failed: the fold is not exact where it must be)`. The run owning the **label** would have reported the fold inexact on the strength of nobody having looked, and the brief itself had pre-registered the split that causes it (§3.2) without foreseeing what the runner would print under it. The same brief also predicted run 1 would return `INCOMPLETE`; run 1 holds all eight arms and returns a real label | `gate_F_measured` separated from `gate_F_ok`; `VOID` is now reachable only from a gate that was measured and failed, and a run missing the gate prints the fold term plus `GATE-F-NOT-MEASURED-HERE` naming the run that owes it. Brief §3.3 written before run 2, changing no threshold or arm. **All three states planted and shown to fire** (`3aa1bef`), the failing one via a copy with `GATE_F_FP32_TOL = -1.0` -- without it the patch would only have been shown not to say `VOID`, and a guard that never fires looks identical to one that fires correctly |
| **a budget formula charged the same milliseconds twice** | `SPEED_LEDGER` §12.2 computes `Delivered: 493,961,216 / 17.833 ms = 27.7 G-weights/s`, where **17.833 ms is the wall** — so the 1.17 ms of non-weight work is already inside the denominator. It then prices the budget as `27.7 × (20 − 1.17) = 522 M`, subtracting it a second time. Self-consistently the figure is **554 M** (`r_wall × 20`) or **558 M** (`r_w × (20 − f)`), which agree to 0.7% | the error was ~6% and in the **safe** direction, which is why nothing caught it for two weeks. `e3_budget_by_shape.py` prints both conventions side by side and never mixes them. **A rate and a reservation must be read off the same denominator**, and the way to check is that the two self-consistent forms agree |
| **a planted control was planted outside the range the instrument is used in** | E3's Gate V1 compared an **all-zero-code** model to a no-zero one and fired at 15× the IQR, voiding the run. The kernel was innocent: `qkv_proj`, `o_proj` and `head` — pure packed matvecs — were flat to **0.005 ms**, and the whole 2.080 ms sat in `ffn`, the only organ holding a transcendental. An all-zero model feeds `expf` exactly `0.0f`, every libm's early-out; 2.080 ms / 116,736 calls = **17.8 ns per call**. A zero fraction of 1.0 is not something any exporter can produce, and across the range a real export occupies (0.47 vs 0.00) the timing is flat to 0.08 tok/s | the `VOID` was honoured and no arm was generated under it. The replacement gate compares synthetic against a **real exported artifact of the same shape**, fixed at a scale where no synthetic file existed yet — and passed at **0.000%** (1.5 B). The failed gate is reported next to the pass, not deleted. The accident is that it also **measured** what it was meant to assume: the matvec's value-independence is no longer a reading of the source |
| **a speed gate contained a hard constant, and its own ground truth could not pass it** | E3 brief §8.4 fixed Gate V2′ at *"57.790, IQR 0.190"*, measured on a real artifact twenty minutes earlier. The synthetic file gave 56.180 and failed. Re-run **interleaved A/B/A/B**, the *same real file* gave **56.095** — it had moved **2.9%** — while synthetic and real differed by **0.200 tok/s, 0.357%**, with one pair in which both dipped together | **fourth time this ledger has been bitten by the same family**: §11 a rate under contention, §12 a timer bracketing the wrong work, E3 §2.3 an anchor on the wrong head, now a constant that did not survive twenty minutes. The rule is now explicit: **a speed gate may not contain a constant; it must name a file to be measured concurrently, interleaved** |
| **a brief asked for one configuration in prose and the code read another from a config file** | E3 §3 asked for *"a ternary head (the runnable configuration E2 settled)"*. `synth_export.py`'s `SHAPES` carried each donor's own `tied` flag, and **a tied model runs its head as the fp32 embedding** — 544.6 MB/token, 13.8 ms, **52% of the token**. Six arms were not generated on the strength of a configuration nobody intends to ship. `SPEED_LEDGER` §12.2 had said its anchor was untied and packed without naming it: `136,134,656 weights, 68.1 MB/token` is **0.5 bytes per weight** | found because the synthetic file agreed with the **real** artifact of the same configuration to 0.36% while both missed the ledger by 35% — a disagreement that could only be about *which* configuration. `--head {ternary,donor}` added, default `ternary`. **A configuration named in prose must be named in a flag**, and a table's units are a claim about which configuration produced it |

## 6. Working rules this programme has paid for

- **A negative is only as strong as the sweep behind the setting it was measured at.** (D0c, D4b, T2)
- **A control must itself be shown to fire.** A gate never seen to trip is decoration. (§5, row 1)
- **Profile before optimising.** The head was 40% of every token and nobody had looked. (R1)
- **When reproducing a failure, reproduce the CONFIGURATION, not just the model.** A different
  library default invalidated a conclusion. (§5, row 3)
- **Byte-rate ÷ bytes-per-weight is only valid where the path is bandwidth-bound.** (P2 → R1)
- **A Δ against a baseline cannot support a claim about one arm versus another.** The arms are
  correlated across sequences; the contrast has to be bootstrapped paired. Doing it moved two of
  T2's apparent results — TWN and GPTQ-on-the-naive-grid — from "worse/better" to *not
  distinguishable*. (T2 §4)
- **When an instrument returns an impossible ordering, test the instrument before the finding.**
  GPTQ scoring below its own starting point is not physically possible; two controls showed the
  code was right and the objective was wrong. (T2 §5)
- **A profiler bucket is a claim about what is inside it.** `rope()` inside the `qkv` timer both
  hid a 9.6% cost and manufactured a 4× anomaly that two experiments were then aimed at. Before
  deriving a rate from a bucket, name every operation the bracket contains. (§12.1)
- **A throughput number carries its bench length.** Attention is `O(position)`, so a longer bench
  has a genuinely lower tok/s — 1.7 ms/token between 300 and 800 on this donor. Two rates taken at
  different lengths are not comparable and their difference is not contention. (§12.3)
- **Quantization damage does not add across organs.** The head costs `+0.339` alone and
  `−0.009 ± 0.020` on top of a ternary FFN+attention; the three single-organ arms sum to `+3.184`
  where the combination measures `+2.708`. A per-organ cost is only a cost *in the company it was
  measured in*. (T2b §4)
- **A smoke run establishes that the apparatus runs, not the sign of the effect.** T3's 4-of-28-layer
  smoke passed every control (X at `6.4e-07`, planted null clean) and flipped the sign of all three
  contrasts, returning the opposite label to the run. **A truncated model is not a small model.**
  (T3 §4.5)
- **A gate constant is a claim, and it is checkable against the arm that produced it.** T3's
  `VOID` was a mis-specified constant, not a broken harness — provable because an independent
  runner had already measured the identical arm to the last bit. Report the label the rule
  returns, then show the check; do not rewrite the rule. (T3 §1.3)
- **A pre-registered threshold needs its own interval before the label is read as settled.** T2b
  passed its 1.60 bar at 1.584 — but the ratio's ci95 is `[1.514, 1.649]` and a third of the
  bootstrap lands on the other side. The label stands; the confidence in it does not. (T2b §5)

- **A ceiling nothing has reached is a ceiling nobody has tested.** The runtime could not load a
  model above 2 GB; the largest one ever handed to it was 1.84 GB, so the limit had never fired,
  and the error it produced accused the artifact instead of the reader. The target model is ~5 GB.
  **Before scaling a number up, check that the apparatus can hold the object the number is about.**
  (E1 §4.4)
- **An error message is a hypothesis, not a diagnosis.** `bad magic` was reported about a file
  whose magic was intact, by code that had allocated a zero-byte buffer and passed its own
  short-read guard because zero bytes read equals zero bytes requested. **A guard that compares a
  quantity to itself checks nothing.** (E1 §4.4)
- **A pre-registration that splits a decision across runs must say which run owns the label.**
  E1's 1.5B arms were split across two invocations by the brief, so neither could evaluate the
  decision function and both returned `INCOMPLETE` on numbers that met every term. Second time in
  three probes that the brief, not the code, produced the mechanical label. (E1 §6.2, T3 §1.3)
- **A small donor establishes the sign, not the magnitude — and not always the ordering.** Across
  E2's two donors every sign that mattered replicated and **no magnitude did**: the fold's benefit
  differed 1.5×, the head fold's cost 3.6×, the benefit a ternary head adds 8.9×, and one whole
  comparison **inverted** (`--fold all` is worse than no fold on the 0.5B, better on the 1.5B).
  Same shape as E1's "the head is free is a 1.5B result". **Give the deciding donor the label in
  writing before the small one produces a number.** (E2 §5.0, §3.2 of its brief)
- **A replication that shares the estimator's seed reproduces its dispersion too, and that is not
  extra evidence.** E2 recovered T3's fold as `−0.220001 ± 0.052861` — point *and* paired SE at the
  printed precision. Forced, not corroborating: same seed 7, same 24 sequences, same byte weights,
  and per-sequence nats agreeing to `1e-07`. **Count it once.** (E2 §4)
- **`ci95 excludes 0` is a statement about accumulation when the effect is at the ulp.** The same
  contrast, same arms, same code, **excluded** zero on one donor (`+2.037e-08`) and **contained**
  it on the other (`+3.827e-08`). Exclusion is not magnitude; a rule must pair it with a threshold.
  (E2 §5.2)
- **A default that changes behaviour is only safe when both sides are pinned to something already
  on disk.** Adopting the fold flipped `--fold` to `layers`, which would have silently re-exported
  every caller that omits the flag — including E1, whose reference builder still defaults to
  unfolded. The flip and the pin went in one commit, and both directions were checked by **sha256
  against artifacts that had actually been measured**. (E2, `49b6654`)
- **A planted control must be planted inside the range the instrument will actually be used in.**
  An all-zero model is not a model any exporter can produce; the gate that used one measured `expf`
  instead of the kernel it was defending, and voided a run for it. (E3 §2.2)
- **A speed gate may not contain a constant.** It must name a file to be measured **interleaved**
  with the thing under test. E3's reference artifact moved 2.9% in twenty minutes and could not pass
  the gate its own measurement had defined. Fourth instance of the same family in this ledger.
  (E3 §2.5)
- **A rate and a reservation must be read off the same denominator.** `27.7 G-w/s` is weights/**wall**,
  so pricing `27.7 × (20 − f)` charges `f` twice. The check that catches it is cheap: the two
  self-consistent forms — `r_wall × 20` and `r_w × (20 − f)` — must agree, and they do (554 vs 558 M).
  (E3 §4.3)
- **A configuration named in prose must be named in a flag.** E3's brief said "ternary head" and the
  code read `tied` out of a donor config; the two differ by 52% of the token. (E3 §2.3)
- **Extrapolating a fit requires publishing its residuals first.** `f` is fitted per donor shape only
  because the fit reproduces the twelve measured points to 7.7% worst and **0.0–1.2% at the shapes
  that decide**; the residual table is the licence, and it is printed above the extrapolation.
  (E3, `speed/e3_budget_by_shape.py`)
- **A shape can be measured without weights, but only if the instrument is gated against a real
  artifact.** E3's synthetic files match real exports **byte-for-byte in size** and to **0.000% in
  time** at 1.5 B — which is what buys the right to measure shapes nobody has 20 GB to download.
  (E3 §2.1, §2.4)

## 7. The head, the tokenizer, and the thing nobody priced

> ### ⚠ SUPERSEDED BY E3 (`probes/E3_ENGINE_AT_TARGET_SCALE.md`)
>
> The table below screens twelve donors against **522 M**. That constant is wrong twice: it charges
> the fixed cost twice (§0), and — far larger — it uses a **1.17 ms** non-weight reservation that E3
> measured at **10.576 ms** at a 10 B shape. The screen is re-run below from measured `f`, per shape.
> `donor_speed_budget.py`, which produced the old table, says of itself that *"every attention row
> below is priced by ANALOGY to the proj-GEMV path, and that analogy is the single largest source of
> error here."* E3 measured it, so the analogy is retired.
>
> **The re-run.** `speed/e3_budget_by_shape.py` fits `f = A·L·NH·HD·pos + B·L·D` on E3's twelve
> measured points — worst residual **7.7%**, and **0.0–1.2% at the 7–10 B shapes that decide** — then
> applies it per donor. `budget = r_w × (20 ms − f)`, `r_w` measured per size class.
>
> | donor | D | V | head | fitted `f` @300 | **budget** | **head as % of budget** | was (§7) |
> |---|---|---|---|---|---|---|---|
> | Qwen3-8B | 4096 | 151,936 | 622 M | 7.928 | 409 M | **152%** | 112% |
> | OLMo-2-7B | 4096 | 100,352 | 411 M | 7.047 | 438 M | **94%** | — |
> | Mistral-7B-v0.3 | 4096 | 32,768 | 134 M | 7.047 | 438 M | **31%** | 24% |
> | Phi-3-mini | 3072 | 32,064 | 99 M | 5.286 | 468 M | **21%** | 18% |
> | Qwen3-1.7B | 2048 | 151,936 | 311 M | 3.083 | 528 M | **59%** | — |
> | Qwen2.5-1.5B | 1536 | 151,936 | 233 M | 2.312 | 552 M | **42%** | — |
> | SmolLM2-1.7B | 2048 | 49,152 | 101 M | 2.643 | 542 M | **19%** | — |
>
> **At 800 tokens of context the screen stops being about the head at all:**
>
> | donor | budget @800 | head as % |
> |---|---|---|
> | Qwen3-8B | 23 M | **2763%** |
> | OLMo-2-7B | 95 M | **432%** |
> | **Mistral-7B-v0.3** | 95 M | **141%** |
> | Phi-3-mini | 226 M | 44% |
>
> **Mistral-7B has the smallest vocabulary on the disk and still cannot reach 50 tok/s at 800
> context** — not because of its head, but because `f` has eaten the budget. Every 7–8 B dense donor
> is over the line at 800 context regardless of tokenizer.
>
> **Nine of the eighteen donors on disk are not priced by this fit at all** — MoE (`Qwen3-30B-A3B`,
> `Qwen3-Next-80B`, `OLMoE`, `DeepSeek-V2-Lite`, `Mixtral`, `gpt-oss-20b`, `granite-4.0-h`) and
> hybrid/SSM (`Zamba2`, `Nemotron-H`, `Falcon-H1`, `mamba2`) do not have uniform dense attention in
> every layer, so `f` at their shape has not been measured. **`gpt-oss-20b` at 105% and
> `Nemotron-H-8B` at 97% in the old table rested entirely on the analogy `donor_speed_budget.py`
> flagged as its own largest error, and are withdrawn rather than restated.**
>
> What survives from the old table unchanged: **the head's size is set by the tokenizer, not the
> model** — and E3 turned that from a weight count into an end-to-end measurement, `M7` vs `Q8`,
> 4.637× the vocabulary for 4.629× the head time (0.2%).

### 7.1 The original screen, kept for the record

`donor_speed_budget.py` prices the output head against the measured **27.7 G-weights/s** of the
packed kernel after the rope hoist (`SPEED_LEDGER.md` §12.2). The head is a dense GEMV of `D × V`
touched on **every** token, and **no** MoE, carve, sparsity or reconstruction result in this
programme touches it.

| donor | D | V | head | % of the 522 M budget for 10B @ 50 tok/s |
|---|---|---|---|---|
| Qwen3-8B | 4096 | 151,936 | 622 M | **112%** |
| openai/gpt-oss-20b | 2880 | 201,088 | 579 M | **105%** |
| Qwen2.5-Coder-7B | 3584 | 152,064 | 545 M | 98% |
| nvidia/Nemotron-H-8B | 4096 | 131,072 | 537 M | 97% |
| mistralai/Mistral-7B-v0.3 | 4096 | 32,768 | 134 M | **24%** |
| microsoft/Phi-3-mini | 3072 | 32,064 | 98 M | 18% |

**Two of twelve donors on this disk have an output head larger than the entire 50 tok/s budget,
and two more are within 3% of it.** Qwen3-8B cannot reach 50 tok/s on this machine even if every
other weight in it were free. It was four of twelve at the 495 M budget: **the entire 10.3% rope
win bought exactly two rows crossing back under the line.**

**Qwen3-8B and Mistral-7B are the same width.** Their heads differ by 4.6× entirely because of
vocabulary size. Every donor-adaptation probe this programme owns was measured on Qwen, which has
the worst head-to-body ratio on the disk. **Vocabulary is a speed variable, it is chosen rather
than earned, and it has never been treated as one.**

Two measured facts sharpen it: the head was 40.4% of every token in fp32 before R1 ternarized it
(`probes/R1_DONOR_RUNTIME.md` §3), and it is the one organ that pays **nothing** for int8
activations on the LUT path — 0.13997 → 0.14084 when excluded — because its input is the final
RMSNorm output (`donor_engine.c --lut-no-head`, commit `95b7fd3`).
