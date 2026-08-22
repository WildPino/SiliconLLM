# Controller re-review — stage −1 pre-registration, pass 2 (§7.2 pre-run)

**Target:** `docs/research/donor_adaptation/decisions/DONOR_ADAPTATION_FIRST_RESPONSE.md` §3 (rebuilt), §5 read for context.
**Prior pass:** `CONTROLLER_PREREG_REVIEW.md` — verdict NO, 7 BLOCKs, 6 FLAGs, 3 PASSes.
**Also read since:** `PTQ_SOURCE_VERIFICATION.md` (landed after pass 1; changed the method's premise),
`DONOR_ADAPTATION_STAGE0_DECISION.md` §7 (Owner ruling + hardware disclosure), `docs/CANONICAL_EVAL.md`,
mandate §6, §9 stage −1, S1, S4.
**Reviewer status:** read-only, protocol audit. No code exists.

**Amended mid-review:** the Owner approved a CUDA install and the RTX 3060 for stage −1, and §3.8 was
rewritten to pin device, dtype and determinism. That rewrite is audited in **N6**, which is the longest
section here because a device move on a determinism-gated project is where plausible artefacts live.

> **Verdict: NO — but narrowly, and the remaining fixes are desk edits plus two cheap probes.**
> 0 of 7 BLOCKs fully CLOSED, 5 PARTIALLY CLOSED, **2 STILL OPEN** (B1, B2). Two new BLOCKs (N6.2, N6.5);
> N1 downgraded and N3 largely closed by §3.8.
> The rebuild is a real improvement and it surfaced its own errors honestly. It is not yet runnable.

---

## Verdict per original BLOCK

### B1 — eval slice unpinned → **STILL OPEN**, and the rebuild argues the wrong thing

§3.6 says the metric is measured "on a slice pinned before anything is scored" and then justifies the
absolute gate with: *"Why absolute: it cannot be moved by slice choice, which closes B1 at the root
rather than patching it."*

**That claim is false, and it is the reason B1 did not get fixed.** The absolute gate removes the
*denominator* lever (headroom no longer scales with `BPB_donor`) — that half is genuine and it is a real
improvement. It does **not** make `ΔBPB` slice-invariant. Quantization damage is content-dependent, and
the project's own newly-verified evidence says so loudly: the *same* closed-form method moves from 1.79×
to 5.24× across evaluation targets (`PTQ_SOURCE_VERIFICATION.md` §1, Table 1). Prose vs code vs wiki will
not produce the same `ΔBPB` from the same converted weights. **A motivated person still satisfies a 0.10
gate by choosing the eval corpus** — they simply choose one where ternarization happens to hurt less,
instead of one where the donor scores high.

`grep -in "stride\|sha256\|BOS\|tokenizer\|bytes/token" §3` returns **zero hits**. Nothing from
CANONICAL_EVAL's standard is present: no corpus name/revision/licence, no byte range, no sha256, no token
and byte counts, no tokenizer repo *and revision*, no context length, **no stride**, no BOS or
final-window handling, no loss reduction, no decontamination procedure (only the word "Decontaminated",
applied to the calibration set, with no method). `BPB_donor` is still not pinned as a number.

Additionally new, and load-bearing: **bytes/token `B` on the pinned slice is never measured or reported.**
Every comparison of `Δ*` to the literature in §3.6 is a `log2(ratio)/B` conversion; `B` *is* the
conversion. Unmeasured, the gate's position relative to the published frontier swings by ±30%.

### B2 — gate shape → additive, **but the magnitude is STILL OPEN and its justification rests on a number this same document retracts**

The *shape* fix is correct and I accept it: `ΔBPB = BPB_conv − BPB_donor` in excess bits per byte is the
honest quantity. **The magnitude is not defensible as written, and the arithmetic is inverted.**

> §3.6: "Why 0.10: it sits **~3.5× above the candidate method's own published Δ (≈0.028** at the paper's
> scale)…"

`0.028 = log2(1.08)/4.0`. **1.08× is the fabricated ratio.** §3.3 of the *same document*, three
paragraphs earlier, strikes it: *"its real W1.58 result is **2.11× perplexity** at LLaMA-2-7B, not the
fabricated 1.04–1.08×."* §3.6 then calibrates the replacement gate on the withdrawn figure. This is the
same evidence-chain failure the rebuild exists to repair, reappearing one section later.

Redone on the **verified** table (PT²-LLM W1.58, LLaMA-2-7B, 5.47 → 11.56 = 2.113×):

| quantity | PPL ratio | ΔBPB @ B=4.0 | ΔBPB @ B=3.5…4.5 |
|---|---|---|---|
| QuaRot-class 4-bit (≤0.47 PPL on 5.47) | 1.086× | **0.030** | 0.026…0.034 |
| **Gate Δ\* = 0.10** | — | 0.100 | *implies* 1.27…1.37× |
| TWLA W1.58A16 @7B (**gradient-assisted, S1-ineligible**) | 1.27× | **0.086** | 0.076…0.098 |
| **PT²-LLM W1.58 @7B — the closed-form method to be run** | 2.113× | **0.270** | 0.240…0.308 |
| PT²-LLM W1.58 @LLaMA-3-8B | 5.24× | 0.598 | — |

**The gate is not 3.5× above the published Δ. It is ~2.7× BELOW it.** The sign is inverted. Concretely:
**PT²-LLM reproducing its own publication exactly, at 7B — its best-case scale — FAILS this gate**, and
every verified source says 1.5B is worse. Where 0.10 actually lands is on the *gradient-assisted* ternary
frontier (TWLA, 0.086), which S1 forbids as a primary path.

This may still be the right number — a gate only ternary-with-gradients could clear is a coherent choice
for a kill gate. **But it must be defended as that, in bits, against the verified table, with the
consequence stated: at ternary, FAIL is close to foreordained.** The sentence "it kills only what is
clearly dead" is not true of the method actually being run. `Δ*` also has no stated mapping to S4's
sealed ≥90% retention — so as it stands the number is anchored to neither the literature nor the mandate.

### B3 — control-1 anchor → **PARTIALLY CLOSED**

The 1a/1b split is the right structure and the UNVALIDATED-AGAINST-PUBLICATION declaration is exactly the
honest move. Four residuals:

1. **The tolerance is still not stated.** §3.7.1(1a) says "tolerance stated in advance as an absolute PPL
   band" — that is a promise to state it, not a statement. B3 defect #1 is verbatim unrepaired. A
   pre-registration containing "a number will be chosen" is post-hoc by construction (§6.1).
2. **1a presupposes an anchor nobody has shown exists.** "Reproduce a well-replicated published 4-bit
   number (GPTQ/QuaRot class) **on the chosen donor**" — the survey's QuaRot entry is LLaMA-2-70B. No
   published 4-bit WikiText2 row for `Qwen2.5-1.5B` is recorded in any artefact in this repo. The
   §3.2/§3.7.1 conflict is softened in wording, not resolved: **name the paper, table and row, or 1a is
   as unrunnable as v1 was.**
3. **The PPL↔BPB conversion is named but not pinned.** §3.7.1 correctly cites the E5-bis burn, then
   defers. Pin it in the document: `BPB = log2(PPL)/B`, with `B` = measured bytes/token on the pinned
   slice and PPL under the paper's own seqlen/stride convention (2048, non-overlapping), both emitted per
   arm. Also: **1b's practical blocker is unacknowledged** — LLaMA-2-7B is a gated-licence artefact and
   ~27 GB in fp32; "too large for the CPU budget" must not be a judgement made after the fact.
4. B3's requirement (b) — a **bridge** proving the project's BPB harness agrees with the published PPL
   harness on the *unquantized* donor — is answered with "the apparatus must emit both". Emitting both
   checks the arithmetic, not the harness. The cheap real bridge is reproducing a **published fp16
   baseline** (e.g. 5.47 on LLaMA-2-7B WikiText2) with the project's own loader and scorer.

### B4 — no σ → **PARTIALLY CLOSED.** The band is one-sided, and it leans toward PASS.

Replicates, a bootstrap SE and an INCONCLUSIVE band are now present, and deleting the σ_seed import is
correct. The arithmetic is not sound:

> "PASSES iff ΔBPB ≤ 0.10, FAILS iff ΔBPB ≥ 0.10 + 2·SE, INCONCLUSIVE in between."

The band is `[Δ*, Δ*+2·SE]` — **entirely on the FAIL side.** A true `Δ` sitting exactly on 0.10 PASSES
outright with ~50% probability; a result at 0.099 is a clean PASS with zero noise margin. B4 asked for
*within 2·SE of Δ\* is neither PASS nor FAIL* — two-sided. Correct form:
**PASS iff `Δ + 2·SE ≤ Δ*`; FAIL iff `Δ − 2·SE ≥ Δ*`; INCONCLUSIVE otherwise.** As written, every
borderline result resolves in the method's favour, which is the one direction a kill gate must not lean.

Three further defects in the replicate design:

- **The axes are confounded, and one of them contradicts §3.6.** "≥3 replicates varying the PTQ/rotation
  seed, **the calibration draw and the slice sample**" varies three things jointly (§6.2), and *varying
  the slice* directly contradicts "a slice pinned before anything is scored" — and it breaks pairing
  against a donor baseline measured once on the full slice. Slice resampling is the **bootstrap** (the SE
  estimator), not a replicate axis. Separate them.
- **`SE = σ_r√2` is the wrong estimator here.** `BPB_donor` is one fixed number carrying no PTQ variance,
  and slice-sampling error is *common-mode* across arms and largely cancels in a paired difference. Use a
  **paired bootstrap over the same eval windows** for the slice term and the replicate sd for the method
  term; do not sum two arms that are not independent.
- **Unhandled: what if σ_r = 0?** PT²-LLM as verified has *no learning rate and no documented RNG* (ITF is
  an alternating fit; AGA freezes the assignment and solves (α, μ) once). Seed replicates may return
  bit-identical BPB. Then `2·SE = 0`, the INCONCLUSIVE band collapses, and the gate is binary again — the
  exact state B4 rejected. Pre-register both readings (genuine determinism vs a dead seed path, §6.3) and
  an **SE floor** from the paired bootstrap. Note also that n=3 gives an sd on 2 degrees of freedom; the
  band's own width is then very poorly determined.
- The dead-guard fix (§3.5's "harness σ" now defined as the bootstrap SE) is accepted. **CLOSED.**

### B5 — lesion may be a known-negative → **PARTIALLY CLOSED (a sketch, not yet a protocol)**

The diagnosis and direction are right, and the self-report ("I designed a control that might never fire
and called it a planted control") is the standard §6.10 asks for. It is not operational: no perturbation
family (scaling vs Gaussian noise), no ε grid or rung count, no named non-redundant structure, no
monotonicity assertion, and — the original B5 point 2, still unaddressed — **no pre-registered response if
the ladder fails to reach Δ\* within the grid, or comes back non-monotone.** An operator handed this must
invent the instrument. One paragraph fixes it.

### B6 — calibration-influence control → **PARTIALLY CLOSED**

Control 5 (random-vocabulary calibration) is the right control and is correctly argued against the
disjoint-set substitute. But the pre-registered expectation covers only the **bit-identical** case. B6
asked for a quantitative threshold: **`|BPB(random calib) − BPB(real calib)| ≥ k·SE`, with `k` stated
now.** Bit-identity is a necessary check, not a sufficient one — a scale that is computed and then almost
entirely overwritten yields a difference of 1e-6, a dead path this control as written would pass. State
the **expected direction** too (random-vocabulary calibration should be *worse*); a better-than-real
result is an apparatus finding, not a datum. Log both directions per §6.4.

### B7 — sweep is not one variable → **PARTIALLY CLOSED**, and §3.4 was never edited

§3.7b is good on the committed config table, constant group size and outlier policy, and effective
bits/weight. Two required items were dropped silently, and both live in §3.4, which is **unchanged from
v1**:

- **§3.4 still reads "One variable, five points"** while §3.7b says the sweep is not one variable. The two
  sections now contradict each other inside one document. fp16 must be labelled the **reference**, not a
  curve point ("no quantizer" is a different arm by definition).
- **The overclaim survives verbatim:** *"the curve answers the question the Owner will actually ask — which
  bit-width would the engine have to grow to?"* It cannot; that needs engine rate and footprint per
  bit-width, which do not exist, and which §2.3 / Controller #2 show is counter-intuitive on this silicon
  (**the LUT path is compute-bound by ~16×**, so bits buy footprint far more than they buy speed). The
  curve gives quality-vs-bits only. This matters *more* after the re-scope, not less — see N2.

Also missing from B7's required list: quantizer levels, symmetric/affine, scale dtype and count,
clipping/search procedure, per-layer fitted-parameter count, RNG seed.

---

## New findings

### N1 — **BLOCK.** Nothing has shown the method's implementation runs CPU-only, and the install ask does not include it

§3.8 asks for `pip install datasets accelerate` on `torch 2.12.0+cpu`. **The PT²-LLM implementation is not
in that list**, and §3.7b / §6.5 require reading the per-point config *from the implementation* — so it
must be obtained regardless. The verification records the method as developed and timed on a single
**A800-80GB**; rotation-PTQ codebases of this family (QuaRot, QuIP#, SpinQuant, PT²-LLM) routinely depend
on **CUDA-only fused Hadamard / GPTQ kernels** (`fast-hadamard-transform` and relatives) with no CPU path.
Nobody has checked this, and the failure lands **after** the owner's machine is committed.

**Required before the 12–20 h run:** a **sub-hour CPU smoke test** — obtain the implementation at a pinned
commit, quantize **one** layer of the donor end-to-end on CPU, and report the resolved code path. If it
needs CUDA, stage −1 as costed does not exist and the device question (N3) becomes primary rather than
optional. This is the cheapest item on this page and the most expensive to discover at hour 11.

### N2 — The §3.6 gate is the wrong instrument for the re-scoped §3.3 purpose

§3.3 re-scopes stage −1 from "does ternary work" to **"locate the knee"**. §3.6's gate was not re-scoped
with it: a single threshold applied per point yields five pass/fails, which brackets the knee between two
of five widely spaced points and cannot locate it. On the verified numbers the outcome is already close to
determined — 4-bit ≈0.030 PASS, ternary ≈0.27 FAIL — so the gate returns roughly one bit of information
the literature already supplied, for 12–20 h of the owner's CPU.

**What should read a curve instead** (and this is the deliverable that earns the machine time):
`ΔBPB` plotted against **effective bits/weight including scales**, with per-point 2·SE bars from §3.6b,
plus the first difference between adjacent points; the reported result is **the lowest bit-width whose
upper 2·SE bound is ≤ Δ\***, stated together with the two neighbouring points that bracket it. Keep Δ\* as
the readout line — but pre-register it as a **curve-reading rule**, not a pass/fail on ternary, and state
explicitly that a knee in *quality* is not a knee in *engine feasibility* (N2 ↔ B7's overclaim).

### N3 — **Device-dependent result paths. FLAG, with one BLOCK-grade consequence.**

Hardware disclosed: Ryzen 5 3600X (Zen 2, AVX2, no AVX-512/VNNI), 80 GB DDR4, RTX 3060 (Ampere) + GTX 1660
(Turing); installed torch is **`2.12.0+cpu`, `cuda False`**. The pre-registration is device-silent:
`grep -in "torch_dtype\|thread\|deterministic"` over §3 returns **zero hits**. Paths that would differ on a
device move:

1. **dtype (F3, unrepaired and now worse).** §3.4 still says "fp16 baseline". On a CPU-only build,
   `transformers` loads fp32 unless `torch_dtype` is passed explicitly, and CPU fp16 matmul is emulated
   where it exists at all — **"the fp16 baseline" is plausibly an fp32 baseline**, a different number. On
   GPU it would be a genuine fp16 number; the RTX 3060 has bf16, **the GTX 1660 (Turing) does not**, so
   dtype policy is per-device. CANONICAL_EVAL pins *fp32, CPU, no autocast* precisely for this, and
   quantifies bf16-autocast eval noise at ~1e-4 — negligible against Δ\* = 0.10, **not** negligible
   against a 2·SE band if σ_r comes back small (B4).
2. **Determinism, and it breaks control 3.** Control 3 ("same weights twice → bit-identical BPB") holds on
   single-threaded CPU. Under cuBLAS split-k / atomic reductions it can fail *for reasons unrelated to the
   weights* — the comparator raises a false alarm, and the live risk is that it gets quietly relaxed to a
   tolerance mid-run. Declare `torch.set_num_threads(N)` with N pinned,
   `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG` if CUDA ever appears, and run
   control 3 **across process restarts** (pass-1 P2's addition, not yet incorporated).
3. **Replicate variance does not transfer.** CPU and CUDA RNG streams differ; a σ_r measured on CPU is not
   the σ_r of a GPU run, so the INCONCLUSIVE band is device-scoped.
4. **The binding rule.** CANONICAL_EVAL: *"A pre-registered gate is read with the same harness that
   produced its anchor — never across two harnesses."* **Device + dtype + thread count are part of harness
   identity.** Pre-register now: if the environment moves to CUDA, `BPB_donor` and **every** arm is re-run
   on the new harness; a GPU converted-arm may never be differenced against a CPU donor baseline. Without
   that sentence, the tempting mid-run optimisation ("the 3060 would cut this to 1–2 h") silently produces
   a cross-harness gate. §7.3 of the stage-0 decision flags the intent; the pre-registration must carry the
   rule.

### N4 — Original FLAGs: 2 half-addressed, 4 silently dropped

| flag | status |
|---|---|
| **F1** calibration 512 vs the paper's 128 | **DROPPED.** §3.5 still seals 512, still unmotivated; §3.7.1's anchors must run at the paper's **128** or they do not reproduce the published number. State the split (anchor @128, gate @512) and report both. |
| **F2** "composition mixed" is not a specification | **DROPPED.** §3.5 unchanged: no source datasets, proportions by token, sampling script, seed or shuffle procedure; the *second disjoint* calibration set's decontamination is still unstated. |
| **F3** `torch_dtype` hidden default | **DROPPED**, and promoted — see N3.1. |
| **F4** control 4 never exercised | **HALF.** §3.7.4 now says "must be exercised in both directions and logged" — the principle is in. Still missing the specifics: which architectural field is mutated (`num_key_value_heads` / `num_hidden_layers` / `tie_word_embeddings`) and the **architecture-sensitive numerical probe** (fixed-prompt fp32 logits hash recorded in the manifest before the sweep, asserted at every arm's load). Qwen2.5 at this scale ties embeddings; a silent untie moves BPB while `strict=True` stays happy. |
| **F5** FAIL side over-read | **HALF, and correctly handled where it was done.** §3.6 is now scoped to *this bounded method, bit-width and donor*, matches §3.9 and the mandate's §9 wording, and the disagreement with the Owner's stronger phrasing is **flagged rather than resolved silently** — that is the right move and I endorse it. But **§3.1 is unedited: "a clear failure makes the ternary-primary route moot."** §3.1 and §3.6 now contradict each other. One-line fix. |
| **F6** dense-donor → MoE-target transfer, per-expert calibration coverage | **DROPPED.** `grep -i expert` over §3 returns zero hits. §3.6's scope paragraph still does not state that a PASS licenses nothing about a sparse-MoE target, where 1.05M calibration tokens spread over E experts at top-k give each expert ~k/E of the data and the tail far less. |

### N6 — Audit of the rewritten §3.8 (GPU execution environment)

**Overall:** pinning the environment in the pre-registration is the right instinct and it closes most of
N3. But §3.8 pins the *wrong half* of the numerics. Its centrepiece — TF32 — is largely inert under its
own bf16 dtype policy, while **the defaults that actually bite under bf16 are not mentioned**, and its
claim that control 3 checks the whole policy is false in the direction that matters. Two BLOCKs below.

#### N6.1 — Device pinning: the name assertion is the weakest of the available checks. **FLAG.**

`CUDA_VISIBLE_DEVICES` "set explicitly" does not say **to what**, and that matters: `CUDA_DEVICE_ORDER`
defaults to `FASTEST_FIRST`, a *heuristic*, not a stable identity — so index `0` is exactly the thing the
§3.8 text worries about. Ways to still land wrong:

- `device_map="auto"` (and `accelerate` is in the install list) will **shard across every visible device**;
  with both GPUs visible it can place layers on the 1660 while `get_device_name(0)` still says 3060 and
  the assertion passes. This is a live path, not a hypothetical.
- A subprocess, notebook kernel or launcher that resets the environment; a driver update that reorders.
- `get_device_name` is a marketing string: "NVIDIA GeForce RTX 3060" covers the 12 GB GA106 and the 8 GB
  GA104 part, and a laptop variant, and the string itself has changed across driver generations. It also
  pins none of the things that determine the numbers — compute capability, CUDA/cuBLAS/cuDNN versions.

**Required, and it is strictly cheaper than what is written:**
`CUDA_DEVICE_ORDER=PCI_BUS_ID` **and** `CUDA_VISIBLE_DEVICES=GPU-<uuid>` (CUDA accepts a GPU UUID) — this
is the project's own law 9 applied to hardware: *a tag is not a reproducible specification; a hash is.*
Then assert, in this order: **`torch.cuda.device_count() == 1`** (the masking worked — this single check
subsumes most of the risk), `get_device_capability(0) == (8, 6)`, the device UUID from
`get_device_properties(0)`, and **the device of an actually-materialised parameter**
(`{p.device for p in model.parameters()} == {cuda:0}`) — law 5, the artefact is the authority, not the
environment variable. Forbid `device_map="auto"` explicitly. Record CUDA, cuDNN, driver and torch build
strings in the manifest: they are part of harness identity as much as the die is.

#### N6.2 — **BLOCK.** The TF32 reasoning is inverted, and the defaults that actually bite under bf16 are missing.

> §3.8: "**TF32 is enabled by default on Ampere for matmul** and silently truncates the mantissa."

Two problems, and the second is the serious one.

**(a) The stated default is wrong for modern PyTorch.** `torch.backends.cuda.matmul.allow_tf32` has
defaulted to **False** since torch 1.12 (it was True only in the 1.7–1.11 window); it is
`torch.backends.cudnn.allow_tf32` that defaults to **True**, and that governs convolutions, of which a
decoder-only transformer has none. Recent torch also **deprecated** `allow_tf32` in favour of
`torch.backends.cuda.matmul.fp32_precision` / `torch.backends.fp32_precision` — so on a 2.12 build the
line as written may set a deprecated flag, warn, and not be the authoritative knob. Setting both to False
is harmless and I do not object to the pin; **the justification is false, and a pin justified by a false
premise is one refactor away from being removed as redundant.** I cannot verify 2.12's defaults from here,
and that is the point: *do not assert a default, read it back and probe it* (N6.3).

**(b) TF32 is mostly inert under §3.8's own dtype policy, and the live knobs are unlisted.** TF32 governs
**fp32** matmul inputs. §3.8 loads the donor in **bf16**, so the transformer's matmuls are bf16 and TF32
never touches them. What does touch them, all defaulting to permissive:

| default | effect | why it matters here |
|---|---|---|
| `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` (**True**) | split-k GEMM accumulates in reduced precision | **This is the one.** With bf16 weights it is the direct analogue of the TF32 trap and it is on. Set **False**. |
| `allow_fp16_reduced_precision_reduction` (**True**) | same, fp16 path | set False for the anchor arms, which the papers run in fp16 |
| `torch.set_float32_matmul_precision` (`"highest"`) | global alias that can re-enable TF32 | pin to `"highest"`; a library setting `"high"` silently overrides your flag |
| **SDPA backend selection** (heuristic) | flash / mem-efficient / math backends round *differently* and are chosen by shape, dtype and version | `attn_implementation` must be pinned (`"eager"` for the reference arm) and the **resolved** value read back from the loaded model; unpinned this is a silent algorithm change across arms |
| `torch.backends.cuda.preferred_linalg_library` | cusolver vs magma | **GPTQ-family layer solves use `linalg.cholesky` / `cholesky_inverse` on ill-conditioned Hessians.** Pin the library *and* the damping factor, and put both in the B7 config table |
| `torch.backends.cudnn.benchmark` | autotuned algorithm choice varies | pin False |
| implicit autocast (`accelerate`, Trainer wrappers) | silent dtype demotion | assert `torch.is_autocast_enabled() is False` inside the scoring loop |

Also: `use_deterministic_algorithms(True)` **raises** on ops lacking a deterministic kernel. Pre-register
that `warn_only` stays **False** and that a raise is a *finding to report*, not friction to route around —
otherwise the first inconvenient op turns the guarantee into a warning nobody reads.

#### N6.3 — **The determinism claim is checkable; the precision claim is not, and control 3 is a weaker test on GPU.** BLOCK-adjacent.

> §3.8: "control 3 … is precisely the guard that tests this whole policy … It is not a separate promise;
> it is checkable."

**Half true, and the false half is the dangerous one.** Nondeterminism splits in two:

- **Run-to-run variation** — atomics, per-launch split-k selection, cuDNN autotuning. Control 3 catches
  these. Genuine gain, and it is the reason the guard is worth keeping.
- **Deterministic-but-degraded** — TF32, reduced-precision reductions, a flash-attention backend, an
  implicit autocast. Every one of these is **perfectly repeatable**. Control 3 returns bit-identical
  output while the number sits at an 8- or 10-bit mantissa. **This is textbook §6.3 — the plausible
  artefact — and §3.8 nominates as its guard the one control that is structurally blind to it.**

Control 3 is also weaker on GPU in a third way: within one process it shares a stream, a workspace and a
cached autotune decision, so it under-samples the space it claims to cover. Run it **across process
restarts** and **across two batch shapes** (batch 1 vs batch N): a BPB that moves with batch size is the
classic reduction-order tell, and it is a correctness question for a metric compared across arms.

**What would actually test the precision half — two cheap probes, both pre-registerable:**
1. **A TF32/reduced-reduction probe.** One fp32 matmul of known conditioning against a **float64** CPU
   reference: true fp32 lands at ~1e-7 relative error, TF32 at ~1e-3. Same trick in bf16 for the
   reduced-precision-reduction flag. This *fires on a known-positive* (deliberately enable the flag and
   watch it break) — a planted control for the precision policy, which is what §6.3 asks for and what a
   list of flag assignments is not.
2. **Read every flag back after setting it** and record the resolved values, the SDPA backend actually
   selected, and `attn_implementation` from the loaded model, in the manifest — never from the argument
   (law 5). The stage-0 ruling already has the Builder validating against a float64 reference; use the
   same standard here.

#### N6.4 — Control 6 (device parity): right instinct, wrong comparison, and here is the tolerance derivation

Asked directly: **is CPU-vs-GPU parity the right check?** Partly. It is *not* a determinism check and
*not* a correctness check — two different kernel families in two different rounding orders will disagree,
and the disagreement is uninformative on its own. As specified it has three defects:

1. **It compares the wrong quantity.** The gate reads `ΔBPB`, a difference *within* one harness, so a
   common-mode CPU/GPU offset largely cancels and an absolute-BPB comparison is both too strict and beside
   the point. **Compare `ΔBPB_cpu` against `ΔBPB_gpu`** — stricter where it matters, forgiving where it
   does not, and it is the quantity that decides.
2. **The dtype legs are not equal.** Under §3.8's policy the CPU leg would run **bf16**, which on CPU is
   emulated with different accumulation — the control would mostly measure the emulation. Specify the CPU
   leg as an **fp32 (or fp64) oracle** and state that the expected gap is the bf16 weight quantization,
   which on a 1.5B model is **not** the ~1e-4 that CANONICAL_EVAL measured for bf16 *autocast* on an 8.3M
   model. Predict sign and magnitude in advance or the control cannot fail.
3. **It covers only the donor path.** A GPU-specific bug in the *quantizer's own* CUDA kernels is the one
   that would void the stage. Run control 6 on the donor arm **and at least one converted arm.**

**The tolerance, derived rather than chosen** (and it must be written down before the numbers exist):
anchor it to the smallest quantity the protocol has to resolve, which is the INCONCLUSIVE half-width.

> **τ = 0.1 × (the pre-registered band half-width) on `|ΔBPB_gpu − ΔBPB_cpu|`.**

That is the same rule this repo already used and already justified: CANONICAL_EVAL accepted its
bf16-autocast noise precisely because it was *"an order of magnitude below every probe delta — no verdict
moves."* An order of magnitude below the statistical resolution means the numerical environment can never
flip a verdict, which is the only property a parity tolerance needs. It also has the right dependency
order: the band comes from the replicate/bootstrap step (§3.6b), which runs before any converted arm is
scored, so τ is a *formula now* and a *number before it can be gamed*. State both.

#### N6.5 — **BLOCK.** The bf16 dtype policy is defensible for the donor and creates two new confounds

The law-5 argument ("load at the dtype the artefact declares") is right in spirit. Three holes:

1. **§3.4 still says "fp16 baseline"; §3.8 says bf16.** That is the **third** internal contradiction in
   this document (with §3.1 vs §3.6, and §3.4 vs §3.7b). Resolve it in §3.4.
2. **It breaks the published anchors (interacts with B3/F1).** QuaRot/GPTQ/PT²-LLM baselines are **fp16**.
   bf16 has 8 mantissa bits against fp16's 11 — on a 1.5B model that is a real, measurable offset. Either
   run the anchor arms in the papers' fp16 or quantify the bf16↔fp16 offset first; do not compare across
   it silently.
3. **It can bend the bit-width curve — a fresh instance of B7.** Rotation-PTQ pipelines build Hessians,
   Hadamard rotations and Cholesky solves. In bf16 those accumulations are materially worse than fp16 and
   far worse than fp32, and layer-wise error compensation does **more** work as bits drop — so a
   bf16-native quantizer degrades *preferentially at the low-bit end*, exactly where the knee is being
   located. The ternary threshold rule is the sharpest case: absmean/λ ternarization is a *comparison*,
   and 8 mantissa bits can flip near-threshold weights' assignment in a way that has no analogue at 4-bit.
   **Required: pin the quantizer's internal working precision (Hessian, rotation, solve, scale fitting) at
   fp32 or fp64, independently of the stored weight dtype, hold it constant across all five points, and
   record it — plus the scale dtype, which B7 already needs for effective bits/weight — in the per-point
   config table.** Otherwise the curve is two-variable again and the second variable is correlated with
   the axis.

#### N6.6 — Effect of the GPU move on the seven BLOCKs, and the right replicate count

- **B1, B2, B6** — unchanged, device-independent. B1 and B2 remain the blocking pair.
- **B3** — *feasibility improves and the escape hatch narrows.* 1b's "too large for the CPU budget"
  declaration should be re-examined: the donor arm is trivial on 12 GB (1.5B bf16 ≈ 3.1 GB), and 7B bf16
  ≈ 13.5 GB does **not** fit the 3060 but is reachable with sequential offload against 80 GB of host RAM.
  UNVALIDATED-AGAINST-PUBLICATION must not be reached by default now. (The licence-gating of LLaMA-2
  remains, and is still unacknowledged.) Note the new 12 GB VRAM ceiling is itself unstated in §3.8.
- **B4 — this is where the GPU changes the answer most, and there is an arithmetic error to fix first.**
  The band is written as `2·SE`. With **n = 3**, a normal multiplier is wrong: the t-multiplier is
  **t₀.₉₇₅,₂ = 4.30**, so "2·SE" **understates the band by ~2.15×** — and the sd itself sits on 2 degrees of
  freedom (its own 95% interval spans roughly 0.5σ–6σ). Also state explicitly that `SE = sd/√n`, not sd.
  At ~1–2 h a sweep, n = 3 is no longer a budget-forced floor. **Right numbers: n ≥ 10 at the decision
  point and at the two points bracketing the knee** (t = 2.26, sd on 9 df, and a bootstrap that means
  something), n ≥ 3 elsewhere. **But note which axis:** PT²-LLM as verified has no learning rate and no
  documented RNG, so "PTQ seed" replicates may be degenerate — the live axis is the **calibration draw**,
  and that is the one to replicate 10×. If seed replicates come back bit-identical, report it as
  determinism *and* check it is not a dead seed path (§6.3).
- **B5** — the ladder gets cheap: ≥6 geometric rungs rather than a token few. No excuse for a coarse grid.
- **B7** — **worse**, via N6.5.3. New confound, new requirement.
- **N1** — **downgraded from BLOCK to a required pre-run smoke test.** CUDA is this method family's native
  environment, so the CUDA-only-Hadamard-kernel risk largely evaporates; the install grows (CUDA torch +
  the method's own repo, still absent from §3.8's `pip` line) and one-layer-end-to-end before committing
  the machine is still the correct gate — it is just now expected to pass.
- **N3** — **largely CLOSED by §3.8**, and credited. The cross-harness rule survives and is now *live*
  rather than hypothetical: `BPB_donor` and every converted arm must share device, dtype, flags and
  library versions, and any CPU-era reasoning is not comparable to the new harness.
- **Budget:** "~1–2 h on the 3060" is an estimate with no derivation, and the replicate count multiplies
  it. At n = 10 on the live axis it is not 1–2 h. Recompute once n is set, and state it as a per-arm cost
  times the arm count, not a single number.

### N5 — Retained PASSes, and what genuinely improved

P1 (PASS-side scope disclaimer), P2 (control 3) and P3 (refusal to start) all stand. Genuinely better in
v2: the additive metric shape, the 1a/1b anchor split with the UNVALIDATED declaration, the σ_seed
deletion, the lesion-ladder replacement, control 5, §3.7b, and the two self-reported failures quoted in
full in §3.6. Added by the §3.8 rewrite: pinning the environment *in the pre-registration* rather than
letting the device drift in mid-run, excluding the Turing card by policy, fp32 loss accumulation
regardless of weight dtype, an identical dtype policy across arms, and control 6 existing at all — the
instinct to keep the CPU path as an oracle is right even though the comparison needs re-specifying.
**The document argues against itself in the right places now.** That is why this verdict is "not yet"
rather than "no".

---

## Verdict

**May this pre-registration proceed to execution? NO.** Two original BLOCKs stand open (B1, B2), two new
ones are raised against the §3.8 rewrite (N6.2 silent-precision defaults, N6.5 bf16 confounds), and B4's
band both leans the wrong way and uses the wrong multiplier. Every fix below is a desk edit except the
smoke test and the precision probe, which are under an hour together. **None requires re-thinking the
stage, and the GPU approval makes several of them cheaper rather than harder.**

### Minimal remaining fix list

1. **(B1)** Pin the eval slice to CANONICAL_EVAL's standard — corpus/revision/licence, byte range, sha256,
   byte and token counts, tokenizer repo **and revision**, context length **and stride**, BOS and
   final-window handling, loss reduction, decontamination procedure — **measure and report bytes/token
   `B`**, and **pin `BPB_donor` as a number** before any converted arm runs. Delete the claim that an
   absolute gate is slice-invariant; it is not.
2. **(B2)** Re-derive Δ\* against the **verified** table, not the retracted 1.08×. State that 0.10 b/byte
   ⇒ ~1.32× PPL at B = 4, that PT²-LLM's *published* ternary result is ≈0.27 b/byte, and therefore that the
   gate rejects the closed-form method at its own published quality. Defend that as the intended kill line,
   or move the number — but the justification sentence as written is false.
3. **(B4)** Make the band two-sided — **PASS iff `Δ + h ≤ Δ*`**, FAIL iff `Δ − h ≥ Δ*`, else INCONCLUSIVE
   — with `h = t₀.₉₇₅,ₙ₋₁ · SE` and `SE = sd/√n`, **not** `2·SE` (at n = 3 the multiplier is 4.30, so the
   written band is ~2.15× too narrow). Raise **n to ≥ 10 on the calibration-draw axis** at the decision
   point and the two points bracketing the knee, now that a run costs ~1–2 h rather than 12–20. Separate
   the replicate axes from the **paired** bootstrap over eval windows; drop "slice sample" as a replicate
   axis. Pre-register the σ_r = 0 case with an SE floor.
4. **(B3)** State control 1a's **numeric tolerance now**, and **name the paper, table and row** of the
   4-bit anchor for this donor — or change the donor. Pin `BPB = log2(PPL)/B` with the paper's seqlen and
   stride convention, and add the fp16-baseline reproduction as the harness bridge.
5. **(N1, downgraded)** Run the **sub-hour smoke test** on the pinned implementation commit — one layer,
   end-to-end, resolved code path reported — before the machine is committed. Add the method's own repo
   and pinned commit to §3.8's install line; it is required by §3.7b regardless.
6. **(N6.2)** Fix the TF32 justification (matmul TF32 has defaulted to False since torch 1.12; `allow_tf32`
   is deprecated in recent builds) and **add the defaults that actually bite under bf16**:
   `allow_bf16_reduced_precision_reduction=False`, `allow_fp16_reduced_precision_reduction=False`,
   `set_float32_matmul_precision("highest")`, a pinned `attn_implementation` / SDPA backend read back from
   the loaded model, a pinned `preferred_linalg_library` and Cholesky damping, `cudnn.benchmark=False`, an
   autocast assertion in the scoring loop, and `warn_only=False` on `use_deterministic_algorithms` with a
   raise pre-registered as a finding.
7. **(N6.3)** Add the **precision probe** — an fp32 matmul against a float64 reference (~1e-7 true fp32 vs
   ~1e-3 TF32), exercised in both directions by deliberately enabling the flag. Control 3 tests
   repeatability, not precision, and cannot substitute. Run control 3 across process restarts **and** two
   batch shapes. Read every flag back into the manifest rather than asserting the default.
8. **(N6.1)** Pin the device by **UUID** with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, assert
   `device_count() == 1`, capability `(8, 6)`, the UUID, and the device of a materialised parameter;
   forbid `device_map="auto"`; record CUDA/cuDNN/driver/torch build strings as harness identity.
9. **(N6.4)** Restate control 6 on **`|ΔBPB_gpu − ΔBPB_cpu|`** with an **fp32/fp64 CPU oracle** (not a
   bf16 CPU leg), run on the donor arm and one converted arm, with the tolerance written now as
   **τ = 0.1 × the band half-width** — an order of magnitude below the statistical resolution, the same
   rule CANONICAL_EVAL already used and justified.
10. **(N6.5)** Resolve §3.4's "fp16" against §3.8's bf16; run the published anchors at the papers' fp16 or
    quantify the bf16↔fp16 offset; and **pin the quantizer's internal working precision at fp32/fp64,
    constant across all five points**, recorded with the scale dtype in the B7 config table — otherwise
    bf16 degrades the low-bit end preferentially and bends the very knee being located.
11. **(B5/B6/B7)** One paragraph each: the ladder's perturbation family, ε grid, named non-redundant rung
   and its failure-to-fire response; control 5's numeric `k·SE` threshold and expected direction; and
   **edit §3.4** — fp16 is the reference not a curve point, "one variable, five points" is withdrawn, and
   the "which bit-width would the engine grow to" claim is dropped or paired with engine rates that do not
   exist.
12. **(N2)** Re-scope §3.6 from a ternary pass/fail to a **curve-reading rule** matching §3.3's knee
   purpose: ΔBPB vs effective bits/weight with 2·SE bars; report the lowest bit-width whose upper bound
   clears Δ\*; state that a quality knee is not an engine-feasibility knee.
13. **(N4)** Reinstate F1, F2 and F6 or record them as accepted residual risk under §7.2 with the reasoning;
   complete F4's specifics; bring **§3.1** to §3.6's scoped wording.

**Blocking set: 1, 2, 3, 4, 6, 7, 10** — the two open originals, the band arithmetic, the anchor,
and the three §3.8 precision items. **5, 8, 9** are the cheap pre-run exercises. **11–13** are tidy-up
that shipping without is a choice, not a constraint.

Re-review after the edits. The post-run pass against the recorded evidence is separate, and on this
protocol the item I will look at first is whether the flags in §3.8 were *read back* or merely *set*.
